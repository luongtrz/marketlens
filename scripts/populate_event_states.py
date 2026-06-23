"""Populate event_state for all BTC records in Supabase using EventExtractor.

For each record:
  - Rule-based taxonomy lookup first (fast, no cost)
  - Keyword fallback second
  - Gemini Flash LLM third (only when < 2 events found, ~720 records)

Then PATCHes event_state back into the Supabase payload so subsequent
regen_optimizer_rest.py runs produce dense event_vec.

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/populate_event_states.py
    PYTHONPATH=/home/luong/marketlens python scripts/populate_event_states.py --dry-run
    PYTHONPATH=/home/luong/marketlens python scripts/populate_event_states.py --llm-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for env_file in [ROOT / ".env", ROOT / "aihub" / ".env"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env.setdefault(k.strip(), v.strip())
    env.update(os.environ)
    return env

_ENV = _load_env()
SUPABASE_URL = _ENV.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = _ENV.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _sb_headers() -> dict[str, str]:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(symbol: str, dry_run: bool, llm_only: bool) -> None:
    from aihub.src.events.extractor import EventExtractor
    from aihub.src.events.schema import EventExtractionRequest
    from aihub.src.llm.groq import GroqClient
    from stockmem.src.models import StockMemRecord
    from stockmem.src.search.event_memory import build_daily_event_state

    # LLM client — use llama-3.1-8b-instant for high rate limits on bulk processing
    groq_key = _ENV.get("AIHUB_GROQ_API_KEY", "")
    groq_model = "llama-3.1-8b-instant"  # high RPM; gpt-oss-120b throttles on bulk
    llm: GroqClient | None = None
    if groq_key:
        llm = GroqClient(api_key=groq_key, model=groq_model)
        log.info("LLM enabled: groq/%s", groq_model)
    else:
        log.warning("AIHUB_GROQ_API_KEY not set — LLM tier disabled")

    extractor = EventExtractor(llm=llm)

    async with httpx.AsyncClient() as client:
        # --- Paginated fetch ---
        log.info("Fetching %s records from Supabase...", symbol)
        all_rows: list[tuple[str, date, dict, StockMemRecord | None]] = []
        limit, offset = 1000, 0
        while True:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/stockmem_records",
                headers={**_sb_headers(), "Prefer": ""},
                params={"select": "id,record_date,payload", "symbol": f"eq.{symbol.upper()}",
                        "order": "record_date.asc", "limit": str(limit), "offset": str(offset)},
                timeout=60,
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            for row in batch:
                payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
                rec_date = date.fromisoformat(row["record_date"])
                try:
                    rec = StockMemRecord.model_validate(payload)
                    all_rows.append((row["id"], rec_date, payload, rec))
                except Exception as exc:
                    log.warning("Skip %s: %s", rec_date, exc)
                    all_rows.append((row["id"], rec_date, payload, None))
            log.info("  fetched %d records", len(all_rows))
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.2)

        log.info("Total: %d rows", len(all_rows))

        # --- Process each record ---
        updated = skipped = errors = already_ok = 0
        all_parsed_recs = [r for _, _, _, r in all_rows if r is not None]

        for i, (rid, rec_date, payload, rec) in enumerate(all_rows):
            if rec is None:
                skipped += 1
                continue

            # Check current event state
            current_events = []
            if rec.event_state and rec.event_state.events:
                current_events = rec.event_state.events

            has_events = len(current_events) >= 2
            if llm_only and has_events:
                already_ok += 1
                continue
            if not llm_only and has_events and not any(e.polarity != 0 for e in current_events):
                # Has events but all polarity=0 (generic mapping) — try to improve
                pass
            elif has_events:
                already_ok += 1
                continue

            # Extract events from raw factors + summary
            factors = rec.factors or []
            summary = payload.get("summary", "") or ""

            req = EventExtractionRequest(
                symbol=symbol,
                title=summary[:300] if summary else (factors[0] if factors else ""),
                summary=summary[:1000] if summary else None,
                factors=factors,
            )

            try:
                resp = await extractor.extract(req)
                new_events = resp.events
                if resp.method == "llm":
                    await asyncio.sleep(0.15)  # pace LLM calls ~6/s
            except Exception as exc:
                log.warning("Extraction failed %s: %s", rec_date, exc)
                errors += 1
                continue

            if not new_events:
                skipped += 1
                continue

            # Build DailyEventState
            history_recs = [r for _, _, _, r in all_rows[:i] if r is not None]
            try:
                es = build_daily_event_state(rec, history_recs)
                es.events = new_events  # type: ignore[assignment]
            except Exception:
                from shared.models.event import DailyEventState
                es = DailyEventState(
                    date=rec_date, symbol=symbol, events=new_events,
                    article_count=len(rec.article_ids or []),
                    source_count=len(set(rec.article_sources or [])),
                )

            # PATCH back to Supabase
            payload["event_state"] = es.model_dump(mode="json")
            log.info("  %s → %d events [%s] (method=%s)",
                     rec_date, len(new_events),
                     ", ".join(f"{e.event_group}/{e.event_type}" for e in new_events[:2]),
                     resp.method)

            if not dry_run:
                patch_r = await client.patch(
                    f"{SUPABASE_URL}/rest/v1/stockmem_records",
                    headers=_sb_headers(),
                    params={"id": f"eq.{rid}"},
                    content=json.dumps({"payload": json.dumps(payload, ensure_ascii=False)}),
                    timeout=15,
                )
                if patch_r.status_code not in (200, 204):
                    log.warning("PATCH failed %s: %s", rec_date, patch_r.text[:100])
                    errors += 1
                    continue
                await asyncio.sleep(0.05)  # rate limit

            updated += 1

            if (i + 1) % 50 == 0:
                log.info("  progress: %d/%d (updated=%d skipped=%d already_ok=%d errors=%d)",
                         i + 1, len(all_rows), updated, skipped, already_ok, errors)

    log.info("Done. updated=%d already_ok=%d skipped=%d errors=%d",
             updated, already_ok, skipped, errors)
    if dry_run:
        log.info("[dry-run] No changes written to Supabase.")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Populate event_state in Supabase via EventExtractor + LLM")
    p.add_argument("--symbol",   default="BTC")
    p.add_argument("--dry-run",  action="store_true", help="Don't write to Supabase")
    p.add_argument("--llm-only", action="store_true", help="Only process records that still have < 2 events")
    args = p.parse_args()
    asyncio.run(main(symbol=args.symbol, dry_run=args.dry_run, llm_only=args.llm_only))


if __name__ == "__main__":
    _cli()
