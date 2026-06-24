"""Seed local stockmem service from Supabase stockmem_records.

Pipeline per record:
  1. Fetch full StockMemRecord payload from Supabase.
  2. If event_state missing/sparse, extract via EventExtractor:
       - rulebase first (always)
       - Gemini flash-lite enhancement if < 2 rulebase events
  3. Fill missing d3/d15 future returns from Binance.
  4. POST to local stockmem service (localhost:8003/record).
     Skips records already stored (idempotent via 409 handling).

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/seed_stockmem_from_supabase.py
    # Custom service URL / symbol:
    STOCKMEM_URL=http://localhost:8003 SYMBOL=BTC python scripts/seed_stockmem_from_supabase.py
    # Dry run (no writes):
    DRY_RUN=1 python scripts/seed_stockmem_from_supabase.py
    # LLM enhancement (requires AIHUB_GEMINI_API_KEY):
    USE_LLM=1 python scripts/seed_stockmem_from_supabase.py

Environment:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  — Supabase project creds
    AIHUB_GEMINI_API_KEY                     — optional, for LLM enhancement
    STOCKMEM_URL                             — default http://localhost:8003
    SYMBOL                                   — default BTC
    DRY_RUN                                  — 1 = skip actual POSTs
    USE_LLM                                  — 1 = enable Gemini flash-lite LLM
    BATCH_CONCURRENCY                        — parallel POST workers (default 4)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Config ──────────────────────────────────────────────────────────────────────

def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for path in [ROOT / ".env", ROOT / "aihub" / ".env"]:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env.setdefault(k.strip(), v.strip())
    env.update(os.environ)
    return env


_ENV = _load_env()
SUPABASE_URL  = _ENV.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY  = _ENV.get("SUPABASE_SERVICE_ROLE_KEY", "")
STOCKMEM_URL  = _ENV.get("STOCKMEM_URL", "http://localhost:8003").rstrip("/")
SYMBOL        = _ENV.get("SYMBOL", "BTC").upper()
DRY_RUN       = _ENV.get("DRY_RUN", "0") == "1"
USE_LLM       = _ENV.get("USE_LLM", "0") == "1"
CONCURRENCY   = int(_ENV.get("BATCH_CONCURRENCY", "4"))
BINANCE_URL   = "https://api.binance.com/api/v3/klines"
HORIZONS      = [(1, "future_return_1d"), (3, "future_return_3d"),
                 (7, "future_return_7d"), (15, "future_return_15d"),
                 (30, "future_return_30d")]
TODAY         = date.today()


def _sb_headers() -> dict[str, str]:
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}


# ── Binance close price ──────────────────────────────────────────────────────────

_price_cache: dict[date, float] = {}


async def _binance_close(client: httpx.AsyncClient, d: date, sym: str) -> float | None:
    if d in _price_cache:
        return _price_cache[d]
    try:
        start_ms = int(d.strftime("%s")) * 1000
        end_ms   = start_ms + 86_400_000
        r = await client.get(
            BINANCE_URL,
            params={"symbol": sym, "interval": "1d", "limit": "1",
                    "startTime": str(start_ms), "endTime": str(end_ms)},
            timeout=15,
        )
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            price = float(data[0][4])  # close
            _price_cache[d] = price
            return price
    except Exception as exc:
        log.debug("Binance %s %s: %s", d, sym, exc)
    return None


# ── Event extraction ─────────────────────────────────────────────────────────────

def _build_extractor():
    from aihub.src.events.extractor import EventExtractor
    llm = None
    if USE_LLM:
        api_key = _ENV.get("AIHUB_GEMINI_API_KEY", "")
        if api_key:
            try:
                from aihub.src.llm.gemini import GeminiClient
                llm = GeminiClient(api_key=api_key, model="gemini-2.0-flash-lite")
                log.info("LLM enhancement: gemini-2.0-flash-lite")
            except Exception as exc:
                log.warning("Could not init Gemini client: %s — using rulebase only", exc)
        else:
            log.warning("USE_LLM=1 but AIHUB_GEMINI_API_KEY not set — using rulebase only")
    return EventExtractor(llm=llm)


async def _enrich_event_state(extractor, rec, history_recs: list):
    """Add/replace event_state on rec if currently missing or has < 2 events."""
    from aihub.src.events.schema import EventExtractionRequest
    from shared.models.event import DailyEventState

    has_events = rec.event_state and len(rec.event_state.events or []) >= 1
    if has_events:
        return  # already populated

    factors = list(rec.factors or [])
    events = []
    for factor_str in factors[:20]:   # cap to avoid runaway
        try:
            req = EventExtractionRequest(
                symbol=SYMBOL,
                title=factor_str,
                factors=[factor_str],
            )
            resp = await extractor.extract(req)
            events.extend(resp.events)
        except Exception as exc:
            log.debug("extract error %s: %s", factor_str, exc)

    # Deduplicate
    seen: set[tuple[str, str]] = set()
    deduped = []
    for ev in events:
        key = (ev.event_group, ev.event_type)
        if key not in seen:
            seen.add(key)
            deduped.append(ev)

    if deduped:
        rec.event_state = DailyEventState(
            date=rec.date,
            symbol=SYMBOL,
            events=deduped,
            article_count=len(rec.article_ids or []),
            source_count=len(set(rec.article_sources or [])),
        )


# ── Already-seeded check ─────────────────────────────────────────────────────────

async def _fetch_existing_dates(client: httpx.AsyncClient) -> set[str]:
    """Return set of record_dates already in local stockmem (to skip)."""
    try:
        r = await client.get(f"{STOCKMEM_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception:
        log.warning("Stockmem service not reachable at %s — proceeding anyway", STOCKMEM_URL)
        return set()
    # No bulk list endpoint — we'll rely on 409-style duplicate handling per record
    return set()


# ── Main ─────────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in .env")

    from stockmem.src.models import StockMemRecord

    extractor = _build_extractor()
    binance_sym = SYMBOL + "USDT"

    async with httpx.AsyncClient() as client:
        # ── 1. Fetch all records from Supabase ──
        log.info("Fetching %s records from Supabase...", SYMBOL)
        all_rows: list[tuple[str, date, dict, StockMemRecord | None]] = []
        limit, offset = 1000, 0
        while True:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/stockmem_records",
                headers=_sb_headers(),
                params={
                    "select": "id,record_date,symbol,payload",
                    "symbol": f"eq.{SYMBOL}",
                    "order":  "record_date.asc",
                    "limit":  str(limit),
                    "offset": str(offset),
                },
                timeout=60,
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            for row in batch:
                raw = row["payload"]
                payload = raw if isinstance(raw, dict) else json.loads(raw)
                rec_date = date.fromisoformat(row["record_date"])
                try:
                    rec = StockMemRecord.model_validate(payload)
                    all_rows.append((row["id"], rec_date, payload, rec))
                except Exception as exc:
                    log.warning("Skip %s parse error: %s", rec_date, exc)
            log.info("  fetched %d records so far", len(all_rows))
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.1)

        log.info("Total from Supabase: %d rows", len(all_rows))

        # ── 2. Process + POST ──
        sem = asyncio.Semaphore(CONCURRENCY)
        stats = {"posted": 0, "skipped": 0, "error": 0, "enriched": 0}

        async def _process_one(row_tuple: tuple) -> None:
            rid, rec_date, payload, rec = row_tuple
            if rec is None:
                stats["skipped"] += 1
                return

            async with sem:
                # Fill returns
                for days, field in HORIZONS:
                    if rec.__dict__.get(field) is None or getattr(rec, field) is None:
                        target_date = rec_date + timedelta(days=days)
                        if target_date <= TODAY:
                            base  = await _binance_close(client, rec_date, binance_sym)
                            close = await _binance_close(client, target_date, binance_sym)
                            if base and close:
                                val = round((close - base) / base * 100.0, 6)
                                setattr(rec, field, val)

                # Enrich event_state if needed
                try:
                    await _enrich_event_state(extractor, rec, [])
                    if rec.event_state and rec.event_state.events:
                        stats["enriched"] += 1
                except Exception as exc:
                    log.debug("Event enrichment failed %s: %s", rec_date, exc)

                if DRY_RUN:
                    stats["posted"] += 1
                    return

                # POST to stockmem service
                try:
                    resp = await client.post(
                        f"{STOCKMEM_URL}/record",
                        json={"record": rec.model_dump(mode="json")},
                        timeout=30,
                    )
                    if resp.status_code in (200, 201):
                        stats["posted"] += 1
                    elif resp.status_code == 409:
                        stats["skipped"] += 1   # already exists
                    else:
                        log.warning("POST %s → HTTP %d: %s", rec_date, resp.status_code, resp.text[:120])
                        stats["error"] += 1
                except Exception as exc:
                    log.warning("POST failed %s: %s", rec_date, exc)
                    stats["error"] += 1

        tasks = [_process_one(row) for row in all_rows]
        # Process in chunks to log progress
        chunk = 50
        for i in range(0, len(tasks), chunk):
            await asyncio.gather(*tasks[i:i + chunk])
            pct = min(i + chunk, len(tasks))
            log.info("  %d/%d done — posted=%d skipped=%d error=%d enriched=%d",
                     pct, len(tasks),
                     stats["posted"], stats["skipped"], stats["error"], stats["enriched"])

    mode = "DRY RUN" if DRY_RUN else "LIVE"
    log.info("[%s] Seed complete: posted=%d skipped=%d error=%d enriched=%d",
             mode, stats["posted"], stats["skipped"], stats["error"], stats["enriched"])


if __name__ == "__main__":
    asyncio.run(main())
