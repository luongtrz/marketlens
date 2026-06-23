"""Regenerate real_optimizer_v2.json from Supabase via REST API.

Does NOT require a direct PostgreSQL connection — uses Supabase HTTPS REST API.
For each record:
  1. Parses payload as StockMemRecord.
  2. Extracts event_state from raw factor strings via EventExtractor (rule-based).
  3. Embeds with RecordEmbedder → event_vec/factor_vec/indicator_vec/price_vec.
  4. Fills d3/d15 from Binance if missing.
  5. Writes updated real_optimizer_v2.json.

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/regen_optimizer_rest.py
    PYTHONPATH=/home/luong/marketlens python scripts/regen_optimizer_rest.py \\
        --output stockmem/data/real_optimizer_v2.json \\
        --symbol BTC
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUT = ROOT / "stockmem" / "data" / "real_optimizer_v2.json"
TODAY = date.today()
BINANCE_URL = "https://api.binance.com/api/v3/klines"

HORIZONS: list[tuple[int, str]] = [
    (1,  "future_return_1d"),
    (3,  "future_return_3d"),
    (7,  "future_return_7d"),
    (15, "future_return_15d"),
    (30, "future_return_30d"),
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    env.update(os.environ)
    return env


_ENV = _load_env()
SUPABASE_URL = _ENV.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = _ENV.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _sb_headers() -> dict[str, str]:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }


# ---------------------------------------------------------------------------
# Supabase fetch (paginated)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Event extraction from raw factors (rule-based, no LLM)
# ---------------------------------------------------------------------------

async def _extract_event_state(
    extractor: Any,
    rec_date: date,
    symbol: str,
    factors: list[str],
    history: list[Any],   # list[StockMemRecord] for novelty calc
) -> Any:
    from aihub.src.events.schema import EventExtractionRequest
    from stockmem.src.search.event_memory import build_daily_event_state
    from stockmem.src.models import StockMemRecord as LocalRec

    event_records = []
    for factor_str in factors:
        try:
            req = EventExtractionRequest(symbol=symbol, title=factor_str, factors=[factor_str])
            resp = await extractor.extract(req)
            event_records.extend(resp.events)
        except Exception as exc:
            log.debug("EventExtractor error for %r: %s", factor_str, exc)

    # Deduplicate by (event_group, event_type)
    seen: set[tuple[str, str]] = set()
    deduped = []
    for er in event_records:
        key = (er.event_group, er.event_type)
        if key not in seen:
            seen.add(key)
            deduped.append(er)

    # Build a minimal StockMemRecord shell for build_daily_event_state
    shell = LocalRec.model_construct(
        date=rec_date,
        symbol=symbol,
        factors=factors,
        article_ids=[],
        article_sources=[],
    )

    try:
        es = build_daily_event_state(shell, history)
        if deduped:
            es.events = deduped  # type: ignore[assignment]
    except Exception as exc:
        log.debug("build_daily_event_state failed: %s", exc)
        from shared.models.event import DailyEventState
        es = DailyEventState(
            date=rec_date,
            symbol=symbol,
            events=deduped,
            article_count=0,
            source_count=0,
        )
    return es


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(symbol: str, output: Path) -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")

    from aihub.src.events.extractor import EventExtractor
    from stockmem.src.models import StockMemRecord
    from stockmem.src.search.embedder import RecordEmbedder

    extractor = EventExtractor(llm=None)
    binance_sym = symbol.upper() + "USDT"
    embedder = RecordEmbedder()

    async with httpx.AsyncClient() as client:
        log.info("Fetching %s records from Supabase...", symbol)

        # Paginated fetch
        records_raw: list[tuple[str, date, dict, StockMemRecord | None]] = []
        limit, offset = 1000, 0
        while True:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/stockmem_records",
                headers=_sb_headers(),
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
                    records_raw.append((row["id"], rec_date, payload, rec))
                except Exception as exc:
                    log.warning("Skip %s: %s", rec_date, exc)
                    records_raw.append((row["id"], rec_date, payload, None))
            log.info("  fetched %d records", len(records_raw))
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.2)

        log.info("Total: %d rows loaded", len(records_raw))

        # Build embedder corpus on all valid records
        valid_recs = [r for _, _, _, r in records_raw if r is not None]
        log.info("Building embedder corpus on %d records...", len(valid_recs))
        embedder.rebuild_corpus(valid_recs)

        # Price cache for Binance
        price_cache: dict[date, float] = {}

        async def _get_price(d: date) -> float | None:
            if d in price_cache:
                return price_cache[d]
            p = _binance_close_sync(d, binance_sym)
            if p is not None:
                price_cache[d] = p
            return p

        out_rows: list[dict] = []
        skipped = 0

        for i, (rid, rec_date, payload, rec) in enumerate(records_raw):
            if rec is None:
                skipped += 1
                continue

            # --- Extract event_state from raw factor strings ---
            history_recs = [r for _, _, _, r in records_raw[:i] if r is not None]
            if not rec.event_state or not rec.event_state.events:
                try:
                    es = await _extract_event_state(
                        extractor, rec_date, symbol, rec.factors or [], history_recs
                    )
                    rec.event_state = es
                except Exception as exc:
                    log.debug("Event extraction failed %s: %s", rec_date, exc)

            # --- Embed ---
            try:
                split = embedder.embed_split(rec)
            except Exception as exc:
                log.warning("Embed failed %s: %s", rec_date, exc)
                skipped += 1
                continue

            # --- Returns: use payload values, fill missing from Binance ---
            returns: dict[str, float | None] = {
                "future_return_1d":  payload.get("future_return_1d"),
                "future_return_3d":  payload.get("future_return_3d"),
                "future_return_7d":  payload.get("future_return_7d"),
                "future_return_15d": payload.get("future_return_15d"),
                "future_return_30d": payload.get("future_return_30d"),
            }

            for days, field in HORIZONS:
                if returns[field] is None and (rec_date + timedelta(days=days)) <= TODAY:
                    base = await _get_price(rec_date)
                    tgt  = await _get_price(rec_date + timedelta(days=days))
                    if base and tgt:
                        returns[field] = round((tgt - base) / base * 100.0, 6)

            out_rows.append({
                "date":               str(rec_date),
                "event_vec":          split.event_vec.tolist(),
                "factor_vec":         split.factor_vec.tolist(),
                "indicator_vec":      split.indicator_vec.tolist(),
                "price_vec":          split.price_vec.tolist(),
                "future_return_1d":   returns["future_return_1d"],
                "future_return_3d":   returns["future_return_3d"],
                "future_return_7d":   returns["future_return_7d"],
                "future_return_15d":  returns["future_return_15d"],
                "future_return_30d":  returns["future_return_30d"],
            })

            if (i + 1) % 100 == 0:
                log.info("  processed %d / %d", i + 1, len(records_raw))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_rows, ensure_ascii=False), encoding="utf-8")

    def _nn(field: str) -> str:
        n = sum(1 for r in out_rows if r.get(field) is not None)
        return f"{n}/{len(out_rows)}"

    def _nz(field: str) -> str:
        n = sum(1 for r in out_rows if any(v != 0 for v in r.get(field, [])))
        return f"{n}/{len(out_rows)}"

    log.info("Wrote %d rows → %s  (skipped=%d)", len(out_rows), output, skipped)
    log.info("  future_return_1d  : %s", _nn("future_return_1d"))
    log.info("  future_return_3d  : %s", _nn("future_return_3d"))
    log.info("  future_return_7d  : %s", _nn("future_return_7d"))
    log.info("  future_return_15d : %s", _nn("future_return_15d"))
    log.info("  future_return_30d : %s", _nn("future_return_30d"))
    log.info("  event_vec nonzero : %s", _nz("event_vec"))
    log.info("  factor_vec nonzero: %s", _nz("factor_vec"))
    log.info("  price_vec nonzero : %s", _nz("price_vec"))


def _binance_close_sync(target: date, binance_sym: str) -> float | None:
    """Synchronous Binance price fetch (used inside async context via direct call)."""
    end_ts = int(
        (datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
         + timedelta(days=1)).timestamp() * 1000
    )
    try:
        r = httpx.get(
            BINANCE_URL,
            params={"symbol": binance_sym, "interval": "1d", "limit": "3", "endTime": str(end_ts)},
            timeout=15,
        )
        r.raise_for_status()
        klines: list[list[Any]] = r.json()
        for kline in reversed(klines):
            cd = datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc).date()
            if cd <= target:
                return float(kline[4])
    except Exception as exc:
        log.warning("Binance error %s: %s", target, exc)
    return None


def main(symbol: str = "BTC", output: Path = DEFAULT_OUT) -> None:
    asyncio.run(_main(symbol, output))


def _cli() -> None:
    p = argparse.ArgumentParser(description="Regen optimizer JSON via Supabase REST API")
    p.add_argument("--symbol",  default="BTC")
    p.add_argument("--output",  default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(symbol=args.symbol, output=Path(args.output))


if __name__ == "__main__":
    _cli()
