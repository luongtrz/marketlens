"""Fill missing future_return_{1d,3d,7d,15d,30d} for all BTC stockmem records.

Uses Supabase REST API (no direct PG connection needed) + Binance HTTPS.
Reads SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY from .env or environment.

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/fill_returns_rest.py
    PYTHONPATH=/home/luong/marketlens python scripts/fill_returns_rest.py --dry-run
    PYTHONPATH=/home/luong/marketlens python scripts/fill_returns_rest.py --symbol ETH
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# --- config ---------------------------------------------------------------

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

SUPABASE_URL   = _ENV.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY   = _ENV.get("SUPABASE_SERVICE_ROLE_KEY", "")
BINANCE_URL    = "https://api.binance.com/api/v3/klines"
TODAY          = date.today()

HORIZONS: list[tuple[int, str]] = [
    (1,  "future_return_1d"),
    (3,  "future_return_3d"),
    (7,  "future_return_7d"),
    (15, "future_return_15d"),
    (30, "future_return_30d"),
]

# --------------------------------------------------------------------------

def _sb_headers() -> dict[str, str]:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }


def _fetch_all_records(client: httpx.Client, symbol: str) -> list[dict]:
    """Paginated fetch of all records for symbol."""
    records: list[dict] = []
    limit = 1000
    offset = 0
    while True:
        r = client.get(
            f"{SUPABASE_URL}/rest/v1/stockmem_records",
            headers={**_sb_headers(), "Prefer": "count=exact"},
            params={
                "select":      "id,record_date,payload",
                "symbol":      f"eq.{symbol.upper()}",
                "order":       "record_date.asc",
                "limit":       str(limit),
                "offset":      str(offset),
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        records.extend(batch)
        log.info("  fetched %d records (offset=%d)", len(records), offset)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return records


def _binance_close(client: httpx.Client, target: date, binance_symbol: str) -> float | None:
    """Return daily close price on `target` from Binance klines."""
    end_ts = int(
        (datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
         + timedelta(days=1)).timestamp() * 1000
    )
    try:
        r = client.get(
            BINANCE_URL,
            params={
                "symbol":   binance_symbol,
                "interval": "1d",
                "limit":    "3",
                "endTime":  str(end_ts),
            },
            timeout=15,
        )
        r.raise_for_status()
        klines: list[list[Any]] = r.json()
        for kline in reversed(klines):
            open_ts = int(kline[0])
            candle_date = datetime.fromtimestamp(open_ts / 1000, tz=timezone.utc).date()
            if candle_date <= target:
                return float(kline[4])
    except Exception as exc:
        log.warning("Binance error for %s: %s", target, exc)
    return None


def _patch_record(client: httpx.Client, record_id: str, payload: dict) -> bool:
    r = client.patch(
        f"{SUPABASE_URL}/rest/v1/stockmem_records",
        headers=_sb_headers(),
        params={"id": f"eq.{record_id}"},
        content=json.dumps({"payload": json.dumps(payload, ensure_ascii=False)}),
        timeout=15,
    )
    return r.status_code in (200, 204)


def main(symbol: str = "BTC", dry_run: bool = False) -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    binance_symbol = symbol.upper() + "USDT"

    with httpx.Client() as client:
        log.info("Fetching all %s records from Supabase...", symbol)
        rows = _fetch_all_records(client, symbol)
        log.info("Total records: %d", len(rows))

        # Build a price cache so we only hit Binance once per date
        price_cache: dict[date, float] = {}

        updated = skipped = errors = 0

        for row in rows:
            record_id   = row["id"]
            record_date = date.fromisoformat(row["record_date"])
            payload     = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])

            # Find which horizons are missing AND matured
            to_fill: list[tuple[int, str]] = []
            for days, field in HORIZONS:
                target_date = record_date + timedelta(days=days)
                if payload.get(field) is None and target_date <= TODAY:
                    to_fill.append((days, field))

            if not to_fill:
                skipped += 1
                continue

            # Fetch close price on record_date if not cached
            if record_date not in price_cache:
                close = _binance_close(client, record_date, binance_symbol)
                if close is None:
                    log.warning("  skip %s: no close price", record_date)
                    errors += 1
                    continue
                price_cache[record_date] = close

            base_price = price_cache[record_date]

            changed = False
            for days, field in to_fill:
                target_date = record_date + timedelta(days=days)
                if target_date not in price_cache:
                    close = _binance_close(client, target_date, binance_symbol)
                    if close is not None:
                        price_cache[target_date] = close
                    else:
                        log.warning("  skip %s horizon=%dd: no Binance close", record_date, days)
                        continue

                ret = (price_cache[target_date] - base_price) / base_price * 100.0
                payload[field] = round(ret, 6)
                changed = True
                log.info("  %s %s = %.3f%%", record_date, field, ret)

            if changed:
                if dry_run:
                    log.info("  [dry-run] would PATCH %s", record_id)
                    updated += 1
                else:
                    ok = _patch_record(client, record_id, payload)
                    if ok:
                        updated += 1
                    else:
                        errors += 1
                        log.warning("  PATCH failed for %s", record_id)

                time.sleep(0.05)  # gentle rate limit

        log.info("Done. updated=%d skipped=%d errors=%d", updated, skipped, errors)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Fill missing future returns via REST API")
    p.add_argument("--symbol",  default="BTC")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(symbol=args.symbol, dry_run=args.dry_run)


if __name__ == "__main__":
    _cli()
