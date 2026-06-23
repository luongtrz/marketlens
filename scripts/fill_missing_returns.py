"""
Fill missing future_return_{1d,3d,7d,15d,30d} for BTC stockmem records.

Connects directly to postgres and Binance REST API — no service dependencies.
Only fills horizons where the target date has already passed (today).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import asyncpg
import httpx

PG_DSN = "postgresql://postgres:pass@localhost:5432/postgres"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTC"
BINANCE_SYMBOL = "BTCUSDT"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TODAY = date.today()
HORIZONS = [(1, "future_return_1d"), (3, "future_return_3d"), (7, "future_return_7d"), (15, "future_return_15d"), (30, "future_return_30d")]


async def fetch_binance_close(client: httpx.AsyncClient, target: date) -> float | None:
    """Return the daily close price on `target` from Binance klines."""
    end_ts = int(
        (datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
         + timedelta(days=1)).timestamp() * 1000
    )
    resp = await client.get(
        BINANCE_URL,
        params={"symbol": BINANCE_SYMBOL, "interval": "1d", "limit": 3, "endTime": str(end_ts)},
        timeout=15,
    )
    resp.raise_for_status()
    klines: list[list[Any]] = resp.json()
    # Each kline: [open_time_ms, open, high, low, close, volume, close_time_ms, ...]
    for kline in reversed(klines):
        open_time_ms = int(kline[0])
        candle_date = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).date()
        if candle_date <= target:
            return float(kline[4])
    return None


async def main() -> None:
    pool = await asyncpg.create_pool(PG_DSN)

    try:
        rows = await pool.fetch(
            "SELECT id, record_date, payload FROM stockmem_records"
            " WHERE symbol = $1"
            " AND ("
            "  (payload::json->>'future_return_1d') IS NULL OR"
            "  (payload::json->>'future_return_3d') IS NULL OR"
            "  (payload::json->>'future_return_7d') IS NULL OR"
            "  (payload::json->>'future_return_15d') IS NULL OR"
            "  (payload::json->>'future_return_30d') IS NULL"
            " )"
            " ORDER BY record_date",
            SYMBOL.upper(),
        )
        log.info("Records with missing returns: %d", len(rows))

        updated = 0
        skipped = 0
        errors: list[str] = []

        async with httpx.AsyncClient() as client:
            for row in rows:
                record_id: str = row["id"]
                record_date: date = date.fromisoformat(row["record_date"])
                payload: dict = json.loads(row["payload"])

                # Base close from the record's own ohlcv
                try:
                    base_close = float(payload["market_snapshot"]["ohlcv"]["close"])
                except (KeyError, TypeError, ValueError):
                    log.warning("%s: cannot read base_close, skipping", record_date)
                    skipped += 1
                    continue

                if base_close == 0:
                    skipped += 1
                    continue

                updates: dict[str, float] = {}

                for offset_days, field in HORIZONS:
                    # Skip if already filled
                    if payload.get(field) is not None:
                        continue
                    target = record_date + timedelta(days=offset_days)
                    if target > TODAY:
                        continue  # window not yet closed
                    try:
                        future_close = await fetch_binance_close(client, target)
                        if future_close is not None:
                            updates[field] = round((future_close - base_close) / base_close * 100.0, 4)
                    except Exception as exc:
                        errors.append(f"{record_date} +{offset_days}d: {exc}")

                if not updates:
                    skipped += 1
                    continue

                # Merge updates into payload and write back
                payload.update(updates)
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE stockmem_records SET payload = $1 WHERE id = $2",
                        json.dumps(payload, ensure_ascii=True),
                        record_id,
                    )
                log.info("%s: filled %s", record_date, list(updates.keys()))
                updated += 1

        log.info("Done — updated=%d skipped=%d errors=%d", updated, skipped, len(errors))
        for e in errors:
            log.warning("  ERR: %s", e)

        # Final summary query
        row = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE (payload::json->>'future_return_1d') IS NOT NULL) as has_1d,
                COUNT(*) FILTER (WHERE (payload::json->>'future_return_3d') IS NOT NULL) as has_3d,
                COUNT(*) FILTER (WHERE (payload::json->>'future_return_7d') IS NOT NULL) as has_7d,
                COUNT(*) FILTER (WHERE (payload::json->>'future_return_15d') IS NOT NULL) as has_15d,
                COUNT(*) FILTER (WHERE (payload::json->>'future_return_30d') IS NOT NULL) as has_30d
            FROM stockmem_records WHERE symbol = 'BTC'
            """
        )
        log.info(
            "Final BTC coverage: total=%s 1d=%s 3d=%s 7d=%s 15d=%s 30d=%s",
            row["total"], row["has_1d"], row["has_3d"], row["has_7d"], row["has_15d"], row["has_30d"],
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
