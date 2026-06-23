"""
Compute and store DailyEventState for all stockmem BTC records.
Processes records in date order — each record's event_state built from all preceding records.
Pure Python, no API calls — should finish in < 2 minutes for 2885 records.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date

import asyncpg

PG_DSN = "postgresql://postgres:pass@localhost:5432/postgres"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


async def main() -> None:
    import sys
    sys.path.insert(0, "/home/luong/marketlens")

    from stockmem.src.models import StockMemRecord
    from stockmem.src.search.event_memory import build_daily_event_state

    pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=5)

    # Load all BTC records sorted by date
    rows = await pool.fetch(
        "SELECT id, record_date, payload FROM stockmem_records WHERE symbol='BTC' ORDER BY record_date ASC"
    )
    log.info("Loaded %d records", len(rows))

    records: list[StockMemRecord] = []
    raw_payloads: list[tuple[str, dict]] = []  # (id, payload_dict)

    for row in rows:
        p = json.loads(row["payload"])
        try:
            rec = StockMemRecord.model_validate(p)
            records.append(rec)
            raw_payloads.append((row["id"], p))
        except Exception as exc:
            log.warning("  skip %s: %s", row["record_date"], exc)

    log.info("Parsed %d records OK", len(records))

    updated = skipped = 0
    batch: list[tuple[str, str]] = []  # (id, payload_json)

    for i, (rec, (rid, payload)) in enumerate(zip(records, raw_payloads)):
        history = records[:i]  # all records before this one
        try:
            event_state = build_daily_event_state(rec, history)
            payload["event_state"] = event_state.model_dump(mode="json")
            batch.append((rid, json.dumps(payload, ensure_ascii=True)))
            updated += 1
        except Exception as exc:
            log.warning("  %s event_state failed: %s", rec.date, exc)
            skipped += 1

        # Batch write every 200 records
        if len(batch) >= 200:
            async with pool.acquire() as conn:
                await conn.executemany(
                    "UPDATE stockmem_records SET payload=$2 WHERE id=$1",
                    batch,
                )
            log.info("  wrote batch up to %s (%d/%d)", rec.date, i + 1, len(records))
            batch.clear()

    # Write remaining
    if batch:
        async with pool.acquire() as conn:
            await conn.executemany(
                "UPDATE stockmem_records SET payload=$2 WHERE id=$1",
                batch,
            )

    log.info("Done — updated=%d skipped=%d", updated, skipped)

    # Verify
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM stockmem_records WHERE symbol='BTC' AND payload::json->>'event_state' IS NOT NULL AND payload::json->>'event_state' != 'null'"
    )
    log.info("Records with event_state stored: %s / %d", count, len(records))
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
