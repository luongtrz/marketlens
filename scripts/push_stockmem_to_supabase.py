"""
Push all stockmem BTC records from local PostgreSQL to Supabase.

Step 1: Creates the table if it doesn't exist (via Supabase SQL editor - see instructions).
Step 2: Batch-upserts all records using Supabase REST API.

Usage:
    python scripts/push_stockmem_to_supabase.py
    python scripts/push_stockmem_to_supabase.py --dry-run   # count only, no writes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import asyncpg
import httpx

LOCAL_DSN = "postgresql://postgres:pass@localhost:5432/postgres"
SUPABASE_URL = "https://esctepjpgpjgrcymnabx.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVzY3RlcGpwZ3BqZ3JjeW1uYWJ4Iiwicm9sZSI6"
    "InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NTQ3MzQ3NCwiZXhwIjoyMDkxMDQ5NDc0fQ"
    ".H2iTfOBVs3LclKLDFJ4mL_jOImv_VFMhJd72OEG7ry4"
)

BATCH_SIZE = 25  # ~25 records × ~15KB = ~375KB per request

CREATE_TABLE_SQL = """
-- Run this once in the Supabase SQL Editor:
-- https://supabase.com/dashboard/project/esctepjpgpjgrcymnabx/sql

CREATE TABLE IF NOT EXISTS stockmem_records (
    id          TEXT PRIMARY KEY,
    record_date TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    payload     JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stockmem_symbol_date ON stockmem_records (symbol, record_date);
CREATE INDEX IF NOT EXISTS idx_stockmem_record_date ON stockmem_records (record_date);
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


async def check_table_exists(client: httpx.AsyncClient) -> bool:
    resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/stockmem_records",
        params={"limit": "1"},
        headers=BASE_HEADERS,
        timeout=15,
    )
    if resp.status_code == 200:
        return True
    if resp.status_code == 404 or "does not exist" in resp.text:
        return False
    log.error("Unexpected response checking table: %d %s", resp.status_code, resp.text[:200])
    return False


async def push_batch(
    client: httpx.AsyncClient,
    records: list[dict],
    sem: asyncio.Semaphore,
) -> tuple[int, int]:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{SUPABASE_URL}/rest/v1/stockmem_records",
                    headers={
                        **BASE_HEADERS,
                        "Prefer": "resolution=merge-duplicates,return=minimal",
                    },
                    content=json.dumps(records, ensure_ascii=True),
                    timeout=60,
                )
                if resp.status_code in (200, 201):
                    return len(records), 0
                log.warning(
                    "Batch attempt %d: HTTP %d — %s",
                    attempt + 1,
                    resp.status_code,
                    resp.text[:150],
                )
                await asyncio.sleep(1 + attempt)
            except Exception as exc:
                log.warning("Batch attempt %d error: %s", attempt + 1, exc)
                await asyncio.sleep(1 + attempt)
    return 0, len(records)


async def main(dry_run: bool = False) -> None:
    # Load from local PG
    pool = await asyncpg.create_pool(LOCAL_DSN, min_size=2, max_size=4)
    rows = await pool.fetch(
        "SELECT id, record_date, symbol, payload "
        "FROM stockmem_records WHERE symbol='BTC' ORDER BY record_date"
    )
    await pool.close()
    log.info("Loaded %d records from local PG", len(rows))

    if dry_run:
        log.info("DRY RUN — no writes. Would push %d records in %d batches.",
                 len(rows), (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE)
        return

    async with httpx.AsyncClient(timeout=60) as client:
        # Check if table exists
        exists = await check_table_exists(client)
        if not exists:
            print("\n" + "=" * 60)
            print("TABLE stockmem_records does not exist on Supabase!")
            print("Run the following SQL in the Supabase SQL Editor:")
            print("https://supabase.com/dashboard/project/esctepjpgpjgrcymnabx/sql")
            print("=" * 60)
            print(CREATE_TABLE_SQL)
            print("=" * 60)
            print("Then re-run this script.")
            sys.exit(1)

        log.info("Table exists — starting upsert of %d records (batch=%d)", len(rows), BATCH_SIZE)

        # Build batches
        batches: list[list[dict]] = []
        for i in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[i : i + BATCH_SIZE]
            batches.append([
                {
                    "id": r["id"],
                    "record_date": r["record_date"],
                    "symbol": r["symbol"],
                    "payload": json.loads(r["payload"]),
                }
                for r in batch_rows
            ])

        # Concurrent upsert (5 parallel batches)
        sem = asyncio.Semaphore(5)
        total_ok = total_err = 0

        for batch_idx in range(0, len(batches), 5):
            chunk = batches[batch_idx : batch_idx + 5]
            results = await asyncio.gather(
                *[push_batch(client, b, sem) for b in chunk],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    total_err += BATCH_SIZE
                else:
                    ok, err = r
                    total_ok += ok
                    total_err += err

            log.info(
                "Progress: %d/%d pushed (errors=%d)",
                total_ok,
                len(rows),
                total_err,
            )

        log.info("Done — pushed=%d errors=%d", total_ok, total_err)

        # Verify
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/stockmem_records",
            params={"select": "count", "symbol": "eq.BTC"},
            headers={**BASE_HEADERS, "Prefer": "count=exact"},
            timeout=15,
        )
        count_header = resp.headers.get("content-range", "?/?")
        log.info("Supabase record count (Content-Range): %s", count_header)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
