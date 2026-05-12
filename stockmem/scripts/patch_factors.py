"""Patch stockmem records that have empty factors by calling AIHub /factors.

Reads records with empty factors from PostgreSQL, fetches articles from Supabase
for the same date, calls AIHub for factors, and updates the records in-place.

Usage:
    PYTHONPATH=/home/luong/marketlens python stockmem/scripts/patch_factors.py
    PYTHONPATH=/home/luong/marketlens python stockmem/scripts/patch_factors.py --concurrency 3 --limit 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

# Silence asyncpg noise
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DB_URL = os.getenv(
    "STOCKMEM_DB_URL",
    "postgresql+asyncpg://postgres:pass@localhost:5432/postgres",
)
AIHUB_URL = os.getenv("AIHUB_URL", "http://localhost:8001")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")


async def call_aihub_factors(client: httpx.AsyncClient, text: str, ticker: str) -> list[dict]:
    """Call AIHub /factors and return list of factor dicts."""
    try:
        resp = await client.post(
            f"{AIHUB_URL}/factors",
            json={"text": text, "ticker": ticker},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json().get("factors", [])
    except Exception as exc:
        logger.warning("AIHub /factors failed: %s", exc)
        return []


async def fetch_articles_for_date(target_date: date, symbol: str) -> list:
    """Fetch articles from Supabase for the given date using the shared module."""
    from shared.supabase_news import fetch_news_articles_from_supabase

    day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    day_end = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, tzinfo=timezone.utc)

    try:
        return await fetch_news_articles_from_supabase(
            limit=15,
            symbol=symbol,
            publish_gte=day_start,
            publish_lte=day_end,
        )
    except Exception as exc:
        logger.warning("Supabase fetch failed for %s: %s", target_date, exc)
        return []


async def _load_env_from_files() -> None:
    """Load .env files so SUPABASE credentials are available."""
    for env_file in [ROOT / ".env", ROOT / "main_controller" / ".env", ROOT / "crawler" / ".env"]:
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k not in os.environ:
                os.environ[k] = v
    if os.environ.get("SUPABASE_URL"):
        logger.info("Loaded Supabase URL: %s", os.environ["SUPABASE_URL"][:40])


async def main(concurrency: int, limit: int, dry_run: bool, symbol: str) -> None:
    import sys
    sys.path.insert(0, str(ROOT))

    import asyncpg

    await _load_env_from_files()

    dsn = DB_URL.replace("postgresql+asyncpg://", "postgresql://")
    logger.info("Connecting to %s ...", dsn.split("@")[-1])
    pool = await asyncpg.create_pool(dsn)

    # Fetch records with empty factors
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, record_date, payload FROM stockmem_records"
            " WHERE symbol = $1"
            "   AND (payload::json->>'factors') = '[]'"
            "   AND (payload::json->>'future_return_7d') IS NOT NULL"
            " ORDER BY record_date"
            " LIMIT $2",
            symbol.upper(),
            limit,
        )

    logger.info("Found %d records with empty factors (limit=%d)", len(rows), limit)
    if not rows:
        logger.info("Nothing to patch.")
        await pool.close()
        return

    sem = asyncio.Semaphore(concurrency)
    patched = 0
    failed = 0

    async def patch_record(row: asyncio.Record) -> None:
        nonlocal patched, failed
        record_id = row["id"]
        record_date = date.fromisoformat(str(row["record_date"]))
        payload = json.loads(row["payload"])

        async with sem:
            articles = await fetch_articles_for_date(record_date, symbol)

            if articles:
                texts = [(getattr(a, "summary", None) or getattr(a, "article_name", None) or "")[:300]
                         for a in articles[:8]]
                combined = " ".join(t for t in texts if t.strip())[:2000]
            else:
                combined = ""

            if not combined.strip():
                summary = payload.get("summary") or ""
                combined = summary[:500] if summary else ""

            if not combined.strip():
                logger.debug("Skip %s — no text available", record_date)
                return

            async with httpx.AsyncClient() as client:
                factors = await call_aihub_factors(client, combined, symbol)

        if not factors:
            logger.warning("No factors returned for %s", record_date)
            failed += 1
            return

        day_start = datetime(record_date.year, record_date.month, record_date.day, tzinfo=timezone.utc)
        first_article_id = str(getattr(articles[0], "id", "patch")) if articles else "patch"

        payload["factors"] = [f["name"] for f in factors]
        payload["normalized_factors"] = [
            {
                "name": f["name"],
                "type": f["type"],
                "weight": f.get("confidence", 0.8),
                "polarity": f.get("polarity", 0.0),
                "sector": None,
                "related_symbols": [],
                "source_article_id": first_article_id,
                "observed_at": day_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            for f in factors
        ]

        if not dry_run:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE stockmem_records SET payload = $1 WHERE id = $2",
                    json.dumps(payload, ensure_ascii=True),
                    record_id,
                )
        else:
            logger.info("[DRY RUN] Would patch %s with %d factors", record_date, len(factors))

        patched += 1
        if patched % 50 == 0:
            logger.info("Progress: patched=%d failed=%d / %d", patched, failed, len(rows))

    tasks = [asyncio.create_task(patch_record(row)) for row in rows]
    await asyncio.gather(*tasks)

    await pool.close()
    logger.info("Done: patched=%d failed=%d total=%d", patched, failed, len(rows))


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=962)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.limit, args.dry_run, args.symbol))


if __name__ == "__main__":
    cli()
