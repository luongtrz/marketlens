"""
Fix records where sentiment_score = 0.0 by re-fetching article sentiment from Supabase.

Strategy:
1. Fetch all BTC records with sentiment_score = 0.0
2. For each record: fetch its article_ids from Supabase to get finbert_sentiment_score
3. If no sentiment from article_ids, fetch articles for that date by published_at
4. Compute average sentiment and update the payload

Supabase has 'finbert_sentiment_score' (0-1 range) and possibly 'sentiment_score' (-1 to 1).
We map finbert_sentiment_score → sentiment_score: (score - 0.5) * 2 to get [-1, 1] range.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, timedelta

import asyncpg
import httpx

PG_DSN = "postgresql://postgres:pass@localhost:5432/postgres"
SUPABASE_URL = "https://esctepjpgpjgrcymnabx.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVzY3RlcGpwZ3BqZ3JjeW1uYWJ4Iiwicm9sZSI6"
    "InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NTQ3MzQ3NCwiZXhwIjoyMDkxMDQ5NDc0fQ"
    ".H2iTfOBVs3LclKLDFJ4mL_jOImv_VFMhJd72OEG7ry4"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def _to_sentiment_score(finbert: float | None, raw: float | None) -> float | None:
    """Pick the best sentiment signal: prefer non-zero raw, fallback to finbert (both in [-1,1])."""
    if raw is not None and raw != 0.0:
        return float(raw)
    if finbert is not None and finbert != 0.0:
        return float(finbert)  # finbert_sentiment_score is already in [-1, 1]
    return None


def _sentiment_label(score: float) -> str:
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


async def fetch_sentiment_by_ids(
    client: httpx.AsyncClient,
    article_ids: list[str],
) -> float | None:
    """Fetch articles by IDs and compute average sentiment."""
    if not article_ids:
        return None
    # Batch up to 20 ids
    ids_str = ",".join(str(aid) for aid in article_ids[:20])
    try:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/news_articles",
            params={
                "select": "id,sentiment_score,finbert_sentiment_score",
                "id": f"in.({ids_str})",
                "limit": "20",
            },
            headers=SUPABASE_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        articles = resp.json()
    except Exception as exc:
        log.warning("  Supabase ID fetch error: %s", exc)
        return None

    scores = []
    for a in articles:
        s = _to_sentiment_score(a.get("finbert_sentiment_score"), a.get("sentiment_score"))
        if s is not None:
            scores.append(s)

    return float(sum(scores) / len(scores)) if scores else None


async def fetch_sentiment_by_date(
    client: httpx.AsyncClient,
    record_date: date,
) -> float | None:
    """Fallback: fetch articles published on this date and compute sentiment.
    Uses list-of-tuples params so httpx sends duplicate published_at filters correctly.
    """
    rd = date.fromisoformat(str(record_date))
    start = f"{rd}T00:00:00"
    end = f"{rd + timedelta(days=1)}T00:00:00"
    try:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/news_articles",
            params=[
                ("select", "id,sentiment_score,finbert_sentiment_score"),
                ("publish_at", f"gte.{start}"),
                ("publish_at", f"lt.{end}"),
                ("limit", "50"),
            ],
            headers=SUPABASE_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        articles = resp.json()
    except Exception as exc:
        log.warning("  Supabase date fetch error for %s: %s", record_date, exc)
        return None

    scores = []
    for a in articles:
        s = _to_sentiment_score(a.get("finbert_sentiment_score"), a.get("sentiment_score"))
        if s is not None:
            scores.append(s)

    return float(sum(scores) / len(scores)) if scores else None


async def process_record(
    row: asyncpg.Record,
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    counters: dict,
) -> None:
    record_id: str = row["id"]
    record_date = row["record_date"]
    payload: dict = json.loads(row["payload"])

    article_ids = payload.get("article_ids") or []

    async with sem:
        # Try 1: fetch by stored article_ids
        score = await fetch_sentiment_by_ids(client, article_ids)

        # Try 2: fetch by date if no score from ids
        if score is None:
            score = await fetch_sentiment_by_date(client, record_date)

    if score is None or score == 0.0:
        log.debug("  %s: no non-zero sentiment found", record_date)
        counters["no_data"] += 1
        return

    # Update payload
    payload["sentiment_score"] = round(score, 6)
    payload["sentiment_label"] = _sentiment_label(score)

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE stockmem_records SET payload=$1 WHERE id=$2",
            json.dumps(payload, ensure_ascii=True),
            record_id,
        )

    counters["updated"] += 1
    log.info("  %s: sentiment %.4f (%s) from %d articles",
             record_date, score, payload["sentiment_label"], len(article_ids))


async def main() -> None:
    pool = await asyncpg.create_pool(PG_DSN, min_size=3, max_size=8)

    # Find all BTC records with zero sentiment
    rows = await pool.fetch(
        """
        SELECT id, record_date, payload
        FROM stockmem_records
        WHERE symbol = 'BTC'
          AND (payload::json->>'sentiment_score')::float = 0.0
        ORDER BY record_date
        """
    )
    log.info("Records with sentiment_score = 0.0: %d", len(rows))

    # Show breakdown by year before fix
    year_counts: dict[int, int] = {}
    for row in rows:
        rd = row["record_date"]
        y = rd.year if hasattr(rd, "year") else int(str(rd)[:4])
        year_counts[y] = year_counts.get(y, 0) + 1
    for y, cnt in sorted(year_counts.items()):
        log.info("  %d: %d zero-sentiment records", y, cnt)

    sem = asyncio.Semaphore(8)  # 8 concurrent Supabase calls — no rate limit concern
    counters = {"updated": 0, "no_data": 0}

    async with httpx.AsyncClient(timeout=20) as client:
        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            tasks = [process_record(row, pool, client, sem, counters) for row in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            done = counters["updated"] + counters["no_data"]
            log.info("Progress: %d/%d (updated=%d no_data=%d)",
                     done, len(rows), counters["updated"], counters["no_data"])

    log.info("Done — updated=%d no_data=%d", counters["updated"], counters["no_data"])

    # Final audit
    stats = await pool.fetchrow(
        """
        SELECT
          COUNT(*) as total,
          COUNT(*) FILTER (WHERE (payload::json->>'sentiment_score')::float = 0.0) as zero,
          COUNT(*) FILTER (WHERE ABS((payload::json->>'sentiment_score')::float) > 0.001) as nonzero
        FROM stockmem_records WHERE symbol='BTC'
        """
    )
    log.info("Final: total=%s zero_sentiment=%s nonzero=%s",
             stats["total"], stats["zero"], stats["nonzero"])

    # Year breakdown
    year_rows = await pool.fetch(
        """
        SELECT
          EXTRACT(year FROM record_date)::int AS yr,
          COUNT(*) as total,
          COUNT(*) FILTER (WHERE (payload::json->>'sentiment_score')::float = 0.0) as zero
        FROM stockmem_records WHERE symbol='BTC'
        GROUP BY yr ORDER BY yr
        """
    )
    for r in year_rows:
        log.info("  %d: %d total, %d zero sentiment", r["yr"], r["total"], r["zero"])

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
