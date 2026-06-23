"""
Fix records with factors=[] by re-extracting via Groq.

Strategy per record:
- If summary exists in payload: use it directly
- Else: re-fetch article headers from Supabase using stored article_ids

Concurrent: Semaphore(5), retry on 429 with short backoff.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import date
from typing import Any

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
GROQ_KEY = "[REDACTED_GROQ_TOKEN]"
GROQ_MODEL = "llama-3.1-8b-instant"

CONCURRENCY = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_SYSTEM = "You are a crypto analyst. Respond only with valid JSON, no markdown."
_VALID_TYPES = {"macro", "regulatory", "technical", "sentiment", "on_chain", "exchange"}


async def fetch_headers_from_supabase(
    client: httpx.AsyncClient,
    article_ids: list[str],
) -> list[str]:
    """Fetch article headers from Supabase by IDs."""
    if not article_ids:
        return []
    ids_str = ",".join(article_ids[:10])
    try:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/news_articles",
            params={"select": "header", "id": f"in.({ids_str})", "limit": 10},
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=15,
        )
        resp.raise_for_status()
        return [r["header"] for r in resp.json() if r.get("header")]
    except Exception as exc:
        log.warning("Supabase fetch failed: %s", exc)
        return []


async def extract_factors(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    record_date: str,
    text: str,
) -> list[dict]:
    prompt = (
        f'BTC news {record_date}: {text[:600]}\n\n'
        f'List top 5 market factors as JSON: {{"factors":[{{"name":"...","type":"macro|regulatory|technical|sentiment|on_chain|exchange","polarity":-1.0}}]}}'
    )

    for attempt in range(6):
        async with sem:
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 300,
                        "temperature": 0.1,
                    },
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = min(1.0 + attempt * 0.5, 3.0) + random.uniform(0, 0.5)
                    log.warning("  %s: 429 retry %.1fs (attempt %d)", record_date, wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                start = content.find("{")
                end = content.rfind("}") + 1
                if start == -1:
                    return []
                data = json.loads(content[start:end])
                factors = []
                for f in data.get("factors", []):
                    raw_type = str(f.get("type", "macro")).lower().replace("-", "_").replace(" ", "_")
                    if raw_type not in _VALID_TYPES:
                        raw_type = "macro"
                    name = str(f.get("name", "")).strip()
                    if name:
                        factors.append({
                            "name": name,
                            "type": raw_type,
                            "polarity": float(f.get("polarity", 0.0)),
                            "confidence": 0.8,
                        })
                return factors
            except asyncio.TimeoutError:
                await asyncio.sleep(1 + attempt * 0.5)
            except Exception as exc:
                log.warning("  %s: error attempt %d: %s", record_date, attempt + 1, str(exc)[:80])
                await asyncio.sleep(0.5)
    return []


async def process_record(
    row: asyncpg.Record,
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    counters: dict,
) -> None:
    record_id: str = row["id"]
    record_date: str = row["record_date"]
    payload: dict = json.loads(row["payload"])

    # Build text to extract from: prefer summary, fallback to re-fetch from Supabase
    text = (payload.get("summary") or "").strip()
    if not text:
        article_ids = payload.get("article_ids") or []
        headers = await fetch_headers_from_supabase(client, article_ids)
        text = " | ".join(headers)

    if not text:
        log.warning("  %s: no text available, skipping", record_date)
        counters["skipped"] += 1
        return

    factors = await extract_factors(client, sem, record_date, text)

    if not factors:
        log.warning("  %s: still 0 factors after retries", record_date)
        counters["failed"] += 1
        return

    # Update payload
    payload["factors"] = [f["name"] for f in factors]
    payload["normalized_factors"] = [
        {
            "name": f["name"],
            "type": f["type"],
            "weight": f["confidence"],
            "polarity": f["polarity"],
            "source_article_id": (payload.get("article_ids") or ["backfill"])[0],
            "observed_at": f"{record_date}T00:00:00+00:00",
        }
        for f in factors
    ]

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE stockmem_records SET payload=$1 WHERE id=$2",
            json.dumps(payload, ensure_ascii=True),
            record_id,
        )

    counters["updated"] += 1
    log.info("  %s: → %d factors (%s)", record_date, len(factors), ", ".join(f["name"] for f in factors[:3]))


async def main() -> None:
    pool = await asyncpg.create_pool(PG_DSN, min_size=3, max_size=10)

    # Find all records with empty factors
    rows = await pool.fetch(
        """
        SELECT id, record_date, payload FROM stockmem_records
        WHERE symbol = 'BTC'
          AND (
            (payload::json->>'factors') IS NULL
            OR (payload::json->>'factors')::jsonb = '[]'::jsonb
          )
        ORDER BY record_date
        """
    )
    log.info("Records with empty factors: %d", len(rows))

    sem = asyncio.Semaphore(CONCURRENCY)
    counters = {"updated": 0, "skipped": 0, "failed": 0}

    async with httpx.AsyncClient(timeout=30) as client:
        # Process in monthly batches to keep logs readable
        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            tasks = [
                process_record(row, pool, client, sem, counters)
                for row in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.error("Task error: %s", r)
                    counters["failed"] += 1

            done = counters["updated"] + counters["skipped"] + counters["failed"]
            log.info("Progress: %d/%d done (updated=%d failed=%d skipped=%d)",
                     done, len(rows), counters["updated"], counters["failed"], counters["skipped"])

    log.info("Done — updated=%d failed=%d skipped=%d", counters["updated"], counters["failed"], counters["skipped"])

    # Final stats
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE (payload::json->>'factors')::jsonb != '[]'::jsonb) as has_factors,
            COUNT(*) FILTER (WHERE (payload::json->>'factors')::jsonb = '[]'::jsonb
                              OR payload::json->>'factors' IS NULL) as empty
        FROM stockmem_records WHERE symbol='BTC'
        """
    )
    log.info("Final: total=%s has_factors=%s empty=%s", row["total"], row["has_factors"], row["empty"])
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
