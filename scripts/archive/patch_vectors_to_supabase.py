"""Patch price_vec, factor_vec, event_vec, indicator_vec from real_optimizer_v3.json
into Supabase stockmem_records payloads.

For each date in the local JSON:
  1. Fetch the current payload from Supabase
  2. Merge vector fields into it
  3. PATCH back via REST API

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/patch_vectors_to_supabase.py
    DRY_RUN=1 python scripts/patch_vectors_to_supabase.py   # preview only
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = ROOT / "stockmem/data/real_optimizer_v3.json"
SYMBOL    = os.getenv("SYMBOL", "BTC")
DRY_RUN   = os.getenv("DRY_RUN", "0") == "1"
CONCURRENCY = int(os.getenv("CONCURRENCY", "8"))

VECTOR_FIELDS = ["event_vec", "factor_vec", "indicator_vec", "price_vec",
                 "future_return_1d", "future_return_3d", "future_return_7d",
                 "future_return_15d", "future_return_30d"]


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for path in [ROOT / ".env", ROOT / "aihub/.env"]:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env.setdefault(k.strip(), v.strip())
    env.update(os.environ)
    return env


async def main() -> None:
    cfg = _load_env()
    sb_url = cfg.get("SUPABASE_URL", "").rstrip("/")
    sb_key = cfg.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not sb_url or not sb_key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")

    headers = {
        "apikey": sb_key,
        "Authorization": f"Bearer {sb_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    log.info("Loading %s ...", DATA_PATH)
    local_rows: list[dict] = json.loads(DATA_PATH.read_text())
    log.info("Loaded %d local rows", len(local_rows))

    # Build date → vectors map
    vec_by_date: dict[str, dict] = {}
    for row in local_rows:
        d = str(row["date"])
        vec_by_date[d] = {f: row.get(f) for f in VECTOR_FIELDS}

    # Fetch ALL Supabase payloads in pages
    log.info("Fetching Supabase records...")
    sb_rows: list[dict] = []
    limit, offset = 500, 0
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            r = await client.get(
                f"{sb_url}/rest/v1/stockmem_records",
                headers=headers,
                params={"select": "id,record_date,payload",
                        "symbol": f"eq.{SYMBOL}",
                        "order": "record_date.asc",
                        "limit": str(limit), "offset": str(offset)},
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            sb_rows.extend(batch)
            log.info("  fetched %d/%s", len(sb_rows), "?")
            if len(batch) < limit:
                break
            offset += limit

    log.info("Supabase records: %d", len(sb_rows))

    # Match and patch
    sem = asyncio.Semaphore(CONCURRENCY)
    stats = {"patched": 0, "skipped": 0, "no_local": 0, "error": 0}

    async def patch_one(sb_row: dict, client: httpx.AsyncClient) -> None:
        rec_date = sb_row["record_date"]
        vecs = vec_by_date.get(rec_date)
        if vecs is None:
            stats["no_local"] += 1
            return

        # Check if already has vectors
        raw = sb_row["payload"]
        payload = raw if isinstance(raw, dict) else json.loads(raw)

        already_ok = all(
            payload.get(f) is not None and len(payload.get(f) or []) > 0
            for f in ["price_vec", "factor_vec", "indicator_vec"]
        )
        if already_ok:
            stats["skipped"] += 1
            return

        # Merge vectors into payload
        merged = {**payload, **{k: v for k, v in vecs.items() if v is not None}}

        if DRY_RUN:
            log.info("[DRY] would patch %s", rec_date)
            stats["patched"] += 1
            return

        async with sem:
            try:
                resp = await client.patch(
                    f"{sb_url}/rest/v1/stockmem_records",
                    headers=headers,
                    params={"record_date": f"eq.{rec_date}", "symbol": f"eq.{SYMBOL}"},
                    content=json.dumps({"payload": merged}, ensure_ascii=True),
                )
                if resp.status_code in (200, 204):
                    stats["patched"] += 1
                else:
                    log.warning("PATCH %s → %d: %s", rec_date, resp.status_code, resp.text[:120])
                    stats["error"] += 1
            except Exception as exc:
                log.warning("PATCH %s error: %s", rec_date, exc)
                stats["error"] += 1

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [patch_one(row, client) for row in sb_rows]
        chunk = 50
        for i in range(0, len(tasks), chunk):
            await asyncio.gather(*tasks[i:i + chunk])
            done = min(i + chunk, len(tasks))
            log.info("  %d/%d — patched=%d skipped=%d no_local=%d error=%d",
                     done, len(tasks),
                     stats["patched"], stats["skipped"], stats["no_local"], stats["error"])

    mode = "DRY RUN" if DRY_RUN else "LIVE"
    log.info("[%s] Done: patched=%d skipped=%d no_local=%d error=%d",
             mode, stats["patched"], stats["skipped"], stats["no_local"], stats["error"])


if __name__ == "__main__":
    asyncio.run(main())
