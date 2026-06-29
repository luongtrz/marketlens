"""Rebuild crawler seen_*.json from Supabase ``source_url`` values.

Why: the crawler marks *skipped* URLs (e.g. published before ``min_publish_year``)
as seen. After lowering ``min_publish_year`` (e.g. 2023 → 2018) those URLs would
never be re-attempted. This script resets each ``crawler/data/seen_*.json`` to
contain only URLs that actually exist in Supabase, so previously skipped old
articles get crawled while already-inserted ones are not re-enriched.

Usage:
    python3 scripts/rebuild_crawler_seen_from_supabase.py            # rewrite all seen_*.json
    python3 scripts/rebuild_crawler_seen_from_supabase.py --dry-run  # only report

Reads SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) from env / .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "crawler" / "data"
PAGE_SIZE = 1000


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / "crawler" / ".env", override=False)


def fetch_all_source_urls(base_url: str, api_key: str, table: str) -> set[str]:
    headers = {"apikey": api_key, "Authorization": f"Bearer {api_key}"}
    urls: set[str] = set()
    offset = 0
    with httpx.Client(timeout=30) as client:
        while True:
            resp = client.get(
                f"{base_url}/rest/v1/{table}",
                headers=headers,
                params={
                    "select": "source_url",
                    "order": "id.asc",
                    "limit": str(PAGE_SIZE),
                    "offset": str(offset),
                },
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            for row in rows:
                u = (row.get("source_url") or "").strip()
                if u:
                    urls.add(u)
            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            print(f"  fetched {offset} rows...", file=sys.stderr)
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, do not write")
    args = parser.parse_args()

    _load_env()
    base_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    api_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
    table = os.getenv("SUPABASE_TABLE", "news_articles")
    if not base_url or not api_key:
        print("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in env", file=sys.stderr)
        return 1

    print(f"Fetching source_url from {table}...")
    db_urls = fetch_all_source_urls(base_url, api_key, table)
    print(f"Supabase has {len(db_urls)} unique source_url values")

    seen_files = sorted(DATA_DIR.glob("seen_*.json")) or [DATA_DIR / "seen.json"]
    for path in seen_files:
        old_count = 0
        if path.exists():
            try:
                old_count = len(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        print(f"{path.name}: {old_count} -> {len(db_urls)} URLs")
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(sorted(db_urls), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    if args.dry_run:
        print("(dry-run: nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
