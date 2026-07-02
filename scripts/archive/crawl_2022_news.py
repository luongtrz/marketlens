"""Crawl REAL 2022 BTC article content from Decrypt sitemaps → Supabase.

Decrypt has ~741 BTC articles in 2022, server-side rendered with full text.
Uses the same fetch_article_content() code from the live crawler (CSS selectors).
After this runs, call POST /backfill for 2022 offsets to regenerate StockMem
records with real CryptoBERT sentiment and LLM factors.

Decrypt 2022 sitemap pages: post-sitemap12 through post-sitemap16.

Usage:
    python scripts/crawl_2022_news.py [--dry-run] [--workers N] [--limit N]
    python scripts/crawl_2022_news.py --dry-run       # discover + print, no writes
    python scripts/crawl_2022_news.py --workers 4     # 4 concurrent fetches
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from crawler.src.rss.fetcher import RSSFetcher, FeedSource

SUPABASE_URL   = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY   = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "news_articles")

START_DATE = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
END_DATE   = datetime.datetime(2022, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)

BTC_RE = re.compile(r"\b(bitcoin|btc)\b", re.IGNORECASE)

# Decrypt post-sitemap pages that cover 2022 (confirmed by probing)
DECRYPT_2022_SITEMAP_PAGES = [12, 13, 14, 15, 16]

DECRYPT_SOURCE = FeedSource(
    name="Decrypt",
    url="https://decrypt.co/feed",
    category="crypto_news",
)


async def discover_decrypt_urls(client: httpx.AsyncClient) -> list[dict]:
    """Fetch 2022 BTC article stubs from Decrypt sitemaps."""
    articles: list[dict] = []
    for n in DECRYPT_2022_SITEMAP_PAGES:
        url = f"https://decrypt.co/post-sitemap{n}.xml"
        try:
            r = await client.get(url, timeout=20.0)
            r.raise_for_status()
        except Exception as exc:
            print(f"  SKIP {url}: {exc}")
            continue

        locs     = re.findall(r"<loc>(https://[^<]+)</loc>", r.text)
        lastmods = re.findall(r"<lastmod>([^<]+)</lastmod>", r.text)

        count = 0
        for loc, d in zip(locs, lastmods):
            if "2022" not in d:
                continue
            if not BTC_RE.search(loc):
                continue
            try:
                pub = datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
            except Exception:
                continue
            articles.append({"url": loc, "published": pub})
            count += 1

        print(f"  sitemap{n}: {count} BTC-2022 articles")

    return articles


async def fetch_content(
    fetcher: RSSFetcher,
    articles: list[dict],
    max_workers: int,
) -> list[dict]:
    """Fetch real article content using the crawler's site-specific selectors."""
    sem = asyncio.Semaphore(max_workers)
    rows: list[dict] = []
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

    async def _fetch_one(a: dict) -> dict | None:
        async with sem:
            try:
                content = await fetcher.fetch_article_content(a["url"])
            except Exception:
                return None
            if not content or len(content) < 150:
                return None
            if not BTC_RE.search(content[:3000]):
                return None
            # Use URL slug as title
            slug = a["url"].rstrip("/").rsplit("/", 1)[-1]
            title = slug.replace("-", " ").title()
            return {
                "header":          title[:200],
                "content":         content[:5000],
                "publish_at":      a["published"].isoformat(),
                "crawled_at":      now_str,
                "source_url":      a["url"],
                "sentiment_score": 0.0,
            }

    total = len(articles)
    tasks = [_fetch_one(a) for a in articles]
    done_count = 0
    ok_count = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        done_count += 1
        if result:
            rows.append(result)
            ok_count += 1
        if done_count % 50 == 0 or done_count == total:
            print(f"    [{done_count}/{total}] fetched={ok_count}", flush=True)
        await asyncio.sleep(0.05)

    return rows


async def write_to_supabase(http: httpx.AsyncClient, rows: list[dict]) -> int:
    """Upsert rows into Supabase news_articles, keyed on source_url."""
    if not rows:
        return 0
    endpoint = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?on_conflict=source_url"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    saved = 0
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        try:
            r = await http.post(endpoint, headers=headers, json=batch, timeout=30.0)
            if 200 <= r.status_code < 300:
                saved += len(batch)
                print(f"    saved batch {i//batch_size + 1}: {len(batch)} rows (total {saved})", flush=True)
            else:
                print(f"    SUPABASE ERROR {r.status_code}: {r.text[:200]}")
        except Exception as exc:
            print(f"    SUPABASE EXCEPTION: {exc}")
    return saved


async def main(dry_run: bool, workers: int, limit: int) -> None:
    if not dry_run and not (SUPABASE_URL and SUPABASE_KEY):
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        sys.exit(1)

    browser_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Step 1: discover URLs from Decrypt sitemaps
    print("Step 1: discovering 2022 BTC article URLs from Decrypt sitemaps...")
    async with httpx.AsyncClient(timeout=20.0, headers=browser_headers, follow_redirects=True) as client:
        articles = await discover_decrypt_urls(client)

    total_discovered = len(articles)
    print(f"  Total discovered: {total_discovered} BTC articles\n")

    if limit and limit < total_discovered:
        articles = articles[:limit]
        print(f"  [limit={limit}] processing first {limit} articles\n")

    if not articles:
        print("No articles found.")
        return

    # Date distribution
    by_month: dict[str, int] = {}
    for a in articles:
        key = a["published"].strftime("%Y-%m")
        by_month[key] = by_month.get(key, 0) + 1
    print("Date distribution:")
    for month, n in sorted(by_month.items()):
        print(f"  {month}: {n}")

    if dry_run:
        print("\n[DRY RUN] Sample articles:")
        for a in articles[:10]:
            print(f"  {a['published'].date()} | {a['url']}")
        print("\nRun without --dry-run to fetch content and write to Supabase.")
        return

    # Step 2: fetch real article content
    fetcher = RSSFetcher(
        sources=[DECRYPT_SOURCE],
        poll_interval_seconds=60,
        timeout_seconds=20,
    )
    # Override the fetcher's httpx client with browser headers
    await fetcher._client.aclose()
    fetcher._client = httpx.AsyncClient(
        timeout=20,
        headers=browser_headers,
        follow_redirects=True,
    )

    print(f"\nStep 2: fetching content for {len(articles)} articles ({workers} workers)...")
    rows = await fetch_content(fetcher, articles, max_workers=workers)
    await fetcher._client.aclose()

    print(f"  Got real content for {len(rows)}/{len(articles)} articles\n")

    if not rows:
        print("No usable content fetched.")
        return

    # Step 3: write to Supabase
    print(f"Step 3: writing {len(rows)} articles to Supabase {SUPABASE_TABLE}...")
    async with httpx.AsyncClient(timeout=30.0) as http:
        saved = await write_to_supabase(http, rows)

    print(f"\nDone. Saved {saved}/{len(rows)} articles with real content.")

    if saved > 0:
        today = datetime.date.today()
        jan1  = datetime.date(2022, 1, 1)
        dec31 = datetime.date(2022, 12, 31)
        off_start = (today - dec31).days
        off_end   = (today - jan1).days
        print("\nNext: rebuild 2022 StockMem records with real factors + sentiment:")
        print(f"  for offset in $(seq {off_start} 30 {off_end + 30}); do")
        print("    curl -s -X POST \"http://localhost:8005/backfill?symbol=BTC&days=30&offset=$offset\"")
        print("  done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl 2022 BTC news with real content from Decrypt")
    parser.add_argument("--dry-run", action="store_true", help="Discover URLs only, no writes")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent fetches (default 4)")
    parser.add_argument("--limit", type=int, default=0, help="Max articles to process (0 = all)")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, workers=args.workers, limit=args.limit))
