"""Fetch real article content for the 714 thin 2022 articles in Supabase.

Reads source_url from news_articles (publish_at in 2022, content < 200 chars),
fetches each CryptoSlate page, extracts article body text, updates Supabase.

After this script completes, run the backfill pipeline to build factor snapshots:
    for offset in 30 60 ... 390; do
        curl -s -X POST "http://localhost:8005/backfill?symbol=BTC&days=30&offset=$offset"
    done

Usage:
    python scripts/crawl_2022_content.py [--dry-run] [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

HEADERS_SUPABASE = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

HEADERS_CRAWL = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Rate limit: max concurrent fetches
SEMAPHORE = asyncio.Semaphore(3)


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def _extract_text(html: str) -> str:
    """Extract article body text from CryptoSlate HTML."""
    # Remove script/style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_article(html: str, url: str) -> str:
    """Try to extract the main article section from CryptoSlate page."""
    # CryptoSlate wraps article body in <div class="article-content"> or <article>
    for pattern in [
        r'<div[^>]+class="[^"]*article-content[^"]*"[^>]*>(.*?)</div>',
        r'<article[^>]*>(.*?)</article>',
        r'<div[^>]+class="[^"]*post-content[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+class="[^"]*entry-content[^"]*"[^>]*>(.*?)</div>',
    ]:
        m = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if m:
            content = _extract_text(m.group(1))
            if len(content) > 200:
                return content[:4000]

    # Fallback: extract all paragraph text
    paras = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL | re.IGNORECASE)
    text = " ".join(_extract_text(p) for p in paras if len(_extract_text(p)) > 40)
    return text[:4000] if len(text) > 200 else ""


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

async def fetch_thin_articles(http: httpx.AsyncClient) -> list[dict]:
    """Fetch all 2022 articles with thin content from Supabase."""
    results, offset = [], 0
    while True:
        r = await http.get(
            f"{SUPABASE_URL}/rest/v1/news_articles",
            headers={k: v for k, v in HEADERS_SUPABASE.items() if k != "Prefer"},
            params=[
                ("publish_at", "gte.2022-01-01"),
                ("publish_at", "lte.2022-12-31"),
                ("select", "id,source_url,header,content"),
                ("limit", "1000"),
                ("offset", str(offset)),
            ],
        )
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        # Only include articles with thin content
        thin = [a for a in batch if len(a.get("content") or "") < 300]
        results.extend(thin)
        if len(batch) < 1000:
            break
        offset += 1000
    return results


async def update_article_content(http: httpx.AsyncClient, article_id: str, content: str) -> bool:
    r = await http.patch(
        f"{SUPABASE_URL}/rest/v1/news_articles?id=eq.{article_id}",
        headers=HEADERS_SUPABASE,
        json={"content": content},
        timeout=15.0,
    )
    return 200 <= r.status_code < 300


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

async def fetch_article_content(
    http: httpx.AsyncClient,
    article: dict,
    dry_run: bool,
    semaphore: asyncio.Semaphore,
) -> tuple[str, bool, int]:
    """Returns (article_id, success, content_length)."""
    url = article.get("source_url", "")
    aid = article["id"]
    if not url:
        return aid, False, 0

    async with semaphore:
        try:
            r = await http.get(url, headers=HEADERS_CRAWL, timeout=20.0, follow_redirects=True)
            if r.status_code != 200:
                return aid, False, 0
            content = _extract_article(r.text, url)
            if not content or len(content) < 150:
                # Fallback: use full-page text (less precise but better than title)
                content = _extract_text(r.text)
                content = content[:3000] if len(content) > 300 else ""
            if not content:
                return aid, False, 0
            if not dry_run:
                ok = await update_article_content(http, aid, content)
                return aid, ok, len(content)
            return aid, True, len(content)
        except Exception:
            return aid, False, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(dry_run: bool, workers: int, limit: int) -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)

    async with httpx.AsyncClient(timeout=30.0) as http:
        print("Fetching thin 2022 articles from Supabase...")
        articles = await fetch_thin_articles(http)
        if limit > 0:
            articles = articles[:limit]
        total = len(articles)
        print(f"Found {total} articles with thin content")

        if not articles:
            print("Nothing to do.")
            return

        sem = asyncio.Semaphore(workers)
        ok_count = fail_count = 0
        total_chars = 0

        tasks = [fetch_article_content(http, a, dry_run, sem) for a in articles]

        done = 0
        for coro in asyncio.as_completed(tasks):
            aid, ok, chars = await coro
            done += 1
            if ok:
                ok_count += 1
                total_chars += chars
            else:
                fail_count += 1
            if done % 20 == 0 or done == total:
                print(
                    f"  [{done}/{total}] ok={ok_count} fail={fail_count} avg_chars={total_chars//max(ok_count,1)}",
                    flush=True,
                )
            # Small delay to avoid hammering the server
            await asyncio.sleep(0.3)

    mode = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode}Done: {ok_count}/{total} articles updated, avg content={total_chars//max(ok_count,1)} chars")

    if ok_count > 0 and not dry_run:
        print("\nNext step — run backfill for all 2022 windows:")
        print("  for offset in $(seq 30 30 420); do")
        print("    curl -s -X POST \"http://localhost:8005/backfill?symbol=BTC&days=30&offset=$offset\" | python3 -m json.tool")
        print("  done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't write to Supabase")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent fetches (default 3)")
    parser.add_argument("--limit", type=int, default=0, help="Limit articles (0 = all)")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, workers=args.workers, limit=args.limit))
