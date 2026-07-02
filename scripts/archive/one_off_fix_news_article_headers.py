#!/usr/bin/env python3
"""One-off: fix news_articles rows where ``header`` was stored as the article URL.

Uses the same title logic as the crawler (:func:`normalize_article_title`).
Requires write access via ``SUPABASE_SERVICE_ROLE_KEY``. Safe to discard after running once.

Updates use concurrent PATCH requests (see ``--concurrency``) so large tables finish in minutes
instead of hours.

Important:

- **Live PATCH** each GET uses ``offset=0``. Patched rows drop out of the URL filter; a growing OFFSET
  would skip the next block (pagination bug on mutating sets).
- **``--dry-run``** uses increasing OFFSET because nothing is PATCHed — the candidate set does not shrink.

Examples::

    cd /path/to/marketlens
    python3 scripts/one_off_fix_news_article_headers.py --dry-run
    python3 scripts/one_off_fix_news_article_headers.py --concurrency 40
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from crawler.src.rss.title_hints import normalize_article_title  # noqa: E402


def _load_dotenv_optional() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (REPO_ROOT / ".env", REPO_ROOT / "crawler" / ".env"):
        if p.exists():
            load_dotenv(p)


def _needs_fix(header: str, source_url: str) -> bool:
    """True when ``header`` is clearly the canonical link, not a human headline."""
    h = header.strip()
    u = (source_url or "").strip()
    if not u or not h:
        return False
    if h.startswith(("http://", "https://", "//")):
        return True
    if h.rstrip("/") == u.rstrip("/"):
        return True
    return False


def _rest_headers(api_key: str) -> dict[str, str]:
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def _patch_headers(api_key: str) -> dict[str, str]:
    h = _rest_headers(api_key)
    h["Content-Type"] = "application/json"
    h["Prefer"] = "return=minimal"
    return h


async def _fetch_page(
    client: httpx.AsyncClient,
    base: str,
    api_key: str,
    table: str,
    *,
    page_size: int,
    offset: int,
) -> list[dict]:
    """URL-shaped headers: ``ILIKE`` patterns use ``*`` as the SQL ``%`` wildcard (PostgREST)."""
    endpoint = f"{base.rstrip('/')}/rest/v1/{table}"
    params: dict[str, str | int] = {
        "select": "id,header,source_url",
        "or": "(header.ilike.http://*,header.ilike.https://*,header.ilike.//*)",
        "order": "id.asc",
        "limit": page_size,
        "offset": offset,
    }
    resp = await client.get(endpoint, headers=_rest_headers(api_key), params=params)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


async def _patch_row(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base: str,
    api_key: str,
    table: str,
    row_id: object,
    new_header: str,
) -> None:
    ep = f"{base.rstrip('/')}/rest/v1/{table}"
    async with sem:
        for attempt in range(4):
            try:
                r = await client.patch(
                    ep,
                    headers=_patch_headers(api_key),
                    params={"id": f"eq.{row_id}"},
                    json={"header": new_header},
                )
                r.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < 3:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise


async def _run(args: argparse.Namespace, base: str, key: str, table: str) -> int:
    stats: dict[str, int] = {
        "scanned": 0,
        "considered": 0,
        "patched": 0,
        "skipped_no_url": 0,
        "skipped_no_improvement": 0,
        "patch_errors": 0,
    }

    sem = asyncio.Semaphore(max(1, args.concurrency))
    timeout = httpx.Timeout(60.0)
    limits = httpx.Limits(max_connections=args.concurrency + 10, max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        list_offset = 0
        page_idx = 0
        while True:
            page_idx += 1
            fetch_offset = list_offset if args.dry_run else 0
            rows = await _fetch_page(
                client,
                base,
                key,
                table,
                page_size=args.page_size,
                offset=fetch_offset,
            )
            if not rows:
                break

            pending: list[tuple[object, str]] = []
            for row in rows:
                stats["scanned"] += 1
                rid = row.get("id")
                header = str(row.get("header") or "")
                source_url = str(row.get("source_url") or "")

                if not source_url.strip():
                    stats["skipped_no_url"] += 1
                    continue
                if not _needs_fix(header, source_url):
                    continue

                stats["considered"] += 1
                new_h = (normalize_article_title(header, source_url) or "").strip()
                new_h = new_h[:200]

                if not new_h or new_h == "Untitled":
                    stats["skipped_no_improvement"] += 1
                    continue
                if new_h == header[:200]:
                    stats["skipped_no_improvement"] += 1
                    continue

                pending.append((rid, new_h))

            if args.dry_run:
                for rid, new_h in pending:
                    print(f"[dry-run] id={rid} patch header -> {new_h!r}")
                list_offset += len(rows)
            else:
                if args.max_updates:
                    cap = args.max_updates - stats["patched"]
                    if cap <= 0:
                        print(stats)
                        return 0
                    pending = pending[:cap]

                if pending:
                    print(
                        f"PATCH {len(pending)} rows (page={page_idx}, total patched so far {stats['patched']})…",
                        flush=True,
                    )
                    tasks = [
                        _patch_row(client, sem, base, key, table, rid, nh) for rid, nh in pending
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for res in results:
                        if isinstance(res, BaseException):
                            stats["patch_errors"] += 1
                            print(f"PATCH failed: {res}", file=sys.stderr)
                        else:
                            stats["patched"] += 1

                if args.max_updates and stats["patched"] >= args.max_updates:
                    print(stats)
                    return 0
                if (
                    not pending
                    and rows
                    and len(rows) >= args.page_size
                ):
                    print(
                        "Stopping: full page matched the URL filter but no row was patchable "
                        "(all skipped or errors). Avoiding an infinite loop — check data or relax logic.",
                        file=sys.stderr,
                    )
                    print(stats)
                    return 3

            if len(rows) < args.page_size:
                break

    print(stats)
    return 0 if stats["patch_errors"] == 0 else 2


def main() -> int:
    _load_dotenv_optional()

    parser = argparse.ArgumentParser(description="Fix URL-shaped news article headers on Supabase.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only, no PATCH.")
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Rows per GET request (default 200).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Max concurrent PATCH requests (default 32).",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Stop after this many successful updates (0 = no cap).",
    )
    args = parser.parse_args()

    base = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
    table = os.getenv("SUPABASE_TABLE", "news_articles")

    if not base or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.", file=sys.stderr)
        return 1

    return asyncio.run(_run(args, base, key, table))


if __name__ == "__main__":
    raise SystemExit(main())
