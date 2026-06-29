#!/usr/bin/env python3
"""Backfill Supabase ``news_articles.coin`` from article title/content/url.

Requires a ``coin text[]`` column and write access via ``SUPABASE_SERVICE_ROLE_KEY``.

Examples:
    python3 scripts/backfill_news_article_coin.py --dry-run
    python3 scripts/backfill_news_article_coin.py --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.asset_tags import detect_asset_tags  # noqa: E402


def _load_dotenv_optional() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in (REPO_ROOT / ".env", REPO_ROOT / "crawler" / ".env"):
        if path.exists():
            load_dotenv(path, override=False)


def _rest_headers(api_key: str) -> dict[str, str]:
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def _patch_headers(api_key: str) -> dict[str, str]:
    headers = _rest_headers(api_key)
    headers["Content-Type"] = "application/json"
    headers["Prefer"] = "return=minimal"
    return headers


def _current_coin(row: dict[str, Any]) -> list[str]:
    raw = row.get("coin")
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        value = str(item).strip()
        if not value:
            continue
        if value.upper() == "GENERAL":
            values.append("General")
        else:
            values.append(value.upper())
    return sorted(values)


def _coin_value(row: dict[str, Any]) -> list[str]:
    tags = sorted(
        detect_asset_tags(
            str(row.get("header") or ""),
            str(row.get("content") or ""),
            str(row.get("source_url") or ""),
        )
    )
    return tags if tags else ["General"]


async def _fetch_page(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    table: str,
    *,
    page_size: int,
    offset: int,
) -> list[dict[str, Any]]:
    endpoint = f"{base_url.rstrip('/')}/rest/v1/{table}"
    params = {
        "select": "id,header,content,source_url,coin",
        "order": "id.asc",
        "limit": str(page_size),
        "offset": str(offset),
    }
    for attempt in range(5):
        resp = await client.get(endpoint, headers=_rest_headers(api_key), params=params)
        try:
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {429, 500, 502, 503, 504} and attempt < 4:
                await asyncio.sleep(0.75 * (2**attempt))
                continue
            raise
    data = resp.json()
    return data if isinstance(data, list) else []


async def _patch_row(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    api_key: str,
    table: str,
    row_id: object,
    coin: list[str],
) -> None:
    endpoint = f"{base_url.rstrip('/')}/rest/v1/{table}"
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.patch(
                    endpoint,
                    headers=_patch_headers(api_key),
                    params={"id": f"eq.{row_id}"},
                    json={"coin": coin},
                )
                resp.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < 3:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise


async def _run(args: argparse.Namespace, base_url: str, api_key: str, table: str) -> int:
    stats = {
        "scanned": 0,
        "changed": 0,
        "patched": 0,
        "btc": 0,
        "eth": 0,
        "btc_eth": 0,
        "none": 0,
        "patch_errors": 0,
    }

    sem = asyncio.Semaphore(max(1, args.concurrency))
    timeout = httpx.Timeout(60.0)
    limits = httpx.Limits(max_connections=args.concurrency + 10, max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        offset = 0
        while True:
            rows = await _fetch_page(
                client,
                base_url,
                api_key,
                table,
                page_size=args.page_size,
                offset=offset,
            )
            if not rows:
                break

            pending: list[tuple[object, list[str]]] = []
            for row in rows:
                stats["scanned"] += 1
                coin = _coin_value(row)

                if coin == ["BTC", "ETH"]:
                    stats["btc_eth"] += 1
                elif coin == ["BTC"]:
                    stats["btc"] += 1
                elif coin == ["ETH"]:
                    stats["eth"] += 1
                else:
                    stats["none"] += 1

                if coin != _current_coin(row):
                    stats["changed"] += 1
                    pending.append((row.get("id"), coin))

            if args.dry_run:
                for row_id, coin in pending[: args.print_limit]:
                    print(f"[dry-run] id={row_id} coin -> {coin}")
            else:
                if args.max_updates:
                    cap = args.max_updates - stats["patched"]
                    if cap <= 0:
                        print(stats)
                        return 0
                    pending = pending[:cap]

                tasks = [
                    _patch_row(client, sem, base_url, api_key, table, row_id, coin)
                    for row_id, coin in pending
                ]
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, BaseException):
                            stats["patch_errors"] += 1
                            print(f"PATCH failed: {result}", file=sys.stderr)
                        else:
                            stats["patched"] += 1
                    print(f"scanned={stats['scanned']} patched={stats['patched']}", flush=True)

                if args.max_updates and stats["patched"] >= args.max_updates:
                    print(stats)
                    return 0

            offset += len(rows)

    print(stats)
    return 0 if stats["patch_errors"] == 0 else 2


def main() -> int:
    _load_dotenv_optional()

    parser = argparse.ArgumentParser(description="Backfill Supabase news_articles.coin.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes only, no PATCH.")
    parser.add_argument("--page-size", type=int, default=500, help="Rows per GET request.")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent PATCH requests.")
    parser.add_argument("--max-updates", type=int, default=0, help="Stop after N patches; 0 = no cap.")
    parser.add_argument("--print-limit", type=int, default=20, help="Max dry-run rows printed per page.")
    args = parser.parse_args()

    base_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    api_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
    table = os.getenv("SUPABASE_TABLE", "news_articles")
    if not base_url or not api_key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.", file=sys.stderr)
        return 1

    return asyncio.run(_run(args, base_url, api_key, table))


if __name__ == "__main__":
    raise SystemExit(main())
