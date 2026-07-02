"""Build daily_factor_snapshots for 2022 using the LLM gateway (OpenCode Go).

Reads the real 2022 BTC articles saved in Supabase, groups them by date,
calls the LLM gateway /complete endpoint with the SKGP factor-extraction prompt,
then upserts daily_factor_snapshots so that POST /backfill uses the pre-computed
factor data instead of calling the AIHub Gemini path.

After this runs, re-run /backfill for 2022 offsets:
    for offset in $(seq 1233 30 1657); do
        curl -s -X POST "http://localhost:8005/backfill?symbol=BTC&days=30&offset=$offset"
    done

Usage:
    python scripts/build_2022_factor_snapshots.py [--dry-run] [--date YYYY-MM-DD]
    python scripts/build_2022_factor_snapshots.py --dry-run   # print without writing
    python scripts/build_2022_factor_snapshots.py             # process all 2022 dates
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from stockmem.src.search.taxonomy import (
    build_type_vector, build_group_vector,
    BEARISH_FACTORS, BULLISH_FACTORS, NEUTRAL_FACTORS,
)

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8006")
SYMBOL = "BTC"

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates,return=minimal",
}

# Build taxonomy phrase list once (100 phrases)
_ALL_PHRASES: list[str] = (
    list(BEARISH_FACTORS.keys())
    + list(BULLISH_FACTORS.keys())
    + list(NEUTRAL_FACTORS.keys())
)
_TAXONOMY_LIST = "\n".join(f"- {p}" for p in _ALL_PHRASES)

_SKGP_SYSTEM = (
    "You are a JSON API. Output ONLY raw JSON — no explanations, no markdown, no reasoning."
)

_PHRASES_JSON_RE = re.compile(r'\{"phrases"\s*:\s*\[.*?\]\s*\}', re.DOTALL)


# ---------------------------------------------------------------------------
# Fetch 2022 articles from Supabase
# ---------------------------------------------------------------------------

BTC_RE_FILTER = re.compile(r"\b(bitcoin|btc)\b", re.IGNORECASE)


async def fetch_2022_articles(http: httpx.AsyncClient) -> dict[datetime.date, list[dict]]:
    """Return articles grouped by publication date (only articles with real content)."""
    by_date: dict[datetime.date, list[dict]] = {}
    offset = 0

    while True:
        params = [
            ("publish_at", "gte.2022-01-01"),
            ("publish_at", "lte.2022-12-31"),
            ("select", "id,header,content,publish_at,source_url"),
            ("limit", "1000"),
            ("offset", str(offset)),
        ]
        r = await http.get(
            f"{SUPABASE_URL}/rest/v1/news_articles",
            headers={k: v for k, v in SUPABASE_HEADERS.items() if k != "Prefer"},
            params=params,
            timeout=20.0,
        )
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break

        for a in batch:
            content = a.get("content") or ""
            header = a.get("header") or ""
            # Only keep articles with real content AND BTC relevance
            if len(content) < 150:
                continue
            if not BTC_RE_FILTER.search(header) and not BTC_RE_FILTER.search(content[:500]):
                continue
            try:
                d = datetime.date.fromisoformat(str(a["publish_at"])[:10])
            except Exception:
                continue
            by_date.setdefault(d, []).append(a)

        if len(batch) < 1000:
            break
        offset += 1000

    return by_date


# ---------------------------------------------------------------------------
# LLM factor extraction via LLM gateway /complete
# ---------------------------------------------------------------------------

async def extract_factors_via_llm_gateway(
    http: httpx.AsyncClient,
    articles: list[dict],
) -> list[str]:
    """Select matching taxonomy phrases for the given articles via LLM gateway.

    Returns a list of exact FACTOR_TYPE_MAP phrases — these map directly
    to bits in the 75-dim factor_vector without any approximation.
    """
    texts: list[str] = []
    for a in articles[:5]:
        t = (a.get("content") or a.get("header") or "").strip()
        if t:
            texts.append(t[:300])

    combined = "\n---\n".join(texts)
    prompt = (
        "News about Bitcoin:\n"
        + combined
        + "\n\nFrom the list below, select ALL phrases that describe what is happening in this news.\n"
        "Output ONLY this JSON: "
        + '{"phrases": ["Fed raises interest rate", "Large market liquidations"]}'
        + "\n\nAvailable phrases:\n"
        + _TAXONOMY_LIST
    )

    try:
        r = await http.post(
            f"{LLM_GATEWAY_URL}/complete",
            params={"max_tokens": 2000},
            json={"prompt": prompt, "system": _SKGP_SYSTEM},
            timeout=60.0,
        )
        r.raise_for_status()
        raw_text = r.json().get("text", "")
    except Exception as exc:
        print(f"    LLM gateway error: {exc}")
        return []

    # Find {"phrases": [...]} in response (model may reason first)
    m = _PHRASES_JSON_RE.search(raw_text)
    if m:
        try:
            data = json.loads(m.group(0))
            phrases = data.get("phrases", [])
            # Validate: only keep phrases that exist in taxonomy
            valid = [p for p in phrases if p in set(_ALL_PHRASES)]
            return valid
        except json.JSONDecodeError:
            pass

    try:
        data = json.loads(raw_text)
        phrases = data.get("phrases", [])
        return [p for p in phrases if p in set(_ALL_PHRASES)]
    except json.JSONDecodeError:
        return []


def _build_factor_vector(factor_names: list[str]) -> list[int]:
    """Build 75-dim binary vector from factor names using taxonomy."""
    tv = build_type_vector(factor_names)   # 62d
    gv = build_group_vector(factor_names)  # 13d
    return [int(x) for x in tv + gv]


# ---------------------------------------------------------------------------
# Upsert into daily_factor_snapshots
# ---------------------------------------------------------------------------

async def upsert_snapshot(
    http: httpx.AsyncClient,
    snapshot_date: datetime.date,
    factors: list[dict],
    factor_vector: list[int],
) -> bool:
    """Upsert one row into daily_factor_snapshots."""
    row = {
        "snapshot_date": snapshot_date.isoformat(),
        "symbol": SYMBOL,
        "factors_json": json.dumps(factors),
        "factor_vector": factor_vector,
    }
    r = await http.post(
        f"{SUPABASE_URL}/rest/v1/daily_factor_snapshots?on_conflict=snapshot_date,symbol",
        headers=SUPABASE_HEADERS,
        json=row,
        timeout=15.0,
    )
    return 200 <= r.status_code < 300


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(dry_run: bool, only_date: datetime.date | None) -> None:
    if not dry_run and not (SUPABASE_URL and SUPABASE_KEY):
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)

    async with httpx.AsyncClient(timeout=20.0) as http:
        print("Fetching 2022 BTC articles from Supabase...")
        by_date = await fetch_2022_articles(http)
        print(f"  Found articles on {len(by_date)} distinct dates\n")

        if not by_date:
            print("No articles found — run crawl_2022_news.py first.")
            return

        dates_to_process = sorted(by_date.keys())
        if only_date:
            dates_to_process = [d for d in dates_to_process if d == only_date]

        ok_count = fail_count = 0
        total = len(dates_to_process)

        for i, d in enumerate(dates_to_process):
            articles = by_date[d]
            print(f"  [{i+1}/{total}] {d}  ({len(articles)} articles)", end=" ", flush=True)

            phrases = await extract_factors_via_llm_gateway(http, articles)
            factor_vector = _build_factor_vector(phrases)
            active_bits = sum(factor_vector)

            print(f"phrases={len(phrases)} bits={active_bits}/75", end="")

            if dry_run:
                print(f"  [DRY] {phrases[:3]}")
                ok_count += 1
                continue

            # Store phrases as factor dicts for factors_json column
            factors_raw = [{"factor": p} for p in phrases]
            ok = await upsert_snapshot(http, d, factors_raw, factor_vector)
            if ok:
                ok_count += 1
                print(" ✓")
            else:
                fail_count += 1
                print(" ✗ (Supabase error)")

            # Small delay between LLM calls
            await asyncio.sleep(0.3)

    print(f"\nDone. Processed={ok_count}  Failed={fail_count}")
    if ok_count > 0 and not dry_run:
        today = datetime.date.today()
        jan1  = datetime.date(2022, 1, 1)
        dec31 = datetime.date(2022, 12, 31)
        off_start = (today - dec31).days
        off_end   = (today - jan1).days
        print("\nNext: re-run 2022 backfill (will use pre-computed factor snapshots):")
        print(f"  for offset in $(seq {off_start} 30 {off_end + 30}); do")
        print("    curl -s -X POST \"http://localhost:8005/backfill?symbol=BTC&days=30&offset=$offset\"")
        print("  done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Extract factors but don't write to Supabase")
    parser.add_argument("--date", default=None, help="Process only this date YYYY-MM-DD")
    args = parser.parse_args()

    only_date = datetime.date.fromisoformat(args.date) if args.date else None
    asyncio.run(main(dry_run=args.dry_run, only_date=only_date))
