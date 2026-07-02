#!/usr/bin/env python3
"""Pull the Supabase stockmem_records table to a local NDJSON file.

Usage:
    SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
    python3 scripts/pull_stockmem_records_from_supabase.py

    python3 scripts/pull_stockmem_records_from_supabase.py \
      --output data/exports/stockmem_records.ndjson \
      --symbol BTC
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "exports" / "stockmem_records.ndjson"


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def read_config() -> tuple[str, str]:
    env = {}
    env.update(load_env_file(ROOT / ".env"))
    env.update(load_env_file(ROOT / "aihub" / ".env"))
    env.update(os.environ)

    url = (env.get("SUPABASE_URL") or "").rstrip("/")
    key = env.get("SUPABASE_SERVICE_ROLE_KEY") or env.get("SUPABASE_ANON_KEY") or ""
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY are required.")
    return url, key


def fetch_batch(base_url: str, key: str, *, offset: int, limit: int, symbol: str | None) -> list[dict]:
    params: dict[str, str] = {
        "select": "id,record_date,symbol,payload",
        "order": "record_date.asc",
        "limit": str(limit),
        "offset": str(offset),
    }
    if symbol:
        params["symbol"] = f"eq.{symbol}"

    req = Request(
        f"{base_url}/rest/v1/stockmem_records?{urlencode(params)}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output NDJSON path.",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Optional symbol filter, for example BTC.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows to fetch per request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        base_url, key = read_config()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total = 0
    offset = 0
    with output.open("w", encoding="utf-8") as fh:
        while True:
            batch = fetch_batch(
                base_url,
                key,
                offset=offset,
                limit=args.batch_size,
                symbol=args.symbol,
            )
            if not batch:
                break

            for row in batch:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                fh.write("\n")

            total += len(batch)
            print(f"fetched {total} rows", file=sys.stderr)

            if len(batch) < args.batch_size:
                break
            offset += args.batch_size

    print(f"saved {total} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
