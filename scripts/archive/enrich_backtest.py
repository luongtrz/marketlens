"""Enrich backtest logs/JSON with future returns 1d/3d/7d/15d/30d from market_data.

Reads an existing .log or .json, fetches missing return horizons, saves enriched JSON.

Usage:
    python scripts/enrich_backtest.py --log /tmp/bt_qwen_combined.log --out data/backtests/qwen.json
    python scripts/enrich_backtest.py --log /tmp/bt_kimi_full.log    --out data/backtests/kimi.json
    python scripts/enrich_backtest.py --json /tmp/bt_deepseek_supabase.json --out data/backtests/deepseek.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

BINANCE_URL = "https://api.binance.com/api/v3/klines"
HORIZONS = {"1d": 1, "3d": 3, "7d": 7, "15d": 15, "30d": 30}

LOG_RE = re.compile(
    r"\[\d+/\d+\]\s+(\d{4}-\d{2}-\d{2})\s+(BUY|SELL|HOLD)\((\d+\.\d+)\)"
    r".*ret7d=([+-]?\d+\.\d+%|N/A)"
)


def parse_log(path: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            d = m.group(1)
            if d in seen:
                continue
            seen.add(d)
            ret7 = None if m.group(4) == "N/A" else float(m.group(4).rstrip("%"))
            rows.append({
                "date": d, "signal": m.group(2), "confidence": float(m.group(3)),
                "ret_1d": None, "ret_3d": None, "ret_7d": ret7, "ret_15d": None, "ret_30d": None,
            })
    rows.sort(key=lambda r: r["date"])
    return rows


def parse_json(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data.get("rows", []):
        rows.append({
            "date": r["date"],
            "signal": r["signal"],
            "confidence": r["confidence"],
            "ret_1d":  r.get("ret_1d")  or r.get("actual_return_1d"),
            "ret_3d":  r.get("ret_3d")  or r.get("actual_return_3d"),
            "ret_7d":  r.get("ret_7d")  or r.get("actual_return_7d"),
            "ret_15d": r.get("ret_15d") or r.get("actual_return_15d"),
            "ret_30d": r.get("ret_30d") or r.get("actual_return_30d"),
        })
    rows.sort(key=lambda r: r["date"])
    return rows


async def fetch_close(http: httpx.AsyncClient, symbol: str, td: date) -> float | None:
    """Fetch closing price for a specific date from Binance."""
    start_ms = int(datetime(td.year, td.month, td.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = start_ms + 86_400_000  # +1 day in ms
    try:
        r = await http.get(BINANCE_URL, params={
            "symbol": f"{symbol}USDT", "interval": "1d",
            "startTime": start_ms, "endTime": end_ms, "limit": 1,
        })
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        return float(data[0][4])  # index 4 = close price
    except Exception:
        return None


async def fetch_returns(http: httpx.AsyncClient, symbol: str, td: date) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: None for k in HORIZONS}
    base = await fetch_close(http, symbol, td)
    if base is None:
        return out
    for label, offset in HORIZONS.items():
        target = td + timedelta(days=offset)
        if target > date.today() - timedelta(days=1):
            continue
        close = await fetch_close(http, symbol, target)
        if close is not None:
            out[label] = round((close - base) / base * 100, 4)
    return out


async def enrich(rows: list[dict], symbol: str, concurrency: int) -> list[dict]:
    needs_fetch = [
        i for i, r in enumerate(rows)
        if any(r.get(f"ret_{h}") is None for h in HORIZONS)
    ]
    print(f"  {len(rows) - len(needs_fetch)} rows already complete, fetching {len(needs_fetch)} rows...")

    queue: asyncio.Queue = asyncio.Queue()
    for i in needs_fetch:
        await queue.put(i)
    lock = asyncio.Lock()
    done = 0

    async def worker():
        nonlocal done
        async with httpx.AsyncClient(timeout=30) as http:
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                row = rows[i]
                td = date.fromisoformat(row["date"])
                fetched = await fetch_returns(http, symbol, td)
                async with lock:
                    for label in HORIZONS:
                        if row.get(f"ret_{label}") is None:
                            rows[i][f"ret_{label}"] = fetched.get(label)
                    done += 1
                    if done % 200 == 0 or done == len(needs_fetch):
                        print(f"  {done}/{len(needs_fetch)} fetched...", flush=True)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return rows


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log",  help="Input .log file")
    p.add_argument("--json", help="Input .json file")
    p.add_argument("--out",  required=True, help="Output enriched JSON path")
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    if args.log:
        rows = parse_log(args.log)
        label = Path(args.log).stem
    elif args.json:
        rows = parse_json(args.json)
        label = Path(args.json).stem
    else:
        p.error("Specify --log or --json")

    print(f"Loaded {len(rows)} rows from {label}")
    print(f"  Date range: {rows[0]['date']} → {rows[-1]['date']}")

    rows = await enrich(rows, args.symbol, args.concurrency)

    filled = {h: sum(1 for r in rows if r.get(f"ret_{h}") is not None) for h in HORIZONS}
    print(f"  Coverage: " + "  ".join(f"ret_{h}={filled[h]}/{len(rows)}" for h in HORIZONS))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows}, f)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
