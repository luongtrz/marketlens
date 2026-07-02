"""Enrich existing backtest logs with ret_1d by fetching from market_data.

Reads a .log file (or JSON), fetches actual_return_1d for each date,
and saves an enriched JSON usable by portfolio_sim.py --json.

Usage:
    python scripts/enrich_log_ret1d.py --log /tmp/bt_qwen_combined.log --out /tmp/bt_qwen_enriched.json
    python scripts/enrich_log_ret1d.py --log /tmp/bt_kimi_full.log --out /tmp/bt_kimi_enriched.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

MARKET_URL = os.getenv("MARKET_URL", "http://localhost:8002")

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
            rows.append({"date": d, "signal": m.group(2), "confidence": float(m.group(3)),
                         "ret_1d": None, "ret_7d": ret7, "ret_30d": None})
    rows.sort(key=lambda r: r["date"])
    return rows


def parse_json(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return sorted(data.get("rows", []), key=lambda r: r["date"])


async def fetch_ret1d(http: httpx.AsyncClient, symbol: str, td: date) -> float | None:
    end_ts = int((datetime(td.year, td.month, td.day, tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000)
    try:
        r = await http.get(f"{MARKET_URL}/history",
                           params={"symbol": symbol, "interval": "1d", "limit": 5, "end_time": str(end_ts)})
        if r.status_code != 200:
            return None
        candles = r.json()
        base = next((c["close"] for c in reversed(candles) if c["timestamp"][:10] == td.isoformat()), None)
        if base is None:
            return None
        target = td + timedelta(days=1)
        t_end = int((datetime(target.year, target.month, target.day, tzinfo=timezone.utc) + timedelta(days=2)).timestamp() * 1000)
        r2 = await http.get(f"{MARKET_URL}/history",
                            params={"symbol": symbol, "interval": "1d", "limit": 5, "end_time": str(t_end)})
        if r2.status_code != 200:
            return None
        for c in reversed(r2.json()):
            if c["timestamp"][:10] <= target.isoformat():
                return round((c["close"] - base) / base * 100, 4)
    except Exception:
        return None
    return None


async def enrich(rows: list[dict], symbol: str, concurrency: int) -> list[dict]:
    lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    for i, row in enumerate(rows):
        await queue.put((i, row))
    done = 0

    async def worker():
        nonlocal done
        async with httpx.AsyncClient(timeout=30) as http:
            while True:
                try:
                    i, row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if row.get("ret_1d") is not None:
                    async with lock:
                        done += 1
                    continue
                td = date.fromisoformat(row["date"])
                ret1d = await fetch_ret1d(http, symbol, td)
                async with lock:
                    rows[i]["ret_1d"] = ret1d
                    done += 1
                    if done % 100 == 0 or done == len(rows):
                        print(f"  {done}/{len(rows)} enriched...", flush=True)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return rows


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", help="Input .log file")
    p.add_argument("--json", help="Input .json file (already has rows[])")
    p.add_argument("--out", required=True, help="Output enriched JSON path")
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
    already = sum(1 for r in rows if r.get("ret_1d") is not None)
    print(f"  {already} already have ret_1d, fetching {len(rows) - already} missing...")

    rows = await enrich(rows, args.symbol, args.concurrency)

    filled = sum(1 for r in rows if r.get("ret_1d") is not None)
    print(f"  Filled: {filled}/{len(rows)}")

    out = {"rows": rows}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
