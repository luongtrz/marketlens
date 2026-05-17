"""Run fixed-window backtest directly via /run for model comparison.

This script does not write to Supabase backtest_results.
It is intended for apples-to-apples model evaluation on the same dates.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import date, datetime, timedelta, timezone

import httpx


MAIN_URL = "http://localhost:8005"
MARKET_URL = "http://localhost:8002"
HORIZONS = {"1d": 1, "7d": 7, "30d": 30}


def check_correct(signal: str, actual_return: float, threshold: float = 1.0) -> bool | None:
    if signal == "BUY":
        return actual_return > 0
    if signal == "SELL":
        return actual_return < 0
    if signal == "HOLD":
        return abs(actual_return) < threshold
    return None


async def fetch_actual_returns(http: httpx.AsyncClient, symbol: str, td: date) -> dict[str, float | None]:
    out: dict[str, float | None] = {"1d": None, "7d": None, "30d": None}
    end_ts = int((datetime(td.year, td.month, td.day, tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000)
    r = await http.get(
        f"{MARKET_URL}/history",
        params={"symbol": symbol, "interval": "1d", "limit": 5, "end_time": str(end_ts)},
    )
    if r.status_code != 200:
        return out
    candles = r.json()
    match = None
    for c in reversed(candles):
        if c["timestamp"][:10] == td.isoformat():
            match = c
            break
    if match is None:
        return out
    base_close = match["close"]

    for label, off in HORIZONS.items():
        target = td + timedelta(days=off)
        if target > date.today() - timedelta(days=1):
            continue
        t_end = int((datetime(target.year, target.month, target.day, tzinfo=timezone.utc) + timedelta(days=2)).timestamp() * 1000)
        r2 = await http.get(
            f"{MARKET_URL}/history",
            params={"symbol": symbol, "interval": "1d", "limit": 5, "end_time": str(t_end)},
        )
        if r2.status_code != 200:
            continue
        candles2 = r2.json()
        for c2 in reversed(candles2):
            if c2["timestamp"][:10] <= target.isoformat():
                out[label] = round((c2["close"] - base_close) / base_close * 100, 4)
                break
    return out


async def run_one(
    http: httpx.AsyncClient,
    symbol: str,
    td: date,
    model: str | None = None,
) -> dict | None:
    t0 = time.time()
    params = {"symbol": symbol, "date": td.isoformat()}
    if model:
        params["model"] = model
    r = await http.post(f"{MAIN_URL}/run", params=params)
    if r.status_code != 200:
        return None
    run_id = r.json()["run_id"]
    for _ in range(40):
        await asyncio.sleep(2)
        sr = await http.get(f"{MAIN_URL}/status/{run_id}")
        if sr.status_code != 200:
            continue
        st = sr.json().get("status")
        if st in {"done", "failed"}:
            break
    rr = await http.get(f"{MAIN_URL}/result/{run_id}")
    if rr.status_code != 200:
        return None
    j = rr.json()
    return {
        "signal": j.get("signal", "HOLD"),
        "confidence": float(j.get("confidence", 0.0) or 0.0),
        "errors": j.get("errors", []),
        "duration_ms": int((time.time() - t0) * 1000),
    }


def build_dates(start: date, end: date, reverse: bool) -> list[date]:
    cur = start
    out: list[date] = []
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    out.sort(reverse=reverse)
    return out


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--model", default=None, help="Optional llm_gateway model override, e.g. qwen3.5-plus")
    args = p.parse_args()

    dates = build_dates(date.fromisoformat(args.start), date.fromisoformat(args.end), args.reverse)
    symbol = args.symbol.upper()
    print(f"Running compare backtest for {symbol}: {len(dates)} dates ({dates[0]} -> {dates[-1]})")

    stats = {
        "total": 0,
        "errors": 0,
        "dur_ms": 0,
        "c1": 0, "n1": 0,
        "c7": 0, "n7": 0,
        "c30": 0, "n30": 0,
    }

    async with httpx.AsyncClient(timeout=120) as http:
        for i, d in enumerate(dates, start=1):
            res = await run_one(http, symbol, d, model=args.model)
            if res is None:
                print(f"[{i}/{len(dates)}] {d} FAIL(no_result)")
                stats["errors"] += 1
                continue
            actual = await fetch_actual_returns(http, symbol, d)
            s = str(res["signal"])
            for h in ("1d", "7d", "30d"):
                v = actual.get(h)
                if v is None:
                    continue
                ok = check_correct(s, v)
                if ok is None:
                    continue
                stats[f"n{h.replace('d', '')}"] += 1
                if ok:
                    stats[f"c{h.replace('d', '')}"] += 1
            stats["total"] += 1
            stats["dur_ms"] += int(res["duration_ms"])
            r7 = actual.get("7d")
            r7s = f"{r7:+.2f}%" if isinstance(r7, (int, float)) else "N/A"
            print(
                f"[{i}/{len(dates)}] {d} {s}({res['confidence']:.2f}) "
                f"ret7d={r7s} dur={res['duration_ms']}ms"
            )

    avg_dur = (stats["dur_ms"] / stats["total"]) if stats["total"] else 0
    print("\n=== Summary ===")
    print(f"Runs: {stats['total']}  Errors: {stats['errors']}  AvgDur: {avg_dur:.0f}ms")
    if stats["n1"]:
        print(f"D+1  DA: {stats['c1']}/{stats['n1']} = {stats['c1']/stats['n1']*100:.1f}%")
    if stats["n7"]:
        print(f"D+7  DA: {stats['c7']}/{stats['n7']} = {stats['c7']/stats['n7']*100:.1f}%")
    if stats["n30"]:
        print(f"D+30 DA: {stats['c30']}/{stats['n30']} = {stats['c30']/stats['n30']*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
