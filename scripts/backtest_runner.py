"""Bulk walk-forward backtest runner.

Runs POST /run for every available date, compares prediction vs actual forward
returns, writes results to Supabase ``backtest_results``.

Usage:
    python scripts/backtest_runner.py --symbol BTC --max 50
    python scripts/backtest_runner.py --symbol BTC --start 2024-01-01 --end 2024-12-31
    python scripts/backtest_runner.py --symbol BTC --stats          # print accuracy only

Design:
  - Ground truth: Binance OHLCV via market_data:8002 (D+1/D+7/D+30 % return).
  - Correctness: BUY  is correct if actual_return > 0
                 SELL is correct if actual_return < 0
                 HOLD is correct if |actual_return| < 1%
  - Each run calls AIHub → LLM (Groq) → costs API tokens.
  - Skips dates where D+30 future data isn't available yet.
  - Upserts into Supabase backtest_results so it's resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip("\"'")
            if k:
                os.environ[k] = v

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
MAIN_URL = "http://localhost:8005"
MARKET_URL = "http://localhost:8002"

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""

HORIZONS = {"1d": 1, "7d": 7, "30d": 30}


def supabase_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


# ── Correctness ───────────────────────────────────────────────────────────────
def check_correct(signal: str, actual_return: float, threshold: float = 1.0) -> bool | None:
    """Return True if signal direction matches actual return.

    HOLD is correct if |return| < threshold (flat market).
    Returns None if signal is unknown.
    """
    if signal == "BUY":
        return actual_return > 0
    if signal == "SELL":
        return actual_return < 0
    if signal == "HOLD":
        return abs(actual_return) < threshold
    return None


# ── Market Data ───────────────────────────────────────────────────────────────
async def fetch_actual_returns(http: httpx.AsyncClient, symbol: str, td: date) -> dict[str, float | None]:
    """Get actual forward returns for a date from Binance."""
    result: dict[str, float | None] = {"1d": None, "7d": None, "30d": None}

    # Get close price on target date
    end_ts = int((datetime(td.year, td.month, td.day, tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000)
    r = await http.get(f"{MARKET_URL}/history", params={"symbol": symbol, "interval": "1d", "limit": 5, "end_time": str(end_ts)})
    if r.status_code != 200:
        return result
    candles = r.json()
    match = None
    for c in reversed(candles):
        if c["timestamp"][:10] == td.isoformat():
            match = c
            break
    if match is None:
        return result
    base_close = match["close"]

    for label, offset_days in HORIZONS.items():
        target = td + timedelta(days=offset_days)
        # Check if target is too recent (no data yet)
        if target > date.today() - timedelta(days=1):
            continue
        t_end = int((datetime(target.year, target.month, target.day, tzinfo=timezone.utc) + timedelta(days=2)).timestamp() * 1000)
        r2 = await http.get(f"{MARKET_URL}/history", params={"symbol": symbol, "interval": "1d", "limit": 5, "end_time": str(t_end)})
        if r2.status_code != 200:
            continue
        candles2 = r2.json()
        for c2 in reversed(candles2):
            if c2["timestamp"][:10] <= target.isoformat():
                result[label] = round((c2["close"] - base_close) / base_close * 100, 4)
                break

    return result


# ── Supabase helpers ──────────────────────────────────────────────────────────
async def supabase_select(http: httpx.AsyncClient, table: str, params: list[tuple[str, str]]) -> list[dict]:
    r = await http.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=supabase_headers(), params=params)
    if 200 <= r.status_code < 300:
        data = r.json()
        return data if isinstance(data, list) else []
    print(f"  Supabase SELECT error {r.status_code}: {r.text[:200]}")
    return []


async def supabase_upsert(http: httpx.AsyncClient, table: str, row: dict, on_conflict: str = "symbol,backtest_date") -> bool:
    """Upsert a row. Returns True on success."""
    h = supabase_headers()
    h["Prefer"] = "resolution=merge-duplicates"
    params = [("on_conflict", on_conflict)]
    r = await http.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=h, params=params, json=row)
    ok = 200 <= r.status_code < 300 or r.status_code == 409
    if not ok:
        print(f"  Supabase UPSERT error {r.status_code}: {r.text[:200]}")
    return ok


async def get_completed_dates(http: httpx.AsyncClient, symbol: str) -> set[str]:
    """Return set of dates already in backtest_results."""
    rows = await supabase_select(http, "backtest_results", [
        ("select", "backtest_date"),
        ("symbol", f"eq.{symbol.upper()}"),
        ("signal", "not.is.null"),  # only completed rows
        ("order", "backtest_date.desc"),
        ("limit", "5000"),
    ])
    return {r["backtest_date"] for r in rows}


async def get_available_dates(http: httpx.AsyncClient, symbol: str,
                               start: date | None, end: date | None) -> list[date]:
    """Get all dates that have daily_factor_snapshots (i.e., can be backtested)."""
    filters: list[tuple[str, str]] = [
        ("select", "snapshot_date"),
        ("symbol", f"eq.{symbol.upper()}"),
        ("order", "snapshot_date.asc"),
        ("limit", "5000"),
    ]
    if start:
        filters.append(("snapshot_date", f"gte.{start.isoformat()}"))
    if end:
        filters.append(("snapshot_date", f"lte.{end.isoformat()}"))
    rows = await supabase_select(http, "daily_factor_snapshots", filters)
    dates = []
    for r in rows:
        d = r.get("snapshot_date", "")
        if d:
            dates.append(date.fromisoformat(d[:10]))
    return sorted(set(dates))


# ── Main runner ───────────────────────────────────────────────────────────────
async def run_backtest(http: httpx.AsyncClient, symbol: str, td: date) -> dict | None:
    """Run a single backtest, return result dict or None on failure."""
    t0 = time.time()

    # 1. Call /run
    r = await http.post(f"{MAIN_URL}/run", params={"symbol": symbol, "date": td.isoformat()})
    if r.status_code != 200:
        print(f"  /run failed: {r.status_code} {r.text[:100]}")
        return None
    run_id = r.json()["run_id"]

    # 2. Poll for result (max 90s)
    for _ in range(30):
        await asyncio.sleep(3)
        sr = await http.get(f"{MAIN_URL}/status/{run_id}")
        if sr.status_code != 200:
            continue
        status = sr.json().get("status")
        if status == "done":
            break
        if status == "failed":
            rr = await http.get(f"{MAIN_URL}/result/{run_id}")
            if rr.status_code == 200:
                j = rr.json()
                return {"run_id": run_id, "signal": j.get("signal", "HOLD"),
                        "confidence": j.get("confidence", 0), "sentiment_score": j.get("sentiment_score", 0),
                        "explanation": j.get("explanation", ""), "errors": j.get("errors", []),
                        "similar_cases": j.get("similar_cases", []),
                        "run_duration_ms": int((time.time() - t0) * 1000)}
            return None
    else:
        print(f"  Timeout waiting for {run_id}")
        return None

    # 3. Fetch result
    rr = await http.get(f"{MAIN_URL}/result/{run_id}")
    if rr.status_code != 200:
        return None
    j = rr.json()
    duration_ms = int((time.time() - t0) * 1000)

    sim_cases = j.get("similar_cases", [])
    top = sim_cases[0] if sim_cases else {}
    top_rec = top.get("record", {})

    return {
        "run_id": run_id,
        "signal": j.get("signal", "HOLD"),
        "confidence": j.get("confidence", 0),
        "sentiment_score": j.get("sentiment_score", 0),
        "explanation": j.get("explanation", ""),
        "errors": j.get("errors", []),
        "similar_cases": sim_cases,
        "n_similar_cases": len(sim_cases),
        "top_similar_date": top_rec.get("date"),
        "top_similar_similarity": top.get("similarity"),
        "top_similar_ret7d": top_rec.get("future_return_7d"),
        "run_duration_ms": duration_ms,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk walk-forward backtest")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (default: all available)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (default: today - 30d)")
    parser.add_argument("--max", type=int, default=0, help="Max runs (0 = unlimited)")
    parser.add_argument("--warmup", type=int, default=60, help="Skip first N days from accuracy stats (still run & save)")
    parser.add_argument("--reverse", action="store_true", help="Run newest dates first (2026 -> 2023). Better when StockMem already has data.")
    parser.add_argument("--stats", action="store_true", help="Only print accuracy stats, no new runs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run, don't execute")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=30)

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")
        return

    async with httpx.AsyncClient(timeout=120) as http:
        # Health checks
        for svc, url in [("main_controller", MAIN_URL), ("market_data", MARKET_URL)]:
            try:
                r = await http.get(f"{url}/health", timeout=5)
                if r.status_code != 200:
                    print(f"ERROR: {svc} not healthy")
                    return
            except Exception as e:
                print(f"ERROR: {svc} unreachable: {e}")
                return

        # ── Stats mode: read existing results ──
        if args.stats:
            rows = await supabase_select(http, "backtest_results", [
                ("select", "signal,confidence,sentiment_score,correct_1d,correct_7d,correct_30d,actual_return_7d,backtest_date"),
                ("symbol", f"eq.{symbol}"),
                ("signal", "not.is.null"),
                ("order", "backtest_date.desc"),
                ("limit", "5000"),
            ])
            if not rows:
                print("No results yet.")
                return
            total = len(rows)
            d1 = sum(1 for r in rows if r.get("correct_1d") is True)
            d7 = sum(1 for r in rows if r.get("correct_7d") is True)
            d30 = sum(1 for r in rows if r.get("correct_30d") is True)
            d1n = sum(1 for r in rows if r.get("correct_1d") is not None)
            d7n = sum(1 for r in rows if r.get("correct_7d") is not None)
            d30n = sum(1 for r in rows if r.get("correct_30d") is not None)

            buys = [r for r in rows if r.get("signal") == "BUY"]
            sells = [r for r in rows if r.get("signal") == "SELL"]

            print(f"\n=== Accuracy for {symbol} ({total} runs) ===")
            print(f"  D+1  DA: {d1}/{d1n} = {d1/d1n*100:.1f}%" if d1n else "  D+1  DA: N/A")
            print(f"  D+7  DA: {d7}/{d7n} = {d7/d7n*100:.1f}%" if d7n else "  D+7  DA: N/A")
            print(f"  D+30 DA: {d30}/{d30n} = {d30/d30n*100:.1f}%" if d30n else "  D+30 DA: N/A")
            print(f"  Signal dist: BUY={len(buys)} SELL={len(sells)} HOLD={total-len(buys)-len(sells)}")
            print(f"  Avg confidence: {sum(r.get('confidence',0) for r in rows)/total:.3f}")
            print(f"  Avg sentiment: {sum(r.get('sentiment_score',0) for r in rows)/total:.3f}")
            return

        # ── Discover available dates ──
        print(f"Discovering available dates for {symbol}...")
        available = await get_available_dates(http, symbol, start, end)
        print(f"  {len(available)} dates with factor snapshots")

        if not available:
            print("No dates found.")
            return

        # Filter: only dates where D+30 data is available
        today = date.today()
        available = [d for d in available if d + timedelta(days=30) <= today]
        print(f"  {len(available)} dates with D+30 data available")

        # ── Find pending dates ──
        completed = await get_completed_dates(http, symbol)
        print(f"  {len(completed)} already completed")

        pending = sorted(set(d.isoformat() for d in available) - completed, reverse=args.reverse)
        order_label = "newest-first" if args.reverse else "oldest-first"
        print(f"  {len(pending)} pending ({order_label})")

        if not pending:
            print("\nAll dates completed!")
            return

        if args.dry_run:
            print(f"\nWould run {len(pending)} dates (dry-run):")
            for d in pending[:10]:
                print(f"  {d}")
            if len(pending) > 10:
                print(f"  ... and {len(pending) - 10} more")
            return

        # ── Run backtests ──
        limit = args.max if args.max > 0 else len(pending)
        to_run = pending[:limit]
        print(f"\nRunning {len(to_run)} backtests...\n")

        stats = {"total": 0, "warmup": 0, "eval": 0,
                 "correct_1d": 0, "correct_7d": 0, "correct_30d": 0,
                 "eval_1d": 0, "eval_7d": 0, "eval_30d": 0, "errors": 0}

        warmup = args.warmup

        for i, d_str in enumerate(to_run):
            td = date.fromisoformat(d_str)
            in_warmup = stats["total"] < warmup
            tag = "[WARMUP]" if in_warmup else f"[{stats['eval']+1}/{len(to_run)-warmup}]"
            print(f"{tag} {td} ...", end=" ", flush=True)

            try:
                result = await run_backtest(http, symbol, td)
            except Exception as exc:
                print(f"FAIL: {exc}")
                stats["errors"] += 1
                continue

            if result is None:
                print("FAIL (no result)")
                stats["errors"] += 1
                continue

            # Compute actual returns
            actual = await fetch_actual_returns(http, symbol, td)

            signal = result["signal"]

            # Build row for Supabase
            row: dict = {
                "symbol": symbol,
                "backtest_date": d_str,
                "run_id": result["run_id"],
                "signal": signal,
                "confidence": result["confidence"],
                "sentiment_score": result["sentiment_score"],
                "explanation": result["explanation"],
                "errors": json.dumps(result["errors"]),
                "n_similar_cases": result["n_similar_cases"],
                "top_similar_date": result["top_similar_date"],
                "top_similar_similarity": result["top_similar_similarity"],
                "top_similar_ret7d": result["top_similar_ret7d"],
                "run_duration_ms": result["run_duration_ms"],
            }

            for h_label in ["1d", "7d", "30d"]:
                val = actual.get(h_label)
                row[f"actual_return_{h_label}"] = val
                correct = check_correct(signal, val) if val is not None else None
                row[f"correct_{h_label}"] = correct
                if not in_warmup and correct is not None:
                    stats[f"eval_{h_label}"] += 1
                    if correct:
                        stats[f"correct_{h_label}"] += 1

            # Write to Supabase
            ok = await supabase_upsert(http, "backtest_results", row)
            if not ok:
                stats["errors"] += 1
                print("WRITE FAIL")
                continue

            stats["total"] += 1
            if in_warmup:
                stats["warmup"] += 1
            else:
                stats["eval"] += 1

            # Print one-line summary
            r7 = actual.get("7d")
            c7 = row.get("correct_7d")
            c7_str = "OK" if c7 else ("XX" if c7 is False else ("??" if in_warmup else "?"))
            warmup_mark = " [w]" if in_warmup else ""
            r7_str = f"{r7:+.2f}%" if isinstance(r7, (int, float)) else "N/A"
            print(f"{signal}({result['confidence']:.2f}) ret7d={r7_str}{warmup_mark} "
                  f"dur={result['run_duration_ms']}ms")

            # Throttle between runs to avoid rate limiting
            await asyncio.sleep(1)

        # ── Final stats ──
        print(f"\n{'=' * 50}")
        print(f"Done. {stats['total']} runs ({stats['warmup']} warmup + {stats['eval']} eval), {stats['errors']} errors")
        if stats["eval_7d"] > 0:
            print(f"  D+1  DA : {stats['correct_1d']}/{stats['eval_1d']} = {stats['correct_1d']/stats['eval_1d']*100:.1f}%")
            print(f"  D+7  DA : {stats['correct_7d']}/{stats['eval_7d']} = {stats['correct_7d']/stats['eval_7d']*100:.1f}%")
            print(f"  D+30 DA : {stats['correct_30d']}/{stats['eval_30d']} = {stats['correct_30d']/stats['eval_30d']*100:.1f}%")
        else:
            print("  No eval data (all warmup). Run with more --max.")

        # Print stats command for later
        print(f"\n  Check stats later: python scripts/backtest_runner.py --symbol {symbol} --stats")


if __name__ == "__main__":
    asyncio.run(main())
