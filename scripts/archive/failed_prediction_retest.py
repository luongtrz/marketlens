"""Re-run only wrong predictions from Supabase with a different model.

Instead of re-running all 1,569 dates, this script targets only the ~359 cases
where the current model predicted wrong (BUY wrong or SELL wrong).
This answers: "Would a better model fix these?"

Usage:
    # Compare deepseek-v4-pro on cases where deepseek-v4-flash was wrong
    python scripts/failed_prediction_retest.py --model deepseek-v4-pro

    # Test only SELL failures
    python scripts/failed_prediction_retest.py --model deepseek-v4-pro --signal SELL

    # Dry run: just list the failing dates without re-running
    python scripts/failed_prediction_retest.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
            k, v = k.strip(), v.strip().strip("\"'")
            if k:
                os.environ[k] = v

import httpx

MAIN_URL = os.getenv("MAIN_URL", "http://localhost:8005")
MARKET_URL = os.getenv("MARKET_URL", "http://localhost:8002")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""


def sb_headers() -> dict[str, str]:
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}


async def fetch_wrong_predictions(signal_filter: str | None = None) -> list[dict]:
    """Fetch rows from Supabase where model was wrong (BUY wrong or SELL wrong)."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL / SUPABASE_KEY not set")
        sys.exit(1)

    url = f"{SUPABASE_URL}/rest/v1/backtest_results"
    params: dict[str, str] = {
        "select": "backtest_date,signal,actual_return_7d,confidence",
        "order": "backtest_date.asc",
        "limit": "5000",
    }

    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.get(url, params=params, headers=sb_headers())
    if r.status_code != 200:
        print(f"Supabase error {r.status_code}: {r.text[:200]}")
        sys.exit(1)

    rows = r.json()
    wrong: list[dict] = []
    for row in rows:
        sig = row.get("signal", "")
        ret7 = row.get("actual_return_7d")
        if ret7 is None:
            continue
        # normalize key to "date" for downstream usage
        row["date"] = row["backtest_date"]
        row["future_return_7d"] = ret7
        if sig == "BUY" and ret7 < 0:
            if signal_filter in (None, "BUY"):
                wrong.append(row)
        elif sig == "SELL" and ret7 > 0:
            if signal_filter in (None, "SELL"):
                wrong.append(row)
    return wrong


async def run_one(http: httpx.AsyncClient, symbol: str, td: date, model: str) -> dict | None:
    t0 = time.time()
    r = await http.post(f"{MAIN_URL}/run", params={"symbol": symbol, "date": td.isoformat(), "model": model})
    if r.status_code != 200:
        return None
    run_id = r.json()["run_id"]
    for _ in range(60):
        await asyncio.sleep(2)
        sr = await http.get(f"{MAIN_URL}/status/{run_id}")
        if sr.status_code != 200:
            continue
        if sr.json().get("status") in {"done", "failed"}:
            break
    rr = await http.get(f"{MAIN_URL}/result/{run_id}")
    if rr.status_code != 200:
        return None
    j = rr.json()
    return {
        "signal": j.get("signal", "HOLD"),
        "confidence": float(j.get("confidence", 0.0) or 0.0),
        "duration_ms": int((time.time() - t0) * 1000),
    }


def check_correct(signal: str, ret7: float) -> bool:
    if signal == "BUY":
        return ret7 > 0
    if signal == "SELL":
        return ret7 < 0
    return abs(ret7) < 1.0


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--model", required=False, default="deepseek-v4-pro",
                   help="Model to re-test with (e.g. deepseek-v4-pro, kimi-k2.5)")
    p.add_argument("--signal", choices=["BUY", "SELL"], default=None,
                   help="Filter to only BUY or SELL failures")
    p.add_argument("--dry-run", action="store_true", help="Print failing dates only, don't re-run")
    p.add_argument("--concurrency", type=int, default=1)
    args = p.parse_args()

    print("Fetching wrong predictions from Supabase...")
    wrong_rows = await fetch_wrong_predictions(signal_filter=args.signal)

    # Breakdown
    buy_wrong = [r for r in wrong_rows if r["signal"] == "BUY"]
    sell_wrong = [r for r in wrong_rows if r["signal"] == "SELL"]
    print(f"\nWrong predictions: {len(wrong_rows)} total")
    print(f"  BUY wrong:  {len(buy_wrong)}")
    print(f"  SELL wrong: {len(sell_wrong)}")

    if args.dry_run:
        print("\nDates (dry-run mode):")
        for row in wrong_rows:
            print(f"  {row['date']} {row['signal']} ret7d={row['future_return_7d']:+.2f}% conf={row['confidence']:.3f}")
        return

    if not wrong_rows:
        print("No wrong predictions found.")
        return

    print(f"\nRe-running {len(wrong_rows)} dates with model={args.model}...")

    # Track results
    results: dict[str, dict] = {}  # date -> {old_signal, new_signal, ret7d, old_correct, new_correct}
    for row in wrong_rows:
        results[row["date"]] = {
            "old_signal": row["signal"],
            "ret7d": row["future_return_7d"],
            "old_correct": False,  # by definition (these are wrong predictions)
        }

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = [0]

    async def process(row: dict) -> None:
        td = date.fromisoformat(row["date"])
        async with sem:
            async with httpx.AsyncClient(timeout=180) as http:
                res = await run_one(http, args.symbol, td, model=args.model)

        async with lock:
            done[0] += 1
            idx = done[0]
            if res is None:
                print(f"[{idx}/{len(wrong_rows)}] {row['date']} FAIL")
                return
            new_sig = res["signal"]
            ret7 = row["future_return_7d"]
            new_correct = check_correct(new_sig, ret7)
            results[row["date"]]["new_signal"] = new_sig
            results[row["date"]]["new_correct"] = new_correct
            old_sig = row["signal"]
            outcome = "✓ FIXED" if new_correct else ("→ changed wrong" if new_sig != old_sig else "✗ same")
            print(
                f"[{idx}/{len(wrong_rows)}] {row['date']} "
                f"{old_sig}→{new_sig}({res['confidence']:.2f}) "
                f"ret7d={ret7:+.2f}% {outcome}"
            )

    await asyncio.gather(*[process(row) for row in wrong_rows])

    # Summary
    completed = [r for r in results.values() if "new_signal" in r]
    fixed = [r for r in completed if r.get("new_correct")]
    changed_signal = [r for r in completed if r.get("new_signal") != r["old_signal"]]
    still_wrong = [r for r in completed if not r.get("new_correct") and r.get("new_signal") == r["old_signal"]]

    # Per original-signal breakdown
    buy_completed = [r for r in completed if r["old_signal"] == "BUY"]
    sell_completed = [r for r in completed if r["old_signal"] == "SELL"]
    buy_fixed = [r for r in buy_completed if r.get("new_correct")]
    sell_fixed = [r for r in sell_completed if r.get("new_correct")]

    print(f"\n{'='*55}")
    print(f"  RETEST SUMMARY: {args.model}")
    print(f"{'='*55}")
    print(f"Tested: {len(completed)}/{len(wrong_rows)}")
    print(f"\nBy original signal:")
    if buy_completed:
        print(f"  BUY  wrong → fixed: {len(buy_fixed)}/{len(buy_completed)} ({len(buy_fixed)/len(buy_completed)*100:.1f}%)")
    if sell_completed:
        print(f"  SELL wrong → fixed: {len(sell_fixed)}/{len(sell_completed)} ({len(sell_fixed)/len(sell_completed)*100:.1f}%)")

    print(f"\nOverall:")
    print(f"  Fixed (now correct): {len(fixed)}/{len(completed)} ({len(fixed)/len(completed)*100:.1f}%)")
    print(f"  Signal changed:      {len(changed_signal)}/{len(completed)}")
    print(f"  Still wrong (same):  {len(still_wrong)}/{len(completed)}")

    # New signal distribution for changed cases
    if changed_signal:
        print(f"\nHow signal changed for wrong→different:")
        from collections import Counter
        transitions = Counter(f"{r['old_signal']}→{r.get('new_signal','?')}" for r in changed_signal)
        for k, v in transitions.most_common():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
