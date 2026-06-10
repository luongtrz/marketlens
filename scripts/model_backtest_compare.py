"""Run fixed-window backtest directly via /run for model comparison.

This script does not write to Supabase backtest_results.
It is intended for apples-to-apples model evaluation on the same dates.

Usage:
    python scripts/model_backtest_compare.py --start 2022-01-01 --end 2026-05-17
    python scripts/model_backtest_compare.py --start 2022-01-01 --end 2026-05-17 --model kimi-k2.5
    python scripts/model_backtest_compare.py --start 2024-01-01 --end 2024-12-31 --concurrency 4
    python scripts/model_backtest_compare.py --start 2022-01-01 --end 2026-05-17 --model kimi-k2.5 --output results/kimi_full.json

Compare saved runs:
    python scripts/model_backtest_compare.py --compare results/kimi_full.json results/qwen_full.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx


MAIN_URL = os.getenv("MAIN_URL", "http://localhost:8005")
MARKET_URL = os.getenv("MARKET_URL", "http://localhost:8002")
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
    for _ in range(60):
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


class Stats:
    """Per-signal accuracy tracker."""

    def __init__(self) -> None:
        self.by_signal: dict[str, dict[str, list]] = {
            s: {h: [0, 0, 0.0] for h in ("1d", "7d", "30d")}
            for s in ("BUY", "SELL", "HOLD")
        }
        self.by_year: dict[int, dict[str, dict[str, list]]] = {}
        self.total = 0
        self.errors = 0
        self.dur_ms = 0
        self.rows: list[dict] = []  # per-date results for --output

    def record(self, td: date, signal: str, confidence: float, actual: dict[str, float | None], dur_ms: int) -> None:
        self.total += 1
        self.dur_ms += dur_ms
        self.rows.append({
            "date": td.isoformat(),
            "signal": signal,
            "confidence": round(confidence, 4),
            "ret_1d": actual.get("1d"),
            "ret_7d": actual.get("7d"),
            "ret_30d": actual.get("30d"),
        })
        yr = td.year
        if yr not in self.by_year:
            self.by_year[yr] = {
                s: {h: [0, 0, 0.0] for h in ("1d", "7d", "30d")}
                for s in ("BUY", "SELL", "HOLD")
            }
        for h, v in actual.items():
            if v is None:
                continue
            ok = check_correct(signal, v)
            if ok is None:
                continue
            self.by_signal[signal][h][0] += int(ok)
            self.by_signal[signal][h][1] += 1
            self.by_signal[signal][h][2] += v
            self.by_year[yr][signal][h][0] += int(ok)
            self.by_year[yr][signal][h][1] += 1

    def to_dict(self, model_label: str, symbol: str, start: str, end: str) -> dict:
        counts = {s: self.by_signal[s]["7d"][1] for s in ("BUY", "SELL", "HOLD")}
        total_sig = sum(counts.values())
        buy_c, buy_n, buy_ret = self.by_signal["BUY"]["7d"]
        sell_c, sell_n, sell_ret = self.by_signal["SELL"]["7d"]
        hold_c, hold_n, _ = self.by_signal["HOLD"]["7d"]
        prec_c = buy_c + sell_c
        prec_n = buy_n + sell_n
        by_year_out = {}
        for yr, yd in self.by_year.items():
            bc, bn = yd["BUY"]["7d"][0], yd["BUY"]["7d"][1]
            sc, sn = yd["SELL"]["7d"][0], yd["SELL"]["7d"][1]
            hc, hn = yd["HOLD"]["7d"][0], yd["HOLD"]["7d"][1]
            yr_total = bn + sn + hn
            pc, pn = bc + sc, bn + sn
            by_year_out[str(yr)] = {
                "buy": [bc, bn], "sell": [sc, sn], "hold": [hc, hn],
                "precision": round(pc / pn * 100, 2) if pn else None,
                "coverage_pct": round(pn / yr_total * 100, 2) if yr_total else None,
            }
        return {
            "meta": {
                "model": model_label,
                "symbol": symbol,
                "start": start,
                "end": end,
                "run_at": datetime.now(timezone.utc).isoformat(),
                "total_runs": self.total,
                "errors": self.errors,
                "avg_dur_ms": round(self.dur_ms / self.total) if self.total else 0,
            },
            "summary": {
                "signal_dist": {s: counts[s] for s in ("BUY", "SELL", "HOLD")},
                "total_with_7d": total_sig,
                "coverage_pct": round(prec_n / total_sig * 100, 2) if total_sig else 0,
                "buy_acc": round(buy_c / buy_n * 100, 2) if buy_n else None,
                "sell_acc": round(sell_c / sell_n * 100, 2) if sell_n else None,
                "hold_acc": round(hold_c / hold_n * 100, 2) if hold_n else None,
                "buy_avg_ret": round(buy_ret / buy_n, 4) if buy_n else None,
                "sell_avg_ret": round(sell_ret / sell_n, 4) if sell_n else None,
                "precision": round(prec_c / prec_n * 100, 2) if prec_n else None,
            },
            "by_year": by_year_out,
            "rows": self.rows,
        }

    def save(self, path: str, model_label: str, symbol: str, start: str, end: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict(model_label, symbol, start, end)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved {len(self.rows)} rows → {out_path}")

    def print_summary(self, model_label: str) -> None:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_label}")
        print(f"{'='*60}")

        counts = {s: self.by_signal[s]["7d"][1] for s in ("BUY", "SELL", "HOLD")}
        total_sig = sum(counts.values())
        print(f"\nSignal distribution ({total_sig} total):")
        for s in ("BUY", "SELL", "HOLD"):
            n = counts[s]
            pct = n / total_sig * 100 if total_sig else 0
            print(f"  {s:4s}: {n:4d} ({pct:.1f}%)")
        coverage = (counts["BUY"] + counts["SELL"]) / total_sig * 100 if total_sig else 0
        print(f"  Coverage (BUY+SELL): {coverage:.1f}%")

        print(f"\nAccuracy at D+7:")
        buy_c, buy_n, buy_ret = self.by_signal["BUY"]["7d"]
        sell_c, sell_n, sell_ret = self.by_signal["SELL"]["7d"]
        hold_c, hold_n, _ = self.by_signal["HOLD"]["7d"]
        if buy_n:
            print(f"  BUY:  {buy_c}/{buy_n} = {buy_c/buy_n*100:.1f}%  avg_ret={buy_ret/buy_n:+.2f}%")
        if sell_n:
            print(f"  SELL: {sell_c}/{sell_n} = {sell_c/sell_n*100:.1f}%  avg_ret={sell_ret/sell_n:+.2f}%")
        if hold_n:
            print(f"  HOLD: {hold_c}/{hold_n} = {hold_c/hold_n*100:.1f}%")

        prec_c = buy_c + sell_c
        prec_n = buy_n + sell_n
        if prec_n:
            print(f"\n  Precision (BUY+SELL): {prec_c}/{prec_n} = {prec_c/prec_n*100:.1f}%")

        print(f"\nAccuracy overview (all signals):")
        for h in ("1d", "7d", "30d"):
            tc = sum(self.by_signal[s][h][0] for s in ("BUY", "SELL", "HOLD"))
            tn = sum(self.by_signal[s][h][1] for s in ("BUY", "SELL", "HOLD"))
            if tn:
                print(f"  D+{h.replace('d',''):>2s}: {tc}/{tn} = {tc/tn*100:.1f}%")

        print(f"\nPer-year precision (D+7, BUY+SELL only):")
        print(f"  {'Year':>4s}  {'Prec':>6s}  {'BUY':>12s}  {'SELL':>12s}  {'Coverage':>8s}")
        for yr in sorted(self.by_year.keys()):
            yd = self.by_year[yr]
            bc, bn = yd["BUY"]["7d"][0], yd["BUY"]["7d"][1]
            sc, sn = yd["SELL"]["7d"][0], yd["SELL"]["7d"][1]
            hc, hn = yd["HOLD"]["7d"][0], yd["HOLD"]["7d"][1]
            yr_total = bn + sn + hn
            pc, pn = bc + sc, bn + sn
            prec_str = f"{pc/pn*100:.1f}%" if pn else "N/A"
            buy_str = f"{bc}/{bn}={bc/bn*100:.0f}%" if bn else "—"
            sell_str = f"{sc}/{sn}={sc/sn*100:.0f}%" if sn else "—"
            cov_str = f"{pn/yr_total*100:.1f}%" if yr_total else "N/A"
            print(f"  {yr}  {prec_str:>6s}  {buy_str:>12s}  {sell_str:>12s}  {cov_str:>8s}")

        avg_dur = self.dur_ms / self.total if self.total else 0
        print(f"\nRuns: {self.total}  Errors: {self.errors}  AvgDur: {avg_dur:.0f}ms")


def cmd_compare(paths: list[str]) -> None:
    """Load multiple saved JSON results and print a comparison table."""
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))

    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON ({len(runs)} runs)")
    print(f"{'='*80}")
    hdr = f"  {'Model':<28} {'Period':<23} {'Prec':>6} {'BUY':>6} {'SELL':>6} {'Cov':>6} {'n':>5}"
    print(hdr)
    print(f"  {'-'*76}")
    for r in runs:
        m = r["meta"]
        s = r["summary"]
        period = f"{m['start']} → {m['end']}"
        prec = f"{s['precision']:.1f}%" if s["precision"] is not None else "N/A"
        buy = f"{s['buy_acc']:.1f}%" if s["buy_acc"] is not None else "N/A"
        sell = f"{s['sell_acc']:.1f}%" if s["sell_acc"] is not None else "N/A"
        cov = f"{s['coverage_pct']:.1f}%"
        n = s["total_with_7d"]
        print(f"  {m['model']:<28} {period:<23} {prec:>6} {buy:>6} {sell:>6} {cov:>6} {n:>5}")

    print(f"\nPer-year precision breakdown:")
    all_years = sorted({yr for r in runs for yr in r["by_year"].keys()})
    print(f"  {'Model':<28} " + "  ".join(f"{y}" for y in all_years))
    print(f"  {'-'*70}")
    for r in runs:
        row = f"  {r['meta']['model']:<28} "
        parts = []
        for yr in all_years:
            yd = r["by_year"].get(yr, {})
            prec = yd.get("precision")
            parts.append(f"{prec:>5.1f}%" if prec is not None else "  N/A ")
        print(row + "  ".join(parts))


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--model", default=None, help="LLM model override, e.g. kimi-k2.5")
    p.add_argument("--concurrency", type=int, default=1, help="Parallel workers (default 1)")
    p.add_argument("--output", default=None, help="Save results to JSON file, e.g. results/kimi_full.json")
    p.add_argument("--compare", nargs="+", metavar="FILE", help="Compare saved JSON result files")
    args = p.parse_args()

    if args.compare:
        cmd_compare(args.compare)
        return

    if not args.start or not args.end:
        p.error("--start and --end are required (unless using --compare)")

    dates = build_dates(date.fromisoformat(args.start), date.fromisoformat(args.end), args.reverse)
    symbol = args.symbol.upper()
    model_label = args.model or "default(kimi-k2.5)"
    print(f"Running backtest for {symbol}: {len(dates)} dates | model={model_label}")

    stats = Stats()
    lock = asyncio.Lock()
    queue: asyncio.Queue[tuple[int, date]] = asyncio.Queue()
    for i, d in enumerate(dates, 1):
        await queue.put((i, d))

    async def worker() -> None:
        while True:
            try:
                i, d = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                async with httpx.AsyncClient(timeout=180) as http:
                    res = await run_one(http, symbol, d, model=args.model)
                    actual = await fetch_actual_returns(http, symbol, d)
            except Exception as exc:
                async with lock:
                    stats.errors += 1
                    print(f"[{i}/{len(dates)}] {d} ERR({type(exc).__name__})", flush=True)
                continue
            async with lock:
                if res is None:
                    stats.errors += 1
                    print(f"[{i}/{len(dates)}] {d} FAIL", flush=True)
                    continue
                s = str(res["signal"])
                stats.record(d, s, res["confidence"], actual, res["duration_ms"])
                r7 = actual.get("7d")
                ok7 = check_correct(s, r7) if r7 is not None else None
                r7s = f"{r7:+.2f}%" if isinstance(r7, (int, float)) else "N/A"
                r1 = actual.get("1d")
                r1s = f"{r1:+.2f}%" if isinstance(r1, (int, float)) else "N/A"
                ok_mark = "✓" if ok7 else ("✗" if ok7 is False else "?")
                print(f"[{i}/{len(dates)}] {d} {s}({res['confidence']:.2f}) ret1d={r1s} ret7d={r7s} {ok_mark}", flush=True)

    await asyncio.gather(*[worker() for _ in range(args.concurrency)])
    stats.print_summary(model_label)

    if args.output:
        stats.save(args.output, model_label, args.start, args.end)


if __name__ == "__main__":
    asyncio.run(main())
