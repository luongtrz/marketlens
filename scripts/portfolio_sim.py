"""Portfolio simulation from backtest log or JSON output.

Simulates $1M starting capital with confidence-weighted position sizing.
SELL = short BTC (profit when price drops).
Overlapping positions allowed (no cap on concurrent trades).

Strategies supported:
  --stop-loss 0.03       Hard stop-loss (3%)
  --take-profit 0.05     Take profit at 5%
  --trailing-stop 0.03   Trailing stop 3% from peak
  --dynamic-exit         Close when opposite signal appears
  --log2 FILE            Ensemble: only trade when both logs agree

Usage:
    python scripts/portfolio_sim.py --log /tmp/bt_kimi_full.log --start 2025-01-01 --end 2026-05-17
    python scripts/portfolio_sim.py --log /tmp/bt_qwen_combined.log \\
        --start 2025-01-01 --end 2026-05-17 \\
        --stop-loss 0.03 --take-profit 0.06 --trailing-stop 0.03 --dynamic-exit
    python scripts/portfolio_sim.py \\
        --log /tmp/bt_kimi_full.log --log2 /tmp/bt_qwen_combined.log \\
        --start 2025-01-01 --end 2026-05-17 --stop-loss 0.03
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

MAIN_URL = os.getenv("MAIN_URL", "http://localhost:8005")
MARKET_URL = os.getenv("MARKET_URL", "http://localhost:8002")
HORIZONS = {"1d": 1, "7d": 7, "30d": 30}


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Row:
    date: date
    signal: str
    confidence: float
    ret_1d: float | None
    ret_7d: float | None
    ret_30d: float | None


@dataclass
class Position:
    open_date: date
    close_date: date       # max hold date (7 days)
    direction: str         # BUY (long) or SELL (short)
    size_usd: float
    confidence: float
    cumret: float = 0.0    # current accumulated return (from position's perspective)
    peak_pnl_pct: float = 0.0  # highest cumret seen (for trailing stop)
    actual_ret: float | None = None   # final return used for P&L
    pnl: float | None = None
    closed: bool = False
    exit_reason: str = "7d"  # "7d", "SL", "TP", "trail", "signal"
    ret_7d: float | None = None  # stored for fallback at 7d expiry

    def update(self, day_ret_1d: float) -> None:
        """Apply one day of market return to the position."""
        factor = day_ret_1d / 100
        if self.direction == "SELL":
            factor = -factor
        self.cumret = (1 + self.cumret) * (1 + factor) - 1
        if self.cumret > self.peak_pnl_pct:
            self.peak_pnl_pct = self.cumret

    def should_exit(
        self,
        stop_loss: float | None,
        take_profit: float | None,
        trailing_stop: float | None,
    ) -> str | None:
        """Return exit reason if any threshold triggered, else None."""
        if stop_loss is not None and self.cumret <= -stop_loss:
            return "SL"
        if take_profit is not None and self.cumret >= take_profit:
            return "TP"
        if trailing_stop is not None and self.peak_pnl_pct >= trailing_stop:
            if self.cumret <= self.peak_pnl_pct - trailing_stop:
                return "trail"
        return None

    def close_now(self, reason: str) -> float:
        self.exit_reason = reason
        self.actual_ret = self.cumret
        self.pnl = self.size_usd * self.cumret
        self.closed = True
        return self.pnl


# ── Parsers ──────────────────────────────────────────────────────────────────

LOG_RE = re.compile(
    r"\[\d+/\d+\]\s+(\d{4}-\d{2}-\d{2})\s+(BUY|SELL|HOLD)\((\d+\.\d+)\)"
    r"\s+ret7d=([+-]?\d+\.\d+%|N/A)"
)
# Also try to capture ret1d if present in a different format (JSON rows have it)
LOG_RE_1D = re.compile(r"ret1d=([+-]?\d+\.\d+%|N/A)")


def parse_log(path: str, start: date | None, end: date | None) -> list[Row]:
    rows: list[Row] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            d = date.fromisoformat(m.group(1))
            if start and d < start:
                continue
            if end and d > end:
                continue
            if d.isoformat() in seen:
                continue
            seen.add(d.isoformat())
            ret7 = None if m.group(4) == "N/A" else float(m.group(4).rstrip("%"))
            m1 = LOG_RE_1D.search(line)
            ret1 = None if (not m1 or m1.group(1) == "N/A") else float(m1.group(1).rstrip("%"))
            rows.append(Row(
                date=d, signal=m.group(2), confidence=float(m.group(3)),
                ret_1d=ret1, ret_7d=ret7, ret_30d=None,
            ))
    rows.sort(key=lambda r: r.date)
    return rows


def parse_json(path: str, start: date | None, end: date | None) -> list[Row]:
    with open(path) as f:
        data = json.load(f)
    rows: list[Row] = []
    for r in data.get("rows", []):
        d = date.fromisoformat(r["date"])
        if start and d < start:
            continue
        if end and d > end:
            continue
        rows.append(Row(
            date=d, signal=r["signal"], confidence=r["confidence"],
            ret_1d=r.get("ret_1d"), ret_7d=r.get("ret_7d"), ret_30d=r.get("ret_30d"),
        ))
    rows.sort(key=lambda r: r.date)
    return rows


def ensemble_merge(rows_a: list[Row], rows_b: list[Row]) -> list[Row]:
    """Keep only dates where both logs agree on BUY or SELL direction."""
    b_map = {r.date: r for r in rows_b}
    merged: list[Row] = []
    for r in rows_a:
        rb = b_map.get(r.date)
        if rb is None:
            merged.append(Row(date=r.date, signal="HOLD", confidence=r.confidence,
                              ret_1d=r.ret_1d, ret_7d=r.ret_7d, ret_30d=r.ret_30d))
            continue
        if r.signal == rb.signal and r.signal in ("BUY", "SELL"):
            avg_conf = (r.confidence + rb.confidence) / 2
            merged.append(Row(date=r.date, signal=r.signal, confidence=round(avg_conf, 3),
                              ret_1d=r.ret_1d, ret_7d=r.ret_7d, ret_30d=r.ret_30d))
        else:
            merged.append(Row(date=r.date, signal="HOLD", confidence=r.confidence,
                              ret_1d=r.ret_1d, ret_7d=r.ret_7d, ret_30d=r.ret_30d))
    return sorted(merged, key=lambda r: r.date)


# ── Random baseline ───────────────────────────────────────────────────────────

def randomize_rows(rows: list[Row], seed: int) -> list[Row]:
    import random
    rng = random.Random(seed)
    return [Row(date=r.date, signal=rng.choice(["BUY", "SELL", "HOLD"]),
                confidence=round(rng.uniform(0.55, 0.74), 2),
                ret_1d=r.ret_1d, ret_7d=r.ret_7d, ret_30d=r.ret_30d)
            for r in rows]


# ── Live fetch ────────────────────────────────────────────────────────────────

async def fetch_actual_returns(http: httpx.AsyncClient, symbol: str, td: date) -> dict[str, float | None]:
    out: dict[str, float | None] = {"1d": None, "7d": None, "30d": None}
    end_ts = int((datetime(td.year, td.month, td.day, tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000)
    r = await http.get(f"{MARKET_URL}/history",
                       params={"symbol": symbol, "interval": "1d", "limit": 5, "end_time": str(end_ts)})
    if r.status_code != 200:
        return out
    candles = r.json()
    match = next((c for c in reversed(candles) if c["timestamp"][:10] == td.isoformat()), None)
    if not match:
        return out
    base = match["close"]
    for label, off in HORIZONS.items():
        target = td + timedelta(days=off)
        if target > date.today() - timedelta(days=1):
            continue
        t_end = int((datetime(target.year, target.month, target.day, tzinfo=timezone.utc) + timedelta(days=2)).timestamp() * 1000)
        r2 = await http.get(f"{MARKET_URL}/history",
                            params={"symbol": symbol, "interval": "1d", "limit": 5, "end_time": str(t_end)})
        if r2.status_code != 200:
            continue
        for c2 in reversed(r2.json()):
            if c2["timestamp"][:10] <= target.isoformat():
                out[label] = round((c2["close"] - base) / base * 100, 4)
                break
    return out


async def run_one(http: httpx.AsyncClient, symbol: str, td: date, model: str | None) -> dict | None:
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
        if sr.json().get("status") in {"done", "failed"}:
            break
    rr = await http.get(f"{MAIN_URL}/result/{run_id}")
    if rr.status_code != 200:
        return None
    j = rr.json()
    return {"signal": j.get("signal", "HOLD"), "confidence": float(j.get("confidence", 0.0) or 0.0),
            "duration_ms": int((time.time() - t0) * 1000)}


async def fetch_live(symbol: str, start: date, end: date, model: str | None, concurrency: int) -> list[Row]:
    dates: list[date] = []
    cur = start
    while cur <= end:
        dates.append(cur)
        cur += timedelta(days=1)
    rows: list[Row] = []
    lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    for i, d in enumerate(dates, 1):
        await queue.put((i, d))

    async def worker():
        while True:
            try:
                i, d = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                async with httpx.AsyncClient(timeout=180) as http:
                    res = await run_one(http, symbol, d, model)
                    actual = await fetch_actual_returns(http, symbol, d)
            except Exception as exc:
                print(f"[{i}/{len(dates)}] {d} ERR({type(exc).__name__})", flush=True)
                continue
            async with lock:
                if res is None:
                    print(f"[{i}/{len(dates)}] {d} FAIL", flush=True)
                    continue
                sig, conf = res["signal"], res["confidence"]
                r7 = actual.get("7d")
                rows.append(Row(date=d, signal=sig, confidence=conf,
                                ret_1d=actual.get("1d"), ret_7d=r7, ret_30d=actual.get("30d")))
                print(f"[{i}/{len(dates)}] {d} {sig}({conf:.2f}) ret7d={f'{r7:+.2f}%' if r7 else 'N/A'}", flush=True)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    rows.sort(key=lambda r: r.date)
    return rows


# ── Simulator ─────────────────────────────────────────────────────────────────

def simulate(
    rows: list[Row],
    initial_capital: float,
    size_pct_per_conf: float,
    no_short: bool = False,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    dynamic_exit: bool = False,
    fixed_size: float | None = None,
    silent: bool = False,
) -> dict:
    # Build date lookup for daily return tracking and dynamic exit
    date_map: dict[date, Row] = {r.date: r for r in rows}

    capital = initial_capital
    peak = capital
    max_drawdown = 0.0
    positions: list[Position] = []
    closed_positions: list[Position] = []
    trade_log: list[str] = []
    monthly: dict[str, float] = {}

    use_daily_tracking = any(x is not None for x in [take_profit, trailing_stop]) or dynamic_exit

    for row in rows:
        # ── 1. Daily update + early exit checks ──────────────────────────────
        still_open: list[Position] = []
        for pos in positions:
            if pos.closed:
                continue

            # Apply today's market move to track cumulative return
            if use_daily_tracking and row.ret_1d is not None:
                pos.update(row.ret_1d)

            # Dynamic exit: close if today's signal is opposite
            if dynamic_exit and not pos.closed:
                opposite = "SELL" if pos.direction == "BUY" else "BUY"
                if row.signal == opposite:
                    pnl = pos.close_now("signal")
                    capital += pnl
                    closed_positions.append(pos)
                    monthly[pos.open_date.strftime("%Y-%m")] = monthly.get(pos.open_date.strftime("%Y-%m"), 0) + pnl
                    tag = f"[{pos.exit_reason.upper()}]"
                    trade_log.append(f"  CLOSE {'LONG' if pos.direction=='BUY' else 'SHORT'} "
                                     f"{pos.open_date}→{row.date} ${pos.size_usd:,.0f} "
                                     f"ret={pos.actual_ret*100:+.2f}% pnl=${pnl:+,.0f} {tag}")
                    continue

            # TP / SL / Trailing checks (only when we have daily tracking)
            if use_daily_tracking and row.ret_1d is not None and not pos.closed:
                reason = pos.should_exit(stop_loss, take_profit, trailing_stop)
                if reason:
                    # Apply threshold directly as actual return
                    if reason == "SL":
                        pos.cumret = -stop_loss  # type: ignore
                    elif reason == "TP":
                        pos.cumret = take_profit  # type: ignore
                    elif reason == "trail":
                        pos.cumret = pos.peak_pnl_pct - trailing_stop  # type: ignore
                    pnl = pos.close_now(reason)
                    capital += pnl
                    closed_positions.append(pos)
                    monthly[pos.open_date.strftime("%Y-%m")] = monthly.get(pos.open_date.strftime("%Y-%m"), 0) + pnl
                    trade_log.append(f"  CLOSE {'LONG' if pos.direction=='BUY' else 'SHORT'} "
                                     f"{pos.open_date}→{row.date} ${pos.size_usd:,.0f} "
                                     f"ret={pos.actual_ret*100:+.2f}% pnl=${pnl:+,.0f} [{reason.upper()}]")
                    continue

            # ── 2. Close at 7-day expiry ──────────────────────────────────────
            if pos.close_date <= row.date:
                # Use cumret if we've been tracking daily, else fall back to ret_7d
                if use_daily_tracking and pos.actual_ret is None:
                    if pos.ret_7d is not None:
                        actual_ret_frac = pos.ret_7d / 100
                        if pos.direction == "SELL":
                            actual_ret_frac = -actual_ret_frac
                        # Apply final SL/TP cap even at expiry
                        if stop_loss is not None:
                            actual_ret_frac = max(actual_ret_frac, -stop_loss)
                        if take_profit is not None:
                            actual_ret_frac = min(actual_ret_frac, take_profit)
                        pos.cumret = actual_ret_frac
                    pnl = pos.close_now("7d")
                else:
                    # Legacy path: use ret_7d directly
                    if pos.ret_7d is None:
                        pos.pnl = 0.0
                        pos.closed = True
                        pos.exit_reason = "7d"
                        pnl = 0.0
                    else:
                        ret = pos.ret_7d / 100
                        if pos.direction == "SELL":
                            ret = -ret
                        if stop_loss is not None:
                            ret = max(ret, -stop_loss)
                        if take_profit is not None:
                            ret = min(ret, take_profit)
                        pos.actual_ret = ret
                        pos.pnl = pos.size_usd * ret
                        pos.closed = True
                        pos.exit_reason = "7d"
                        pnl = pos.pnl
                capital += pnl
                closed_positions.append(pos)
                monthly[pos.open_date.strftime("%Y-%m")] = monthly.get(pos.open_date.strftime("%Y-%m"), 0) + pnl
                outcome = "WIN" if pnl > 0 else "LOSS"
                actual_ret_pct = (pos.actual_ret or 0) * 100
                trade_log.append(f"  CLOSE {'LONG' if pos.direction=='BUY' else 'SHORT'} "
                                 f"{pos.open_date}→{row.date} ${pos.size_usd:,.0f} "
                                 f"ret={actual_ret_pct:+.2f}% pnl=${pnl:+,.0f} [{outcome}]")
                continue

            still_open.append(pos)
        positions = still_open

        # ── 3. Drawdown tracking ──────────────────────────────────────────────
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd

        # ── 4. Open new position ──────────────────────────────────────────────
        effective = row.signal if not (no_short and row.signal == "SELL") else "HOLD"
        if effective in ("BUY", "SELL") and row.ret_7d is not None:
            size = capital * (fixed_size if fixed_size is not None else row.confidence * size_pct_per_conf)
            pos = Position(
                open_date=row.date,
                close_date=row.date + timedelta(days=7),
                direction=effective,
                size_usd=size,
                confidence=row.confidence,
                ret_7d=row.ret_7d,
            )
            positions.append(pos)
            pct = (fixed_size or row.confidence * size_pct_per_conf) * 100
            trade_log.append(f"  OPEN  {'LONG ' if effective=='BUY' else 'SHORT'} {row.date} "
                             f"conf={row.confidence:.2f} ${size:,.0f} ({pct:.1f}% of capital)")

    # Close remaining with available data
    for pos in positions:
        if pos.ret_7d is not None:
            ret = pos.ret_7d / 100
            if pos.direction == "SELL":
                ret = -ret
            if stop_loss is not None:
                ret = max(ret, -stop_loss)
            if take_profit is not None:
                ret = min(ret, take_profit)
            pos.actual_ret = ret
            pos.pnl = pos.size_usd * ret
            pos.closed = True
            pos.exit_reason = "7d"
            capital += pos.pnl
            closed_positions.append(pos)
            monthly[pos.open_date.strftime("%Y-%m")] = monthly.get(pos.open_date.strftime("%Y-%m"), 0) + pos.pnl

    # ── Summary ───────────────────────────────────────────────────────────────
    start_d = rows[0].date if rows else date.today()
    end_d   = rows[-1].date if rows else date.today()
    days = (end_d - start_d).days

    total_pnl = capital - initial_capital
    total_ret  = total_pnl / initial_capital * 100
    wins   = [p for p in closed_positions if (p.pnl or 0) > 0]
    losses = [p for p in closed_positions if (p.pnl or 0) <= 0]
    buys   = [p for p in closed_positions if p.direction == "BUY"]
    sells  = [p for p in closed_positions if p.direction == "SELL"]
    avg_win  = sum(p.pnl for p in wins)   / len(wins)   if wins   else 0
    avg_loss = sum(p.pnl for p in losses) / len(losses) if losses else 0
    win_rate = len(wins) / len(closed_positions) * 100 if closed_positions else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss else 0

    # Exit reason breakdown
    exit_counts: dict[str, int] = {}
    for p in closed_positions:
        exit_counts[p.exit_reason] = exit_counts.get(p.exit_reason, 0) + 1

    metrics = dict(pnl=total_pnl, return_pct=total_ret, max_dd=max_drawdown,
                   trades=len(closed_positions), win_rate=win_rate, wl_ratio=wl_ratio,
                   long_pnl=sum(p.pnl or 0 for p in buys),
                   short_pnl=sum(p.pnl or 0 for p in sells),
                   exit_counts=exit_counts, monthly=monthly)

    if not silent:
        signals_count = {s: sum(1 for r in rows if r.signal == s) for s in ("BUY", "SELL", "HOLD")}
        print(f"\n{'='*65}")
        print(f"  PORTFOLIO  {start_d} → {end_d}  ({days}d)")
        print(f"{'='*65}")
        print(f"Signals: BUY={signals_count['BUY']} SELL={signals_count['SELL']} HOLD={signals_count['HOLD']}")
        print(f"\nCapital: ${initial_capital:>12,.0f}  →  ${capital:>12,.0f}")
        print(f"P&L:     ${total_pnl:>+12,.0f}  ({total_ret:+.2f}%)")
        print(f"Max DD:  {max_drawdown:.2f}%")
        print(f"\nTrades:   {len(closed_positions)}  (LONG {len(buys)} / SHORT {len(sells)})")
        print(f"Win rate: {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"W/L ratio:{wl_ratio:.2f}x  avg_win=${avg_win:+,.0f}  avg_loss=${avg_loss:+,.0f}")
        if buys:
            bw = [p for p in buys if (p.pnl or 0) > 0]
            print(f"LONG:  {len(buys)} trades  win={len(bw)/len(buys)*100:.1f}%  P&L=${sum(p.pnl or 0 for p in buys):+,.0f}")
        if sells:
            sw = [p for p in sells if (p.pnl or 0) > 0]
            print(f"SHORT: {len(sells)} trades  win={len(sw)/len(sells)*100:.1f}%  P&L=${sum(p.pnl or 0 for p in sells):+,.0f}")
        print(f"\nExit reasons: " + "  ".join(f"{k}={v}" for k, v in sorted(exit_counts.items())))
        if monthly:
            print(f"\nMonthly P&L:")
            max_v = max(abs(v) for v in monthly.values()) or 1
            for ym in sorted(monthly):
                v = monthly[ym]
                bar = "█" * int(abs(v) / max_v * 20)
                print(f"  {ym}  {'+' if v>=0 else '-'}${abs(v):>8,.0f}  {'+' if v>=0 else '-'}{bar}")
        print(f"\nTrade log (last 15):")
        for line in trade_log[-15:]:
            print(line)

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log",  help="Primary backtest log")
    p.add_argument("--log2", help="Second log for ensemble (only trade when both agree)")
    p.add_argument("--json", help="Saved JSON from model_backtest_compare --output")
    p.add_argument("--live", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end",   default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--capital",    type=float, default=1_000_000)
    p.add_argument("--size-mult",  type=float, default=0.15,
                   help="Position size = confidence × this (default 0.15 = 15%%)")
    p.add_argument("--fixed-size", type=float, default=None,
                   help="Fixed fraction per trade, e.g. 0.10 (overrides size-mult)")
    p.add_argument("--no-short",   action="store_true")
    p.add_argument("--stop-loss",  type=float, default=None, metavar="PCT")
    p.add_argument("--take-profit",type=float, default=None, metavar="PCT")
    p.add_argument("--trailing-stop", type=float, default=None, metavar="PCT")
    p.add_argument("--dynamic-exit", action="store_true",
                   help="Close position when opposite signal appears")
    p.add_argument("--random-runs", type=int, default=0)
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end) if args.end else date.today()

    if args.log:
        rows = parse_log(args.log, start, end)
        label = Path(args.log).stem
    elif args.json:
        rows = parse_json(args.json, start, end)
        label = Path(args.json).stem
    elif args.live:
        rows = await fetch_live(args.symbol, start, end, args.model, args.concurrency)
        label = args.model or "default"
    else:
        p.error("Specify --log, --json, or --live")

    if args.log2:
        rows2 = parse_log(args.log2, start, end)
        rows = ensemble_merge(rows, rows2)
        label += f"+{Path(args.log2).stem}"
        print(f"Ensemble: {label}")

    print(f"Loaded {len(rows)} dates | "
          f"BUY={sum(1 for r in rows if r.signal=='BUY')} "
          f"SELL={sum(1 for r in rows if r.signal=='SELL')} "
          f"HOLD={sum(1 for r in rows if r.signal=='HOLD')}")

    sim_kwargs = dict(
        initial_capital=args.capital,
        size_pct_per_conf=args.size_mult,
        no_short=args.no_short,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        trailing_stop=args.trailing_stop,
        dynamic_exit=args.dynamic_exit,
        fixed_size=args.fixed_size,
    )

    m = simulate(rows, **sim_kwargs)

    if args.random_runs > 0:
        rand_results = []
        for seed in range(args.random_runs):
            rrows = randomize_rows(rows, seed=seed)
            rm = simulate(rrows, **sim_kwargs, silent=True)
            rand_results.append(rm)
            print(f"  [random seed={seed}] ret={rm['return_pct']:+.2f}%  DD={rm['max_dd']:.2f}%  trades={rm['trades']}")
        avg_ret = sum(r["return_pct"] for r in rand_results) / len(rand_results)
        avg_dd  = sum(r["max_dd"]     for r in rand_results) / len(rand_results)
        avg_pnl = sum(r["pnl"]        for r in rand_results) / len(rand_results)
        print(f"\n{'─'*60}")
        print(f"  RANDOM avg:    ret={avg_ret:+.2f}%  DD={avg_dd:.2f}%  P&L=${avg_pnl:+,.0f}")
        print(f"  MODEL result:  ret={m['return_pct']:+.2f}%  DD={m['max_dd']:.2f}%  P&L=${m['pnl']:+,.0f}")
        print(f"  Edge:          {m['return_pct'] - avg_ret:+.2f}pp")


if __name__ == "__main__":
    asyncio.run(main())
