"""Compute trading metrics after transaction costs for all frozen prediction files.

Outputs:
  artifacts/metrics/trading_table.csv   -- per-model, per-cost-tier metrics

Metrics per model:
  Signal distribution: n_buy, n_sell, n_hold, coverage
  Directional: buy_da, sell_da, directional_da (buy+sell only)
  Trading (after cost): net_return_pct, sharpe, sortino, max_drawdown
  Transaction cost sensitivity: reported at 5bps, 10bps, 20bps

Transaction cost convention:
  BUY  return = future_return_7d - cost
  SELL return = -future_return_7d - cost
  HOLD return = 0
  cost is one-way, applied once per trade entry (round-trip split: half entry, half exit)
  Default: 10 bps = 0.10%

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/compute_trading_table.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "artifacts" / "predictions"
OUT_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "trading_table.csv"

COST_TIERS = [0.0, 0.05, 0.10, 0.20]   # % per trade entry (one-way)
HORIZON_DAYS = 7
PERIODS_PER_YEAR = 252 / HORIZON_DAYS   # ~36 non-overlapping 7d windows per year

# Ordered display names
MODEL_ORDER = [
    "always_hold",
    "random_direction",
    "rsi_momentum",
    "sentiment_only",
    "fixed_knn_test",
    "knn_returns_test",
    "cem_rag_test",
    "xgboost_test",
    "xgboost_price_only_test",
    "xgboost_event_only_test",
    "xgboost_no_event_test",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["date"])


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    mean = arr.mean() * PERIODS_PER_YEAR
    std  = arr.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)
    return float(mean / std) if std > 1e-9 else 0.0


def _sortino(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    mean = arr.mean() * PERIODS_PER_YEAR
    downside = arr[arr < 0]
    dstd = downside.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR) if len(downside) > 1 else 1e-9
    return float(mean / dstd) if dstd > 1e-9 else 0.0


def _max_drawdown(returns: list[float]) -> float:
    """Max peak-to-trough drawdown on cumulative return curve (in %)."""
    if not returns:
        return 0.0
    cum = np.cumprod(1 + np.asarray(returns, dtype=float) / 100) * 100
    peak = cum[0]
    max_dd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return float(max_dd * 100)  # in %


def _net_return(returns_after_cost: list[float]) -> float:
    if not returns_after_cost:
        return 0.0
    cum = np.prod(1 + np.asarray(returns_after_cost, dtype=float) / 100)
    return float((cum - 1) * 100)


def compute_metrics(rows: list[dict], cost_pct: float) -> dict:
    n_buy = n_sell = n_hold = 0
    buy_correct = sell_correct = 0
    strategy_returns: list[float] = []

    for r in rows:
        signal = r.get("signal", "HOLD")
        ret7 = r.get("actual_return_7d")
        if ret7 is None:
            continue
        ret7 = float(ret7)

        if signal == "BUY":
            n_buy += 1
            net = ret7 - cost_pct
            strategy_returns.append(net)
            buy_correct += int(ret7 > 0)
        elif signal == "SELL":
            n_sell += 1
            net = -ret7 - cost_pct
            strategy_returns.append(net)
            sell_correct += int(ret7 < 0)
        else:
            n_hold += 1
            strategy_returns.append(0.0)

    n_total = n_buy + n_sell + n_hold
    n_directional = n_buy + n_sell
    coverage = n_directional / n_total if n_total > 0 else 0.0
    buy_da   = buy_correct   / n_buy  if n_buy  > 0 else 0.0
    sell_da  = sell_correct  / n_sell if n_sell > 0 else 0.0
    directional_da = (buy_correct + sell_correct) / n_directional if n_directional > 0 else 0.0

    return {
        "n":                n_total,
        "n_buy":            n_buy,
        "n_sell":           n_sell,
        "n_hold":           n_hold,
        "coverage":         round(coverage, 4),
        "buy_da":           round(buy_da, 4),
        "sell_da":          round(sell_da, 4),
        "directional_da":   round(directional_da, 4),
        "net_return_pct":   round(_net_return(strategy_returns), 3),
        "sharpe":           round(_sharpe(strategy_returns), 4),
        "sortino":          round(_sortino(strategy_returns), 4),
        "max_drawdown_pct": round(_max_drawdown(strategy_returns), 3),
    }


def run() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Gather all prediction files
    pred_files = sorted(PRED_DIR.glob("*_test.jsonl"))
    if not pred_files:
        print(f"No JSONL files found in {PRED_DIR}")
        return

    all_rows: list[dict] = []
    fieldnames = [
        "model", "cost_bps", "n", "n_buy", "n_sell", "n_hold",
        "coverage", "buy_da", "sell_da", "directional_da",
        "net_return_pct", "sharpe", "sortino", "max_drawdown_pct",
    ]

    print(f"{'Model':<30} {'Cost':>6}  {'cov':>5}  {'BUY-DA':>7}  {'SELL-DA':>7}  {'Dir-DA':>7}  {'Sharpe':>7}  {'NetRet%':>8}")
    print("-" * 90)

    for path in pred_files:
        model_name = path.stem  # e.g. "cem_rag_test"
        rows = _load_jsonl(path)
        for cost in COST_TIERS:
            m = compute_metrics(rows, cost)
            row = {"model": model_name, "cost_bps": round(cost * 100, 1), **m}
            all_rows.append(row)
            if cost in (0.0, 0.10):  # print only 0 and 10bps for readability
                print(
                    f"{model_name:<30} {cost*100:>5.0f}bp  "
                    f"{m['coverage']:>5.3f}  {m['buy_da']:>7.3f}  {m['sell_da']:>7.3f}  "
                    f"{m['directional_da']:>7.3f}  {m['sharpe']:>7.3f}  {m['net_return_pct']:>8.2f}%"
                )

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows → {OUT_PATH}")


if __name__ == "__main__":
    run()
