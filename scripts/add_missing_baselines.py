"""
add_missing_baselines.py — Add buy_and_hold and random_retrieval baselines.

Outputs:
  artifacts/predictions/buy_and_hold_test.jsonl
  artifacts/predictions/random_retrieval_test.jsonl
  artifacts/metrics/main_table.csv   (two rows appended)

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/add_missing_baselines.py
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stockmem.scripts.cem_dataset import label_rows, matured_pool
from stockmem.scripts.optimize_weights import load_rows

# ── Config ──────────────────────────────────────────────────────────────────────
SYMBOL           = "BTC"
K                = 5
CEM_TAU          = 0.22
HORIZON          = "7d"
HORIZONS         = ["1d", "3d", "7d", "15d", "30d"]
COST_PCT         = 0.10        # 10 bps per trade
PERIODS_PER_YEAR = 252 / 7
RANDOM_SIM       = 0.5         # uniform sim for random_retrieval neighbors
SEED             = 42

DATA_PATH    = PROJECT_ROOT / "stockmem/data/real_optimizer_v3.json"
OUT_PRED_DIR = PROJECT_ROOT / "artifacts/predictions"
MAIN_TABLE   = PROJECT_ROOT / "artifacts/metrics/main_table.csv"

MAIN_TABLE_FIELDS = [
    "retriever", "da", "balanced_acc", "macro_f1", "mcc",
    "coverage", "sharpe", "sortino", "max_dd", "hit_at_5", "brier",
]


# ── CEM-RAG signal (same as compute_ablation_table.py) ──────────────────────────

def _cem_signal(
    neighbors,
    sims: list[float],
    tau: float,
    horizon: str = "7d",
) -> tuple[str, float, float, float, float]:
    attr = f"future_return_{horizon}"
    usable = [
        (s, nb)
        for s, nb in zip(sims, neighbors)
        if getattr(nb.row, attr, None) is not None
    ]
    if not usable:
        return "HOLD", 1 / 3, 1 / 3, 1 / 3, 0.50

    weights = [(s + 1) / 2 for s, _ in usable]
    total_w = sum(weights) + 1e-12
    p_up   = sum(w for w, (_, nb) in zip(weights, usable) if getattr(nb.row, attr) > 0) / total_w
    p_down = sum(w for w, (_, nb) in zip(weights, usable) if getattr(nb.row, attr) < 0) / total_w
    p_hold = max(0.0, 1.0 - p_up - p_down)
    du, dd = p_up - p_down, p_down - p_up

    if tau <= 0:
        if p_up >= p_down:
            return "BUY",  round(min(0.50 + du * 0.80, 0.95), 3), p_up, p_down, p_hold
        else:
            return "SELL", round(min(0.50 + dd * 0.80, 0.95), 3), p_up, p_down, p_hold

    if du >= tau:
        return "BUY",  round(min(0.50 + du * 0.80, 0.95), 3), p_up, p_down, p_hold
    if dd >= tau:
        return "SELL", round(min(0.50 + dd * 0.80, 0.95), 3), p_up, p_down, p_hold
    return "HOLD", round(0.50 + max(du, dd) * 0.30, 3), p_up, p_down, p_hold


# ── Metric helpers ───────────────────────────────────────────────────────────────

def _sharpe(rets: list[float]) -> float:
    """Annualised Sharpe with PERIODS_PER_YEAR = 252/7."""
    if len(rets) < 2:
        return 0.0
    arr = np.asarray(rets, dtype=np.float64)
    mean = arr.mean() * PERIODS_PER_YEAR
    std  = arr.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)
    return float(mean / std) if std > 1e-9 else 0.0


def _sortino(rets: list[float]) -> float:
    arr = np.asarray(rets, dtype=np.float64)
    downside = arr[arr < 0]
    if len(downside) < 2:
        return 0.0
    mean = arr.mean() * PERIODS_PER_YEAR
    semi = downside.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)
    return float(mean / semi) if semi > 1e-9 else 0.0


def _max_drawdown(rets: list[float]) -> float:
    if not rets:
        return 0.0
    cum = np.cumprod(1 + np.asarray(rets, dtype=np.float64) / 100)
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / (roll_max + 1e-12)
    return float(dd.min())


def _compute_metrics(rows_out: list[dict]) -> dict:
    """Compute main_table.csv metrics from a list of prediction dicts."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

    predictions: list[str] = []
    actuals_dir: list[str] = []   # binary: BUY / SELL based on ret7>0
    strategy_rets: list[float] = []
    buy_correct = sell_correct = hold_correct = 0
    n_buy = n_sell = n_hold = 0

    for r in rows_out:
        ret7 = r.get("actual_return_7d")
        if ret7 is None:
            continue
        ret7 = float(ret7)
        sig  = r["signal"]

        predictions.append(sig)
        actuals_dir.append("BUY" if ret7 > 0 else "SELL")

        if sig == "BUY":
            n_buy += 1
            strategy_rets.append(ret7 - COST_PCT)
            buy_correct += int(ret7 > 0)
        elif sig == "SELL":
            n_sell += 1
            strategy_rets.append(-ret7 - COST_PCT)
            sell_correct += int(ret7 < 0)
        else:
            n_hold += 1
            strategy_rets.append(0.0)
            hold_correct += int(True)   # HOLD is always "correct" in DA convention here

    n = len(predictions)
    if n == 0:
        return {k: 0.0 for k in MAIN_TABLE_FIELDS if k != "retriever"}

    n_active = n_buy + n_sell

    # DA: fraction correct (BUY→ret>0, SELL→ret<0, HOLD→always correct)
    da_list = [
        (r["signal"] == "BUY"  and float(r["actual_return_7d"]) > 0)
        or (r["signal"] == "SELL" and float(r["actual_return_7d"]) < 0)
        or (r["signal"] == "HOLD")
        for r in rows_out
        if r.get("actual_return_7d") is not None
    ]
    da = float(np.mean(da_list)) if da_list else 0.0

    buy_da  = buy_correct  / n_buy   if n_buy   else 0.0
    sell_da = sell_correct / n_sell  if n_sell  else 0.0
    # hold always correct, so hold_da = 1.0 for HOLD signals
    hold_da_count = n_hold
    hold_da = 1.0 if n_hold > 0 else 0.0

    # balanced_acc via sklearn (binary y_true)
    ba  = balanced_accuracy_score(actuals_dir, predictions)
    mf1 = f1_score(
        actuals_dir, predictions,
        labels=["BUY", "SELL", "HOLD"], average="macro", zero_division=0,
    )
    mcc = matthews_corrcoef(actuals_dir, predictions) if len(set(actuals_dir)) > 1 else 0.0

    coverage = n_active / n if n else 0.0
    sharpe   = _sharpe(strategy_rets)
    sortino  = _sortino(strategy_rets)
    max_dd   = _max_drawdown(strategy_rets)

    # hit_at_5 and brier not applicable for these baselines
    hit_at_5 = 0.0
    brier    = 0.0

    return {
        "da":           round(da, 6),
        "balanced_acc": round(ba, 6),
        "macro_f1":     round(mf1, 6),
        "mcc":          round(mcc, 6),
        "coverage":     round(coverage, 6),
        "sharpe":       round(sharpe, 6),
        "sortino":      round(sortino, 6),
        "max_dd":       round(max_dd, 6),
        "hit_at_5":     round(hit_at_5, 6),
        "brier":        round(brier, 6),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(rows)} rows → {path}")


# ── Baseline 1: buy_and_hold ─────────────────────────────────────────────────────

def run_buy_and_hold(labeled) -> list[dict]:
    rows_out: list[dict] = []
    for query in labeled:
        if query.split != "test":
            continue
        actual = {h: getattr(query.row, f"future_return_{h}", None) for h in HORIZONS}
        rows_out.append({
            "date":     query.row.date,
            "symbol":   SYMBOL,
            "split":    "test",
            "signal":   "BUY",
            "confidence": 0.60,
            "p_up":     1.0,
            "p_down":   0.0,
            "p_hold":   0.0,
            **{f"actual_return_{h}": actual[h] for h in HORIZONS},
        })
    return rows_out


# ── Baseline 2: random_retrieval ─────────────────────────────────────────────────

def run_random_retrieval(labeled, rng: random.Random) -> list[dict]:
    rows_out: list[dict] = []
    for query in labeled:
        if query.split != "test":
            continue
        actual = {h: getattr(query.row, f"future_return_{h}", None) for h in HORIZONS}

        pool = matured_pool(labeled, query, guard=True)
        if not pool:
            rows_out.append({
                "date":     query.row.date,
                "symbol":   SYMBOL,
                "split":    "test",
                "signal":   "HOLD",
                "confidence": 0.50,
                "p_up":     1 / 3,
                "p_down":   1 / 3,
                "p_hold":   1 / 3,
                **{f"actual_return_{h}": actual[h] for h in HORIZONS},
            })
            continue

        neighbors = rng.sample(pool, min(K, len(pool)))
        # uniform sim = 0.5 for all random neighbors (no scoring)
        sims = [RANDOM_SIM] * len(neighbors)

        signal, conf, p_up, p_down, p_hold = _cem_signal(neighbors, sims, CEM_TAU, HORIZON)
        rows_out.append({
            "date":       query.row.date,
            "symbol":     SYMBOL,
            "split":      "test",
            "signal":     signal,
            "confidence": conf,
            "p_up":       round(p_up, 4),
            "p_down":     round(p_down, 4),
            "p_hold":     round(p_hold, 4),
            **{f"actual_return_{h}": actual[h] for h in HORIZONS},
        })
    return rows_out


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)
    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    rows    = load_rows(DATA_PATH)
    labeled = label_rows(rows)
    test_n  = sum(1 for lr in labeled if lr.split == "test")
    print(f"  Loaded {len(rows)} total rows, {test_n} test rows")

    # ── buy_and_hold ──────────────────────────────────────────────────────────
    print("\n[1/2] buy_and_hold")
    bah_rows = run_buy_and_hold(labeled)
    _write_jsonl(OUT_PRED_DIR / "buy_and_hold_test.jsonl", bah_rows)
    bah_metrics = _compute_metrics(bah_rows)
    print("  Metrics:", {k: v for k, v in bah_metrics.items()})

    # ── random_retrieval ──────────────────────────────────────────────────────
    print("\n[2/2] random_retrieval")
    rr_rows = run_random_retrieval(labeled, rng)
    _write_jsonl(OUT_PRED_DIR / "random_retrieval_test.jsonl", rr_rows)
    rr_metrics = _compute_metrics(rr_rows)
    print("  Metrics:", {k: v for k, v in rr_metrics.items()})

    # ── Append to main_table.csv ──────────────────────────────────────────────
    print(f"\nAppending to {MAIN_TABLE}…")

    # Read existing rows
    existing: list[dict] = []
    new_names = {"buy_and_hold", "random_retrieval"}
    if MAIN_TABLE.exists():
        with MAIN_TABLE.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["retriever"] not in new_names:
                    existing.append(row)

    # Build new rows
    new_rows = [
        {"retriever": "buy_and_hold",     **bah_metrics},
        {"retriever": "random_retrieval", **rr_metrics},
    ]

    all_rows = existing + new_rows
    with MAIN_TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MAIN_TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  main_table.csv now has {len(all_rows)} rows")
    print("\nDone.")


if __name__ == "__main__":
    main()
