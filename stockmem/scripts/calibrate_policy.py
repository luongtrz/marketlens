"""calibrate_policy.py

Calibrates a probability-based trading policy (p_up / p_down / p_hold)
using kNN-weighted probabilities derived from the learned retriever.

Workflow
--------
1. Load and label rows (band="0.5sigma").
2. Split into train / val / test using the canonical date boundaries from
   cem_dataset.py.
3. For each val/test query retrieve k nearest neighbors from the chronologically
   preceding, matured pool using BOTH the baseline weighted-cosine and the
   learned diagonal metric.
4. Compute kNN probability estimates.
5. Grid-search tau on VAL only; objective = Sharpe (coverage-gated).
6. Evaluate on TEST with the best val tau.
7. Report calibration quality (Brier score, ECE).
8. Save stockmem/config/policy.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from stockmem.scripts.cem_dataset import LabeledRow, label_rows, matured_pool
from stockmem.scripts.optimize_weights import _compute_sharpe, load_rows, validate_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric, load_learned_metric

# ---------------------------------------------------------------------------
# Core probability helpers
# ---------------------------------------------------------------------------

def _knn_probabilities(
    query: LabeledRow,
    pool: Sequence[LabeledRow],
    score_fn: Callable[[LabeledRow, LabeledRow], float],
    k: int = 5,
) -> tuple[float, float, float]:
    """Return (p_up, p_down, p_hold) using weighted kNN over *pool*.

    Similarity scores are shifted from [-1, 1] to [0, 1] so that even
    low-similarity neighbors contribute a small positive weight.
    """
    scored = sorted(
        ((score_fn(query, c), c) for c in pool),
        key=lambda x: x[0],
        reverse=True,
    )[:k]
    if not scored:
        return 1 / 3, 1 / 3, 1 / 3

    # shift scores to [0, 1]: sim_shifted = (sim + 1) / 2
    weights = [(s + 1) / 2 for s, _ in scored]
    total_w = sum(weights) + 1e-12
    p_up = (
        sum(w for w, (_, c) in zip(weights, scored) if c.row.future_return_7d > 0)
        / total_w
    )
    p_down = (
        sum(w for w, (_, c) in zip(weights, scored) if c.row.future_return_7d < 0)
        / total_w
    )
    p_hold = max(0.0, 1.0 - p_up - p_down)
    return p_up, p_down, p_hold


# ---------------------------------------------------------------------------
# Calibration-quality helpers
# ---------------------------------------------------------------------------

def _brier_score(
    probs: list[tuple[float, float, float]],
    actuals: list[int],
) -> float:
    """Multi-class Brier score. actual in {-1, 0, 1}."""
    if not probs:
        return 0.0
    total = 0.0
    for (pu, ph, pd), act in zip(probs, actuals):
        y = [float(act > 0), float(act == 0), float(act < 0)]
        total += (pu - y[0]) ** 2 + (ph - y[1]) ** 2 + (pd - y[2]) ** 2
    return total / len(probs)


def _ece(
    probs: list[float],
    actuals: list[bool],
    n_bins: int = 5,
) -> float:
    """Expected Calibration Error for binary p_up ~ actual_up."""
    if not probs:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [lo <= p < hi for p in probs]
        if not any(mask):
            continue
        bin_probs = [p for p, m in zip(probs, mask) if m]
        bin_acts = [a for a, m in zip(actuals, mask) if m]
        ece += len(bin_probs) / n * abs(np.mean(bin_probs) - np.mean(bin_acts))
    return float(ece)


# ---------------------------------------------------------------------------
# Strategy metrics helpers
# ---------------------------------------------------------------------------

def _sortino(returns: list[float]) -> float:
    """Annualized Sortino ratio (7-day returns, 52 periods/year)."""
    arr = np.array(returns, dtype=np.float64)
    downside = arr[arr < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std < 1e-12:
        return 0.0
    return float(np.mean(arr)) * np.sqrt(52) / downside_std


def _max_drawdown(returns: list[float]) -> float:
    """Max drawdown from cumulative returns (returns are in percent)."""
    cum = np.cumprod(1 + np.array(returns, dtype=np.float64) / 100)
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def _evaluate_split(
    labeled: list[LabeledRow],
    score_fn: Callable[[LabeledRow, LabeledRow], float],
    tau: float,
    k: int,
    split: str,
) -> dict:
    """Run the p_up/p_down tau-threshold policy on *split* queries.

    Returns a metrics dict and the raw probability/actual lists for
    calibration analysis.
    """
    queries = [row for row in labeled if row.split == split]
    all_probs: list[tuple[float, float, float]] = []
    all_actuals: list[int] = []
    strategy_returns: list[float] = []

    correct = buy_correct = sell_correct = 0
    buy_total = sell_total = hold_total = 0

    for query in queries:
        pool = matured_pool(labeled, query, guard=True)
        if not pool:
            continue

        p_up, p_down, p_hold = _knn_probabilities(query, pool, score_fn, k=k)
        diff_up = p_up - p_down
        diff_down = p_down - p_up

        if diff_up >= tau:
            signal = "BUY"
        elif diff_down >= tau:
            signal = "SELL"
        else:
            signal = "HOLD"

        actual_return = query.row.future_return_7d
        actual_dir = 1 if actual_return > 0 else (-1 if actual_return < 0 else 0)

        is_correct = (
            (signal == "BUY" and actual_return > 0)
            or (signal == "SELL" and actual_return < 0)
            or (signal == "HOLD")
        )
        correct += int(is_correct)

        if signal == "BUY":
            strategy_returns.append(actual_return)
            buy_total += 1
            buy_correct += int(actual_return > 0)
        elif signal == "SELL":
            strategy_returns.append(-actual_return)
            sell_total += 1
            sell_correct += int(actual_return < 0)
        else:
            strategy_returns.append(0.0)
            hold_total += 1

        all_probs.append((p_up, p_down, p_hold))
        all_actuals.append(actual_dir)

    total = buy_total + sell_total + hold_total
    if total == 0:
        return {
            "da": 0.0, "buy_da": 0.0, "sell_da": 0.0,
            "coverage": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "max_drawdown": 0.0, "brier": 0.0, "ece": 0.0,
            "n": 0, "n_buy": 0, "n_sell": 0, "n_hold": 0,
        }

    active_returns = [r for r in strategy_returns if r != 0.0]
    sharpe = _compute_sharpe(strategy_returns, horizon="7d", mode="nonoverlap")
    sortino = _sortino(active_returns) if active_returns else 0.0
    mdd = _max_drawdown(active_returns) if active_returns else 0.0
    coverage = (buy_total + sell_total) / total

    brier = _brier_score(all_probs, all_actuals)
    p_ups = [p[0] for p in all_probs]
    actual_ups = [a > 0 for a in all_actuals]
    ece = _ece(p_ups, actual_ups)

    return {
        "da": correct / total,
        "buy_da": buy_correct / buy_total if buy_total else 0.0,
        "sell_da": sell_correct / sell_total if sell_total else 0.0,
        "coverage": coverage,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "brier": brier,
        "ece": ece,
        "n": total,
        "n_buy": buy_total,
        "n_sell": sell_total,
        "n_hold": hold_total,
    }


# ---------------------------------------------------------------------------
# Tau grid search (val only)
# ---------------------------------------------------------------------------

def grid_search_tau(
    labeled: list[LabeledRow],
    score_fn: Callable[[LabeledRow, LabeledRow], float],
    k: int,
    coverage_min: float = 0.15,
    coverage_max: float = 0.70,
) -> tuple[float, float, list[dict]]:
    """Grid-search tau on the val split.

    Returns (best_tau, best_val_sharpe, all_results_list).
    """
    tau_grid = np.arange(0.02, 0.50, 0.02)
    results: list[dict] = []
    best_tau = 0.10
    best_sharpe = -np.inf

    for tau in tau_grid:
        tau_f = round(float(tau), 4)
        metrics = _evaluate_split(labeled, score_fn, tau_f, k, split="val")
        coverage = metrics["coverage"]
        sharpe = metrics["sharpe"]
        results.append({"tau": tau_f, **metrics})

        if coverage_min <= coverage <= coverage_max and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_tau = tau_f

    return best_tau, best_sharpe if best_sharpe > -np.inf else 0.0, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate kNN probability policy for MarketLens CEM-RAG"
    )
    parser.add_argument(
        "--data",
        default="stockmem/data/real_optimizer_v2.json",
        help="Path to vectorized dataset JSON",
    )
    parser.add_argument(
        "--artifact",
        default="stockmem/config/learned_retriever.json",
        help="Path to learned retriever artifact",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--output",
        default="stockmem/config/policy.json",
        help="Where to save the calibrated policy",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    artifact_path = Path(args.artifact)
    output_path = Path(args.output)

    # ------------------------------------------------------------------
    # 1. Load and label
    # ------------------------------------------------------------------
    print(f"Loading data from {data_path} ...")
    rows = load_rows(data_path)
    validate_rows(rows)
    labeled = label_rows(rows, band="0.5sigma")
    print(f"  {len(labeled)} total rows labeled")

    split_counts = {}
    for row in labeled:
        split_counts[row.split] = split_counts.get(row.split, 0) + 1
    for split, n in sorted(split_counts.items()):
        print(f"  {split}: {n} rows")

    # ------------------------------------------------------------------
    # 2. Build score functions
    # ------------------------------------------------------------------
    DEFAULT_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)

    def baseline_score(q: LabeledRow, c: LabeledRow) -> float:
        return (
            DEFAULT_WEIGHTS[0] * float(np.dot(q.row.factor_vec, c.row.factor_vec))
            + DEFAULT_WEIGHTS[1] * float(np.dot(q.row.indicator_vec, c.row.indicator_vec))
            + DEFAULT_WEIGHTS[2] * float(np.dot(q.row.price_vec, c.row.price_vec))
        )

    metric = load_learned_metric(artifact_path)
    if metric is None:
        print(f"[WARN] Learned retriever not found at {artifact_path}; using baseline only.")

    def learned_score(q: LabeledRow, c: LabeledRow) -> float:
        if metric is None:
            return baseline_score(q, c)
        return metric.score(q.blocks, c.blocks)

    # ------------------------------------------------------------------
    # 3. Val tau grid search — learned retriever
    # ------------------------------------------------------------------
    print("\nGrid-searching tau on VAL set (learned retriever) ...")
    best_tau, best_val_sharpe, val_grid = grid_search_tau(
        labeled, learned_score, k=args.k
    )
    print(f"  Best tau: {best_tau:.4f}  val_sharpe: {best_val_sharpe:.4f}")

    # Also run baseline for comparison
    print("Grid-searching tau on VAL set (baseline) ...")
    best_tau_base, best_val_sharpe_base, _ = grid_search_tau(
        labeled, baseline_score, k=args.k
    )
    print(f"  Baseline best tau: {best_tau_base:.4f}  val_sharpe: {best_val_sharpe_base:.4f}")

    # ------------------------------------------------------------------
    # 4. Test evaluation with best val tau
    # ------------------------------------------------------------------
    print(f"\nEvaluating on TEST set (tau={best_tau:.4f}) ...")

    test_learned = _evaluate_split(labeled, learned_score, best_tau, args.k, "test")
    test_baseline = _evaluate_split(labeled, baseline_score, best_tau_base, args.k, "test")

    # ------------------------------------------------------------------
    # 5. Print summary table
    # ------------------------------------------------------------------
    cols = [
        "da", "buy_da", "sell_da", "coverage",
        "sharpe", "sortino", "max_drawdown",
        "brier", "ece",
    ]
    header = f"{'retriever':<30}" + "".join(f"{c:>14}" for c in cols)
    print("\n" + "=" * (30 + 14 * len(cols)))
    print("TEST SET RESULTS")
    print("=" * (30 + 14 * len(cols)))
    print(header)
    for name, metrics in [
        (f"baseline (tau={best_tau_base:.2f})", test_baseline),
        (f"learned  (tau={best_tau:.2f})", test_learned),
    ]:
        row_str = f"{name:<30}" + "".join(f"{metrics[c]:>14.4f}" for c in cols)
        print(row_str)
    print("=" * (30 + 14 * len(cols)))

    print(f"\n  TEST n={test_learned['n']}  BUY={test_learned['n_buy']}  "
          f"SELL={test_learned['n_sell']}  HOLD={test_learned['n_hold']}")

    # ------------------------------------------------------------------
    # 6. Save policy.json
    # ------------------------------------------------------------------
    policy = {
        "tau": best_tau,
        "val_sharpe": round(best_val_sharpe, 6),
        "method": "knn_platt_v1",
        "retriever": "learned_diagonal",
        "k": args.k,
        "created_at": datetime.now().isoformat(),
        "test_metrics": {k: round(v, 6) for k, v in test_learned.items()},
        "val_best_metrics": {
            row["tau"]: {m: round(row[m], 6) for m in ["sharpe", "coverage", "da"]}
            for row in val_grid
            if abs(row["tau"] - best_tau) < 1e-6
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"\nPolicy saved to {output_path}")

    # Final summary line (easy to grep)
    print(
        f"\nSUMMARY  tau={best_tau:.4f}  val_sharpe={best_val_sharpe:.4f}"
        f"  test_DA={test_learned['da']:.4f}  test_sharpe={test_learned['sharpe']:.4f}"
        f"  coverage={test_learned['coverage']:.4f}"
        f"  brier={test_learned['brier']:.4f}  ece={test_learned['ece']:.4f}"
    )


if __name__ == "__main__":
    main()
