"""
eval_suite.py — ESWA-grade comprehensive evaluation for CEM-RAG retrievers.

Baselines:
  1. always_hold
  2. random_direction
  3. rsi_momentum          (indicator_vec[1] = RSI z-scored)
  4. sentiment_only        (indicator_vec[2] = sentiment_score z-scored)
  5. baseline_fixed_knn    (fixed weighted kNN from evaluate_retriever)
  6. learned_cem_rag       (learned diagonal metric from evaluate_retriever)

Metrics per baseline:
  da, balanced_acc, macro_f1, mcc, coverage, sharpe, sortino, max_dd, hit_at_5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from stockmem.scripts.cem_dataset import LabeledRow, label_rows, matured_pool
from stockmem.scripts.evaluate_retriever import (
    Evaluation,
    _fixed_score,
    evaluate,
    mcnemar_exact,
)
from stockmem.scripts.optimize_weights import _compute_sharpe, load_rows, validate_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric

DEFAULT_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _macro_f1(predictions: list[str], actuals: list[str]) -> float:
    """Macro-F1 for 3 classes: BUY, SELL, HOLD."""
    classes = ["BUY", "SELL", "HOLD"]
    f1s = []
    for cls in classes:
        tp = sum(p == cls and a == cls for p, a in zip(predictions, actuals))
        fp = sum(p == cls and a != cls for p, a in zip(predictions, actuals))
        fn = sum(p != cls and a == cls for p, a in zip(predictions, actuals))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def _mcc_multiclass(predictions: list[str], actuals: list[str]) -> float:
    """Multi-class MCC."""
    classes = ["BUY", "SELL", "HOLD"]
    class_idx = {c: i for i, c in enumerate(classes)}
    conf = np.zeros((3, 3), dtype=np.float64)
    for p, a in zip(predictions, actuals):
        if p in class_idx and a in class_idx:
            conf[class_idx[a], class_idx[p]] += 1
    t = conf.sum()
    if t == 0:
        return 0.0
    t_k = conf.sum(axis=1)  # actual counts per class
    p_k = conf.sum(axis=0)  # predicted counts per class
    c = np.trace(conf)
    numerator = c * t - float(np.dot(t_k, p_k))
    denom_sq = (t**2 - float(np.dot(p_k, p_k))) * (t**2 - float(np.dot(t_k, t_k)))
    if denom_sq <= 0:
        return 0.0
    return float(numerator / np.sqrt(denom_sq))


def _sortino(returns: list[float]) -> float:
    arr = np.array(returns, dtype=np.float64)
    downside = arr[arr < 0]
    if len(downside) < 2:
        return 0.0
    return float(np.mean(arr)) * np.sqrt(52) / float(np.std(downside, ddof=1) + 1e-12)


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    cum = np.cumprod(1 + np.array(returns, dtype=np.float64) / 100)
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / (roll_max + 1e-12)
    return float(dd.min())


def _brier_score(probs: list[float], actuals: list[bool]) -> float:
    """Brier score for a binary outcome (BUY/SELL correct)."""
    if not probs:
        return 0.0
    arr_p = np.array(probs, dtype=np.float64)
    arr_a = np.array(actuals, dtype=np.float64)
    return float(np.mean((arr_p - arr_a) ** 2))


def _block_bootstrap_sharpe_delta(
    baseline_returns: list[float],
    learned_returns: list[float],
    block_size: int = 7,
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Returns 95% CI for (learned_sharpe - baseline_sharpe)."""
    rng = np.random.default_rng(seed)
    n = len(baseline_returns)
    if n < block_size * 2:
        return 0.0, 0.0
    base = np.array(baseline_returns, dtype=np.float64)
    learn = np.array(learned_returns, dtype=np.float64)
    deltas = []
    for _ in range(n_samples):
        n_blocks = max(1, n // block_size)
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        b_sample = base[idx]
        l_sample = learn[idx]
        b_sharpe = float(np.mean(b_sample)) * np.sqrt(52) / (
            float(np.std(b_sample, ddof=1)) + 1e-12
        )
        l_sharpe = float(np.mean(l_sample)) * np.sqrt(52) / (
            float(np.std(l_sample, ddof=1)) + 1e-12
        )
        deltas.append(l_sharpe - b_sharpe)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


# ---------------------------------------------------------------------------
# Signal-to-actual helper
# ---------------------------------------------------------------------------

def _signal_is_correct(
    signal: str,
    actual_return: float,
    buy_threshold: float,
    sell_threshold: float,
) -> bool:
    return (
        (signal == "BUY" and actual_return > 0)
        or (signal == "SELL" and actual_return < 0)
        or (signal == "HOLD" and -sell_threshold <= actual_return <= buy_threshold)
    )


def _actual_signal(
    actual_return: float,
    buy_threshold: float,
    sell_threshold: float,
) -> str:
    if actual_return > buy_threshold:
        return "BUY"
    if actual_return < -sell_threshold:
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Extended metrics bundle
# ---------------------------------------------------------------------------

@dataclass
class ExtendedMetrics:
    da: float
    balanced_acc: float
    macro_f1: float
    mcc: float
    coverage: float
    sharpe: float
    sortino: float
    max_dd: float
    hit_at_5: float
    brier: float
    correct: list[bool] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    hit_correct: list[bool] = field(default_factory=list)


def _extend_from_evaluation(ev: Evaluation) -> ExtendedMetrics:
    """Derive extra metrics from an evaluate_retriever.Evaluation object."""
    # We need predictions and actuals — re-derive from what we can.
    # evaluate() only gives us correct/returns/metrics so compute approx balanced_acc
    m = ev.metrics
    buy_da = m.get("buy_da", 0.0)
    sell_da = m.get("sell_da", 0.0)
    hold_da = m.get("hold_da", 0.0)
    balanced_acc = (buy_da + sell_da + hold_da) / 3.0

    # macro_f1 / mcc — we do NOT have per-prediction labels from evaluate(); return 0
    # (these baselines are kNN-based and we'll compute predictions fresh below)
    sortino_val = _sortino(ev.returns)
    max_dd_val = _max_drawdown(ev.returns)

    return ExtendedMetrics(
        da=m["da"],
        balanced_acc=balanced_acc,
        macro_f1=0.0,   # placeholder — filled by compute_*_metrics where we have preds
        mcc=0.0,        # placeholder
        coverage=m["coverage"],
        sharpe=m["sharpe"],
        sortino=sortino_val,
        max_dd=max_dd_val,
        hit_at_5=m["hit_at_5"],
        brier=0.0,
        correct=ev.correct,
        returns=ev.returns,
        hit_correct=ev.hit_correct or [],
    )


# ---------------------------------------------------------------------------
# Per-baseline full metric computation (with predictions for F1/MCC)
# ---------------------------------------------------------------------------

def _compute_full_knn_metrics(
    labeled: Sequence[LabeledRow],
    score_fn,
    *,
    k: int,
    guard: bool,
    buy_threshold: float,
    sell_threshold: float,
) -> ExtendedMetrics:
    """Run kNN retrieval and collect per-prediction data for all metrics."""
    predictions: list[str] = []
    actuals: list[str] = []
    correct: list[bool] = []
    strategy_returns: list[float] = []
    hit_correct_list: list[bool] = []
    hit_count = hit_total = 0
    buy_probs: list[float] = []
    brier_actuals: list[bool] = []

    for query in labeled:
        if query.split != "test":
            continue
        pool = matured_pool(labeled, query, guard=guard)
        if not pool:
            continue

        scored = sorted(
            ((score_fn(query, c), c) for c in pool),
            key=lambda item: item[0],
            reverse=True,
        )[: max(k, 5)]
        ranked = [c for _, c in scored]
        downstream = ranked[:k]
        neighbor_returns = [row.row.future_return_7d for row in downstream]
        predicted_return = float(np.mean(neighbor_returns))

        if predicted_return > buy_threshold:
            signal = "BUY"
        elif predicted_return < -sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        actual_return = query.row.future_return_7d
        actual_label = _actual_signal(actual_return, buy_threshold, sell_threshold)
        is_correct = _signal_is_correct(signal, actual_return, buy_threshold, sell_threshold)

        predictions.append(signal)
        actuals.append(actual_label)
        correct.append(is_correct)

        if signal == "BUY":
            strategy_returns.append(actual_return)
        elif signal == "SELL":
            strategy_returns.append(-actual_return)
        else:
            strategy_returns.append(0.0)

        # Brier: prob of correct directional prediction for BUY/SELL queries
        if signal in ("BUY", "SELL"):
            # Simple probability proxy: fraction of neighbors agreeing with signal
            if signal == "BUY":
                prob = float(np.mean([r > 0 for r in neighbor_returns]))
            else:
                prob = float(np.mean([r < 0 for r in neighbor_returns]))
            buy_probs.append(prob)
            brier_actuals.append(is_correct)

        if query.direction != 0:
            hit_total += 1
            top_five = scored[:5]
            this_hit = any(row.direction == query.direction for _, row in top_five)
            hit_count += int(this_hit)
            hit_correct_list.append(this_hit)

    n = len(correct)
    if n == 0:
        return ExtendedMetrics(
            da=0.0, balanced_acc=0.0, macro_f1=0.0, mcc=0.0,
            coverage=0.0, sharpe=0.0, sortino=0.0, max_dd=0.0,
            hit_at_5=0.0, brier=0.0,
        )

    n_active = sum(1 for p in predictions if p in ("BUY", "SELL"))
    da = float(np.mean(correct))
    sharpe = _compute_sharpe(strategy_returns, horizon="7d", mode="nonoverlap")
    buy_da = (
        sum(c for p, c in zip(predictions, correct) if p == "BUY")
        / max(1, sum(1 for p in predictions if p == "BUY"))
    )
    sell_da = (
        sum(c for p, c in zip(predictions, correct) if p == "SELL")
        / max(1, sum(1 for p in predictions if p == "SELL"))
    )
    hold_da = (
        sum(c for p, c in zip(predictions, correct) if p == "HOLD")
        / max(1, sum(1 for p in predictions if p == "HOLD"))
    )

    return ExtendedMetrics(
        da=da,
        balanced_acc=(buy_da + sell_da + hold_da) / 3.0,
        macro_f1=_macro_f1(predictions, actuals),
        mcc=_mcc_multiclass(predictions, actuals),
        coverage=n_active / n,
        sharpe=sharpe,
        sortino=_sortino(strategy_returns),
        max_dd=_max_drawdown(strategy_returns),
        hit_at_5=hit_count / hit_total if hit_total else 0.0,
        brier=_brier_score(buy_probs, brier_actuals),
        correct=correct,
        returns=strategy_returns,
        hit_correct=hit_correct_list,
    )


def _compute_simple_baseline_metrics(
    labeled: Sequence[LabeledRow],
    get_signal_fn,
    *,
    buy_threshold: float,
    sell_threshold: float,
) -> ExtendedMetrics:
    """Metrics for non-retrieval baselines (always_hold, rsi_momentum, etc.)."""
    predictions: list[str] = []
    actuals: list[str] = []
    correct: list[bool] = []
    strategy_returns: list[float] = []

    for query in labeled:
        if query.split != "test":
            continue

        signal = get_signal_fn(query)
        actual_return = query.row.future_return_7d
        actual_label = _actual_signal(actual_return, buy_threshold, sell_threshold)
        is_correct = _signal_is_correct(signal, actual_return, buy_threshold, sell_threshold)

        predictions.append(signal)
        actuals.append(actual_label)
        correct.append(is_correct)

        if signal == "BUY":
            strategy_returns.append(actual_return)
        elif signal == "SELL":
            strategy_returns.append(-actual_return)
        else:
            strategy_returns.append(0.0)

    n = len(correct)
    if n == 0:
        return ExtendedMetrics(
            da=0.0, balanced_acc=0.0, macro_f1=0.0, mcc=0.0,
            coverage=0.0, sharpe=0.0, sortino=0.0, max_dd=0.0,
            hit_at_5=0.0, brier=0.0,
        )

    n_active = sum(1 for p in predictions if p in ("BUY", "SELL"))
    da = float(np.mean(correct))
    sharpe = _compute_sharpe(strategy_returns, horizon="7d", mode="nonoverlap")
    buy_da = (
        sum(c for p, c in zip(predictions, correct) if p == "BUY")
        / max(1, sum(1 for p in predictions if p == "BUY"))
    )
    sell_da = (
        sum(c for p, c in zip(predictions, correct) if p == "SELL")
        / max(1, sum(1 for p in predictions if p == "SELL"))
    )
    hold_da = (
        sum(c for p, c in zip(predictions, correct) if p == "HOLD")
        / max(1, sum(1 for p in predictions if p == "HOLD"))
    )

    return ExtendedMetrics(
        da=da,
        balanced_acc=(buy_da + sell_da + hold_da) / 3.0,
        macro_f1=_macro_f1(predictions, actuals),
        mcc=_mcc_multiclass(predictions, actuals),
        coverage=n_active / n,
        sharpe=sharpe,
        sortino=_sortino(strategy_returns),
        max_dd=_max_drawdown(strategy_returns),
        hit_at_5=0.0,  # non-retrieval baselines have no hit@5
        brier=0.0,
        correct=correct,
        returns=strategy_returns,
        hit_correct=[],
    )


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

COLUMNS = ["da", "balanced_acc", "macro_f1", "mcc", "coverage",
           "sharpe", "sortino", "max_dd", "hit_at_5"]

COL_WIDTHS = {
    "retriever": 22,
    "da": 7,
    "balanced_acc": 12,
    "macro_f1": 9,
    "mcc": 7,
    "coverage": 9,
    "sharpe": 7,
    "sortino": 8,
    "max_dd": 8,
    "hit_at_5": 8,
}


def _print_table(results: dict[str, ExtendedMetrics]) -> None:
    header = (
        f"{'retriever':<22} "
        + " ".join(f"{c:>{COL_WIDTHS[c]}}" for c in COLUMNS)
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = (
            f"{name:<22} "
            + f"{m.da:>{COL_WIDTHS['da']}.4f} "
            + f"{m.balanced_acc:>{COL_WIDTHS['balanced_acc']}.4f} "
            + f"{m.macro_f1:>{COL_WIDTHS['macro_f1']}.4f} "
            + f"{m.mcc:>{COL_WIDTHS['mcc']}.4f} "
            + f"{m.coverage:>{COL_WIDTHS['coverage']}.4f} "
            + f"{m.sharpe:>{COL_WIDTHS['sharpe']}.4f} "
            + f"{m.sortino:>{COL_WIDTHS['sortino']}.4f} "
            + f"{m.max_dd:>{COL_WIDTHS['max_dd']}.4f} "
            + f"{m.hit_at_5:>{COL_WIDTHS['hit_at_5']}.4f}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ESWA-grade eval suite for CEM-RAG")
    parser.add_argument("--data", default="stockmem/data/real_optimizer_v2.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/metrics")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    rows = load_rows(Path(args.data))
    validate_rows(rows)
    print(f"Loaded {len(rows)} rows from {args.data}")

    artifact_payload = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    metric = LearnedDiagonalMetric.from_payload(artifact_payload)
    band = str(artifact_payload.get("band", "0.5sigma"))
    labeled = label_rows(rows, band=band)

    # ------------------------------------------------------------------
    # Load thresholds and weights
    # ------------------------------------------------------------------
    weights = DEFAULT_WEIGHTS
    buy_threshold = 2.0
    sell_threshold = 2.0
    weights_path = Path(args.weights)
    if weights_path.exists():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        source = payload.get("weights", payload)
        weights = (
            float(source["w1_factor"]),
            float(source["w2_indicator"]),
            float(source["w3_price"]),
        )
        buy_threshold = float(payload.get("buy_threshold", buy_threshold))
        sell_threshold = abs(float(payload.get("sell_threshold", sell_threshold)))
    k = int(artifact_payload.get("hyperparameters", {}).get("k", 5))

    test_queries = [lr for lr in labeled if lr.split == "test"]
    print(f"Test split: {len(test_queries)} queries | band={band} | k={k}")
    print(f"buy_threshold={buy_threshold} | sell_threshold={sell_threshold}")
    print()

    # ------------------------------------------------------------------
    # Determine class prior from kNN baseline (for random_direction)
    # ------------------------------------------------------------------
    # We'll compute the class distribution from test predictions of fixed kNN
    # For simplicity: use the marginal from labeled test labels
    test_actual_labels = [
        _actual_signal(lr.row.future_return_7d, buy_threshold, sell_threshold)
        for lr in test_queries
    ]
    total_test = len(test_actual_labels)
    buy_prior = test_actual_labels.count("BUY") / max(1, total_test)
    sell_prior = test_actual_labels.count("SELL") / max(1, total_test)
    hold_prior = 1.0 - buy_prior - sell_prior

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    results: dict[str, ExtendedMetrics] = {}

    # 1. always_hold
    print("Computing: always_hold ...")
    results["always_hold"] = _compute_simple_baseline_metrics(
        labeled,
        lambda _q: "HOLD",
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # 2. random_direction (class-prior matching)
    print("Computing: random_direction ...")
    cum_probs = [buy_prior, buy_prior + sell_prior, 1.0]

    def _random_signal(_q: LabeledRow) -> str:
        r = rng.random()
        if r < cum_probs[0]:
            return "BUY"
        elif r < cum_probs[1]:
            return "SELL"
        return "HOLD"

    results["random_direction"] = _compute_simple_baseline_metrics(
        labeled,
        _random_signal,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # 3. rsi_momentum (indicator_vec[1] is RSI z-scored)
    print("Computing: rsi_momentum ...")
    RSI_OVERSOLD_Z = -1.0   # z-score roughly corresponding to RSI < 35
    RSI_OVERBOUGHT_Z = 1.0  # z-score roughly corresponding to RSI > 65

    def _rsi_signal(q: LabeledRow) -> str:
        rsi_z = float(q.row.indicator_vec[1])
        if rsi_z < RSI_OVERSOLD_Z:
            return "BUY"
        elif rsi_z > RSI_OVERBOUGHT_Z:
            return "SELL"
        return "HOLD"

    results["rsi_momentum"] = _compute_simple_baseline_metrics(
        labeled,
        _rsi_signal,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # 4. sentiment_only (indicator_vec[2] is sentiment_score z-scored)
    print("Computing: sentiment_only ...")
    SENTIMENT_BUY_Z = 0.5
    SENTIMENT_SELL_Z = -0.5

    def _sentiment_signal(q: LabeledRow) -> str:
        sent_z = float(q.row.indicator_vec[2])
        if sent_z > SENTIMENT_BUY_Z:
            return "BUY"
        elif sent_z < SENTIMENT_SELL_Z:
            return "SELL"
        return "HOLD"

    results["sentiment_only"] = _compute_simple_baseline_metrics(
        labeled,
        _sentiment_signal,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # 5. baseline_fixed_knn
    print("Computing: baseline_fixed_knn ...")

    def fixed_score(query: LabeledRow, candidate: LabeledRow) -> float:
        return _fixed_score(query, candidate, weights)

    results["baseline_fixed_knn"] = _compute_full_knn_metrics(
        labeled,
        fixed_score,
        k=k,
        guard=True,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # 6. learned_cem_rag
    print("Computing: learned_cem_rag ...")

    def learned_score(query: LabeledRow, candidate: LabeledRow) -> float:
        return metric.score(query.blocks, candidate.blocks)

    results["learned_cem_rag"] = _compute_full_knn_metrics(
        labeled,
        learned_score,
        k=k,
        guard=True,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("EVAL SUITE RESULTS (test split)")
    print("=" * 100)
    _print_table(results)
    print()

    # ------------------------------------------------------------------
    # Stat tests
    # ------------------------------------------------------------------
    baseline_ev = results["baseline_fixed_knn"]
    learned_ev = results["learned_cem_rag"]

    base_only, learned_only, p_value = mcnemar_exact(
        baseline_ev.correct, learned_ev.correct
    )
    print("McNemar test (learned_cem_rag vs baseline_fixed_knn):")
    print(f"  baseline_only_correct={base_only}, learned_only_correct={learned_only}, p={p_value:.6f}")
    sig = "SIGNIFICANT" if p_value < 0.05 else ("TREND" if p_value < 0.10 else "n.s.")
    print(f"  => {sig} (alpha=0.05)")
    print()

    ci_low, ci_high = _block_bootstrap_sharpe_delta(
        baseline_ev.returns,
        learned_ev.returns,
        block_size=7,
        n_samples=2000,
        seed=args.seed,
    )
    print("Block-bootstrap Sharpe delta 95% CI (learned - baseline, block=7):")
    print(f"  [{ci_low:.4f}, {ci_high:.4f}]")
    delta_significant = ci_low > 0
    print(f"  => {'Sharpe improvement is significant' if delta_significant else 'No significant Sharpe improvement'}")
    print()

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    # main_table.csv
    csv_path = os.path.join(args.output_dir, "main_table.csv")
    fieldnames = ["retriever"] + COLUMNS + ["brier"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in results.items():
            writer.writerow({
                "retriever": name,
                "da": round(m.da, 6),
                "balanced_acc": round(m.balanced_acc, 6),
                "macro_f1": round(m.macro_f1, 6),
                "mcc": round(m.mcc, 6),
                "coverage": round(m.coverage, 6),
                "sharpe": round(m.sharpe, 6),
                "sortino": round(m.sortino, 6),
                "max_dd": round(m.max_dd, 6),
                "hit_at_5": round(m.hit_at_5, 6),
                "brier": round(m.brier, 6),
            })
    print(f"Saved: {csv_path}")

    # stat_tests.json
    stat_tests = {
        "mcnemar_da": {
            "baseline": "baseline_fixed_knn",
            "learned": "learned_cem_rag",
            "baseline_only_correct": base_only,
            "learned_only_correct": learned_only,
            "p_value": round(p_value, 8),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.10": p_value < 0.10,
        },
        "block_bootstrap_sharpe_ci": {
            "baseline": "baseline_fixed_knn",
            "learned": "learned_cem_rag",
            "block_size": 7,
            "n_samples": 2000,
            "seed": args.seed,
            "ci_95_low": round(ci_low, 6),
            "ci_95_high": round(ci_high, 6),
            "delta_significant_positive": delta_significant,
        },
        "summary": {
            name: {
                "da": round(m.da, 6),
                "balanced_acc": round(m.balanced_acc, 6),
                "macro_f1": round(m.macro_f1, 6),
                "mcc": round(m.mcc, 6),
                "coverage": round(m.coverage, 6),
                "sharpe": round(m.sharpe, 6),
                "sortino": round(m.sortino, 6),
                "max_dd": round(m.max_dd, 6),
                "hit_at_5": round(m.hit_at_5, 6),
                "brier": round(m.brier, 6),
            }
            for name, m in results.items()
        },
    }
    json_path = os.path.join(args.output_dir, "stat_tests.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stat_tests, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
