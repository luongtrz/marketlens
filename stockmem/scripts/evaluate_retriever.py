from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from stockmem.scripts.cem_dataset import LabeledRow, label_rows, matured_pool
from stockmem.scripts.optimize_weights import _compute_sharpe, load_rows, validate_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


DEFAULT_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)


@dataclass(frozen=True)
class Evaluation:
    metrics: dict[str, float]
    correct: list[bool]
    returns: list[float]
    hit_correct: list[bool] = None  # type: ignore[assignment]


def _fixed_score(
    query: LabeledRow,
    candidate: LabeledRow,
    weights: tuple[float, float, float],
) -> float:
    return (
        weights[0] * float(np.dot(query.row.factor_vec, candidate.row.factor_vec))
        + weights[1]
        * float(np.dot(query.row.indicator_vec, candidate.row.indicator_vec))
        + weights[2] * float(np.dot(query.row.price_vec, candidate.row.price_vec))
    )


def evaluate(
    labeled: Sequence[LabeledRow],
    score: Callable[[LabeledRow, LabeledRow], float],
    *,
    k: int,
    guard: bool,
    split: str = "test",
    buy_threshold: float = 2.0,
    sell_threshold: float = 2.0,
) -> Evaluation:
    correct: list[bool] = []
    strategy_returns: list[float] = []
    hit_correct_list: list[bool] = []
    buy_correct = buy_total = sell_correct = sell_total = hold_correct = hold_total = 0
    hit_count = hit_total = 0
    same_scores: list[float] = []
    opposite_scores: list[float] = []

    for query in labeled:
        if query.split != split:
            continue
        pool = matured_pool(labeled, query, guard=guard)
        if not pool:
            continue
        scored = sorted(
            ((score(query, candidate), candidate) for candidate in pool),
            key=lambda item: item[0],
            reverse=True,
        )[: max(k, 5)]
        ranked = [candidate for _, candidate in scored]
        downstream_neighbors = ranked[:k]
        predicted_return = float(
            np.mean([row.row.future_return_7d for row in downstream_neighbors])
        )
        if predicted_return > buy_threshold:
            signal = "BUY"
        elif predicted_return < -sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        actual_return = query.row.future_return_7d
        is_correct = (
            (signal == "BUY" and actual_return > 0)
            or (signal == "SELL" and actual_return < 0)
            or (
                signal == "HOLD"
                and -sell_threshold <= actual_return <= buy_threshold
            )
        )
        correct.append(is_correct)
        if signal == "BUY":
            strategy_returns.append(actual_return)
            buy_total += 1
            buy_correct += int(is_correct)
        elif signal == "SELL":
            strategy_returns.append(-actual_return)
            sell_total += 1
            sell_correct += int(is_correct)
        else:
            strategy_returns.append(0.0)
            hold_total += 1
            hold_correct += int(is_correct)
        if query.direction != 0:
            hit_total += 1
            top_five = scored[:5]
            this_hit = any(row.direction == query.direction for _, row in top_five)
            hit_count += int(this_hit)
            hit_correct_list.append(this_hit)
            same_scores.extend(
                candidate_score
                for candidate_score, row in top_five
                if row.direction == query.direction
            )
            opposite_scores.extend(
                candidate_score
                for candidate_score, row in top_five
                if row.direction == -query.direction
            )

    da = float(np.mean(correct)) if correct else 0.0
    sharpe = _compute_sharpe(strategy_returns, horizon="7d", mode="nonoverlap")
    combined = 0.6 * da + 0.4 * min(max(sharpe, -2.0), 2.0) / 2.0
    metrics = {
        "hit_at_5": hit_count / hit_total if hit_total else 0.0,
        "hard_negative_gap": (
            float(np.mean(same_scores)) - float(np.mean(opposite_scores))
            if same_scores and opposite_scores
            else 0.0
        ),
        "da": da,
        "buy_da": buy_correct / buy_total if buy_total else 0.0,
        "sell_da": sell_correct / sell_total if sell_total else 0.0,
        "hold_da": hold_correct / hold_total if hold_total else 0.0,
        "coverage": (buy_total + sell_total) / len(correct) if correct else 0.0,
        "sharpe": sharpe,
        "combined": combined,
        "n": float(len(correct)),
    }
    return Evaluation(metrics=metrics, correct=correct, returns=strategy_returns, hit_correct=hit_correct_list)


def mcnemar_exact(left: Sequence[bool], right: Sequence[bool]) -> tuple[int, int, float]:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(left_only, right_only) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return left_only, right_only, p_value


def bootstrap_da_delta(
    baseline: Sequence[bool],
    learned: Sequence[bool],
    *,
    seed: int = 42,
    samples: int = 5000,
) -> tuple[float, float]:
    if not baseline:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    base = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(learned, dtype=np.float64)
    positions = rng.integers(0, base.size, size=(samples, base.size))
    deltas = candidate[positions].mean(axis=1) - base[positions].mean(axis=1)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _zero_block_scale(scales: np.ndarray, index: int) -> np.ndarray:
    ablated = np.asarray(scales, dtype=np.float64).copy()
    original_total = float(ablated.sum())
    ablated[index] = 0.0
    remaining_total = float(ablated.sum())
    if remaining_total <= 0.0:
        raise ValueError("Cannot ablate the only active retriever block")
    return ablated * (original_total / remaining_total)


def _print_table(results: dict[str, Evaluation]) -> None:
    columns = [
        "hit_at_5",
        "da",
        "buy_da",
        "sell_da",
        "hold_da",
        "coverage",
        "sharpe",
        "combined",
    ]
    print("retriever\t" + "\t".join(columns))
    for name, result in results.items():
        print(name + "\t" + "\t".join(f"{result.metrics[key]:.4f}" for key in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed and learned StockMem retrievers")
    parser.add_argument("--data", default="stockmem/data/real_optimizer.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--buy-threshold", type=float, default=None)
    parser.add_argument("--sell-threshold", type=float, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    validate_rows(rows)
    artifact_payload = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    metric = LearnedDiagonalMetric.from_payload(artifact_payload)
    band = str(artifact_payload.get("band", "0.5sigma"))
    labeled = label_rows(rows, band=band)
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
    if args.buy_threshold is not None:
        buy_threshold = args.buy_threshold
    if args.sell_threshold is not None:
        sell_threshold = abs(args.sell_threshold)
    k = args.k or int(artifact_payload.get("hyperparameters", {}).get("k", 5))
    def fixed(query: LabeledRow, candidate: LabeledRow) -> float:
        return _fixed_score(query, candidate, weights)

    def learned(query: LabeledRow, candidate: LabeledRow) -> float:
        return metric.score(query.blocks, candidate.blocks)
    results = {
        "baseline_fixed_guarded": evaluate(
            labeled,
            fixed,
            k=k,
            guard=True,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ),
        "baseline_fixed_leaky": evaluate(
            labeled,
            fixed,
            k=k,
            guard=False,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ),
        "learned_diagonal": evaluate(
            labeled,
            learned,
            k=k,
            guard=True,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ),
    }

    factor_index = 1 if len(metric.block_dims) == 4 else 0
    factor_zero_scales = _zero_block_scale(metric.block_scales, factor_index)
    factor_zero_metric = LearnedDiagonalMetric(
        block_dims=metric.block_dims,
        diagonal=metric.diagonal.copy(),
        block_scales=factor_zero_scales,
    )
    if len(metric.block_dims) == 4:
        event_zero_scales = _zero_block_scale(metric.block_scales, 0)
        event_zero_metric = LearnedDiagonalMetric(
            block_dims=metric.block_dims,
            diagonal=metric.diagonal.copy(),
            block_scales=event_zero_scales,
        )
        results["learned_event_zeroed"] = evaluate(
            labeled,
            lambda query, candidate: event_zero_metric.score(
                query.blocks,
                candidate.blocks,
            ),
            k=k,
            guard=True,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
    results["learned_factor_zeroed"] = evaluate(
        labeled,
        lambda query, candidate: factor_zero_metric.score(query.blocks, candidate.blocks),
        k=k,
        guard=True,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    fixed_band_rows = label_rows(rows, band="fixed")
    results["learned_fixed_band"] = evaluate(
        fixed_band_rows,
        lambda query, candidate: metric.score(query.blocks, candidate.blocks),
        k=k,
        guard=True,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    _print_table(results)

    baseline = results["baseline_fixed_guarded"]
    candidate = results["learned_diagonal"]
    val_baseline = evaluate(
        labeled,
        fixed,
        k=k,
        guard=True,
        split="val",
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    val_candidate = evaluate(
        labeled,
        learned,
        k=k,
        guard=True,
        split="val",
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    val_delta = val_candidate.metrics["combined"] - val_baseline.metrics["combined"]
    test_delta = candidate.metrics["combined"] - baseline.metrics["combined"]
    val_hit_delta = val_candidate.metrics["hit_at_5"] - val_baseline.metrics["hit_at_5"]
    test_hit_delta = candidate.metrics["hit_at_5"] - baseline.metrics["hit_at_5"]
    base_only, learned_only, p_value = mcnemar_exact(baseline.correct, candidate.correct)
    hit_base_only, hit_learned_only, hit_p_value = mcnemar_exact(
        baseline.hit_correct or [], candidate.hit_correct or []
    )
    ci_low, ci_high = bootstrap_da_delta(
        baseline.correct,
        candidate.correct,
        samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "mcnemar_da": {
                    "baseline_only_correct": base_only,
                    "learned_only_correct": learned_only,
                    "p_value": p_value,
                },
                "mcnemar_hit_at_5": {
                    "baseline_only_hit": hit_base_only,
                    "learned_only_hit": hit_learned_only,
                    "p_value": hit_p_value,
                },
                "bootstrap_da_delta_95pct": [ci_low, ci_high],
                "leak_delta_combined": (
                    results["baseline_fixed_leaky"].metrics["combined"]
                    - baseline.metrics["combined"]
                ),
                "acceptance": {
                    "combined_plus_0_01": (
                        candidate.metrics["combined"] >= baseline.metrics["combined"] + 0.01
                    ),
                    "balanced_action_da_plus_1pp": (
                        (candidate.metrics["buy_da"] + candidate.metrics["sell_da"]) / 2
                        >= (baseline.metrics["buy_da"] + baseline.metrics["sell_da"]) / 2 + 0.01
                    ),
                    "mcnemar_da_p_lt_0_10": p_value < 0.10,
                    "mcnemar_hit_p_lt_0_10": hit_p_value < 0.10,
                    "seed_std_lt_0_03": float(artifact_payload.get("seed_std", 1.0)) < 0.03,
                    "val_and_test_combined_delta_same_sign": val_delta * test_delta > 0,
                    "val_and_test_hit_delta_same_sign": val_hit_delta * test_hit_delta > 0,
                },
                "val_combined_delta": val_delta,
                "test_combined_delta": test_delta,
                "val_hit_delta": val_hit_delta,
                "test_hit_delta": test_hit_delta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
