from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from stockmem.scripts.cem_dataset import LabeledRow, label_rows, matured_pool
from stockmem.scripts.optimize_weights import load_rows, validate_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


DEFAULT_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)
CLASSES = ("UP", "HOLD", "DOWN")


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    n: int
    overall_accuracy: float
    action_accuracy: float
    coverage: float
    confusion: dict[str, dict[str, int]]
    actual_counts: dict[str, int]
    predicted_counts: dict[str, int]


def _fixed_score(
    query: LabeledRow,
    candidate: LabeledRow,
    weights: tuple[float, float, float],
) -> float:
    return (
        weights[0] * float(np.dot(query.row.factor_vec, candidate.row.factor_vec))
        + weights[1] * float(np.dot(query.row.indicator_vec, candidate.row.indicator_vec))
        + weights[2] * float(np.dot(query.row.price_vec, candidate.row.price_vec))
    )


def _to_class(value: float, buy_threshold: float, sell_threshold: float) -> str:
    if value > buy_threshold:
        return "UP"
    if value < -sell_threshold:
        return "DOWN"
    return "HOLD"


def _evaluate_model(
    labeled: Sequence[LabeledRow],
    score_fn: Callable[[LabeledRow, LabeledRow], float],
    *,
    split: str,
    k: int,
    buy_threshold: float,
    sell_threshold: float,
) -> ComparisonResult:
    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    actual_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    total = correct = action_total = action_correct = predicted_action = 0

    for query in labeled:
        if query.split != split:
            continue
        pool = matured_pool(labeled, query, guard=True)
        if not pool:
            continue
        ranked = sorted(
            ((score_fn(query, candidate), candidate) for candidate in pool),
            key=lambda item: item[0],
            reverse=True,
        )[:k]
        predicted_return = float(np.mean([row.row.future_return_7d for _, row in ranked]))
        predicted = _to_class(predicted_return, buy_threshold, sell_threshold)
        actual = _to_class(query.row.future_return_7d, buy_threshold, sell_threshold)

        confusion[actual][predicted] += 1
        actual_counts[actual] += 1
        predicted_counts[predicted] += 1
        total += 1
        correct += int(predicted == actual)
        if predicted != "HOLD":
            predicted_action += 1
            action_total += 1
            action_correct += int(predicted == actual)

    return ComparisonResult(
        name="",
        n=total,
        overall_accuracy=(correct / total) if total else 0.0,
        action_accuracy=(action_correct / action_total) if action_total else 0.0,
        coverage=(predicted_action / total) if total else 0.0,
        confusion=confusion,
        actual_counts=dict(actual_counts),
        predicted_counts=dict(predicted_counts),
    )


def _print_result(result: ComparisonResult) -> None:
    print(f"\n[{result.name}]")
    print(
        f"n={result.n} overall_acc={result.overall_accuracy:.4f} "
        f"action_acc={result.action_accuracy:.4f} coverage={result.coverage:.4f}"
    )
    print("actual_counts:", json.dumps(result.actual_counts, ensure_ascii=True, sort_keys=True))
    print(
        "predicted_counts:",
        json.dumps(result.predicted_counts, ensure_ascii=True, sort_keys=True),
    )
    print("confusion_matrix(actual -> predicted)")
    header = "actual\\pred\t" + "\t".join(CLASSES)
    print(header)
    for actual in CLASSES:
        row = [str(result.confusion[actual][pred]) for pred in CLASSES]
        print(f"{actual}\t" + "\t".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline fixed-kNN vs learned retriever on UP/HOLD/DOWN accuracy."
    )
    parser.add_argument("--data", default="stockmem/data/real_optimizer_finbert.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--buy-threshold", type=float, default=2.0)
    parser.add_argument("--sell-threshold", type=float, default=2.0)
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    validate_rows(rows)

    artifact_payload = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    metric = LearnedDiagonalMetric.from_payload(artifact_payload)
    band = str(artifact_payload.get("band", "fixed"))
    labeled = label_rows(rows, band=band, fixed_band=args.buy_threshold)

    weights = DEFAULT_WEIGHTS
    weights_path = Path(args.weights)
    if weights_path.exists():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        source = payload.get("weights", payload)
        weights = (
            float(source["w1_factor"]),
            float(source["w2_indicator"]),
            float(source["w3_price"]),
        )
    k = args.k or int(artifact_payload.get("hyperparameters", {}).get("k", 5))

    baseline = _evaluate_model(
        labeled,
        lambda q, c: _fixed_score(q, c, weights),
        split=args.split,
        k=k,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    learned = _evaluate_model(
        labeled,
        lambda q, c: metric.score(q.blocks, c.blocks),
        split=args.split,
        k=k,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    baseline = ComparisonResult(
        name="baseline_fixed_knn",
        n=baseline.n,
        overall_accuracy=baseline.overall_accuracy,
        action_accuracy=baseline.action_accuracy,
        coverage=baseline.coverage,
        confusion=baseline.confusion,
        actual_counts=baseline.actual_counts,
        predicted_counts=baseline.predicted_counts,
    )
    learned = ComparisonResult(
        name="learned_finbert",
        n=learned.n,
        overall_accuracy=learned.overall_accuracy,
        action_accuracy=learned.action_accuracy,
        coverage=learned.coverage,
        confusion=learned.confusion,
        actual_counts=learned.actual_counts,
        predicted_counts=learned.predicted_counts,
    )

    print(
        json.dumps(
            {
                "split": args.split,
                "k": k,
                "buy_threshold": args.buy_threshold,
                "sell_threshold": args.sell_threshold,
                "note": (
                    "overall_acc is strict 3-class UP/HOLD/DOWN accuracy; "
                    "action_acc only counts predicted UP or DOWN rows."
                ),
            },
            indent=2,
        )
    )
    _print_result(baseline)
    _print_result(learned)


if __name__ == "__main__":
    main()
