from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from stockmem.scripts.evaluate_learned_strict_test import (
    LearnedRow,
    _fixed_score,
    _load_knn_weights,
    _load_rows,
    _matured_pool,
)
from stockmem.scripts.ndjson_eval_common import actual_signal, summarize_predictions
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


def _class_da(metrics: Any, label: str) -> float:
    count = metrics.actual_counts.get(label, 0)
    if count <= 0:
        return 0.0
    return metrics.confusion[label][label] / count


def _rank_weight(rank: int) -> float:
    return 1.0 / math.log2(rank + 2.0)


def _candidate_label(row: LearnedRow, label_threshold: float) -> str:
    return actual_signal(row.future_returns.get("7d"), label_threshold)


def _weighted_return(row: LearnedRow, weights: dict[str, float]) -> float | None:
    total_w = total_v = 0.0
    for horizon, weight in weights.items():
        value = row.future_returns.get(horizon)
        if value is not None:
            total_w += weight
            total_v += float(value) * weight
    if total_w <= 0:
        return None
    return total_v / total_w


def _make_return_average_head(
    *,
    threshold: float,
    return_weights: dict[str, float],
) -> Callable[[list[LearnedRow], float], tuple[str, float]]:
    def head(ranked: list[LearnedRow], _label_threshold: float) -> tuple[str, float]:
        values = [value for row in ranked if (value := _weighted_return(row, return_weights)) is not None]
        if not values:
            return "HOLD", 0.5
        avg = float(np.mean(values))
        if avg > threshold:
            return "BUY", round(min(0.55 + min((avg - threshold) / 15.0, 0.35), 0.95), 3)
        if avg < -threshold:
            return "SELL", round(min(0.55 + min((abs(avg) - threshold) / 15.0, 0.35), 0.95), 3)
        return "HOLD", 0.5

    return head


def _make_median_return_head(*, threshold: float) -> Callable[[list[LearnedRow], float], tuple[str, float]]:
    def head(ranked: list[LearnedRow], _label_threshold: float) -> tuple[str, float]:
        values = [float(row.future_returns["7d"]) for row in ranked if row.future_returns.get("7d") is not None]
        if not values:
            return "HOLD", 0.5
        median = float(np.median(values))
        if median > threshold:
            return "BUY", round(min(0.55 + min((median - threshold) / 15.0, 0.35), 0.95), 3)
        if median < -threshold:
            return "SELL", round(min(0.55 + min((abs(median) - threshold) / 15.0, 0.35), 0.95), 3)
        return "HOLD", 0.5

    return head


def _make_count_vote_head(
    *,
    buy_votes: int,
    sell_votes: int,
) -> Callable[[list[LearnedRow], float], tuple[str, float]]:
    def head(ranked: list[LearnedRow], label_threshold: float) -> tuple[str, float]:
        counts = Counter(_candidate_label(row, label_threshold) for row in ranked)
        if counts["SELL"] >= sell_votes and counts["SELL"] >= counts["BUY"]:
            return "SELL", min(0.95, round(0.50 + counts["SELL"] / (2.0 * max(len(ranked), 1)), 3))
        if counts["BUY"] >= buy_votes and counts["BUY"] > counts["SELL"]:
            return "BUY", min(0.95, round(0.50 + counts["BUY"] / (2.0 * max(len(ranked), 1)), 3))
        return "HOLD", 0.5

    return head


def _make_rank_weighted_vote_head(
    *,
    buy_weight: float,
    sell_weight: float,
) -> Callable[[list[LearnedRow], float], tuple[str, float]]:
    def head(ranked: list[LearnedRow], label_threshold: float) -> tuple[str, float]:
        scores = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
        for rank, row in enumerate(ranked):
            scores[_candidate_label(row, label_threshold)] += _rank_weight(rank)
        total = sum(scores.values()) or 1.0
        if scores["SELL"] / total >= sell_weight and scores["SELL"] >= scores["BUY"]:
            return "SELL", round(0.50 + 0.45 * scores["SELL"] / total, 3)
        if scores["BUY"] / total >= buy_weight and scores["BUY"] > scores["SELL"]:
            return "BUY", round(0.50 + 0.45 * scores["BUY"] / total, 3)
        return "HOLD", 0.5

    return head


def _evidence_summary(cached: list[tuple[LearnedRow, list[LearnedRow]]], *, top_k: int, label_threshold: float) -> dict[str, Any]:
    dist: Counter[int] = Counter()
    by_actual: dict[str, Counter[int]] = defaultdict(Counter)
    first_ranks: list[int] = []
    for query, ranked_all in cached:
        ranked = ranked_all[:top_k]
        actual = actual_signal(query.future_returns.get("7d"), label_threshold)
        labels = [_candidate_label(row, label_threshold) for row in ranked]
        same_count = sum(label == actual for label in labels)
        dist[same_count] += 1
        by_actual[actual][same_count] += 1
        for rank, label in enumerate(labels, start=1):
            if label == actual:
                first_ranks.append(rank)
                break

    n = sum(dist.values())
    majority_threshold = (top_k + 1) // 2

    def summarise_counter(counter: Counter[int]) -> dict[str, Any]:
        total = sum(counter.values())
        if total == 0:
            return {"n": 0}
        return {
            "n": total,
            "distribution": dict(sorted(counter.items())),
            "hit_at_k": sum(v for k, v in counter.items() if k >= 1) / total,
            "majority_same_at_k": sum(v for k, v in counter.items() if k >= majority_threshold) / total,
            "mean_same_count": sum(k * v for k, v in counter.items()) / total,
            "weighted_same_score": sum((k / top_k) * v for k, v in counter.items()) / total,
        }

    return {
        "top_k": top_k,
        "majority_threshold": majority_threshold,
        **summarise_counter(dist),
        "mean_first_correct_rank_when_hit": (sum(first_ranks) / len(first_ranks)) if first_ranks else None,
        "by_actual": {label: summarise_counter(counter) for label, counter in sorted(by_actual.items())},
    }


def _evaluate_head(
    name: str,
    cached: list[tuple[LearnedRow, list[LearnedRow]]],
    *,
    top_k: int,
    label_threshold: float,
    head: Callable[[list[LearnedRow], float], tuple[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_out: list[dict[str, Any]] = []
    for query, ranked_all in cached:
        ranked = ranked_all[:top_k]
        predicted, confidence = head(ranked, label_threshold)
        actual = actual_signal(query.future_returns.get("7d"), label_threshold)
        rows_out.append(
            {
                "date": query.date.isoformat(),
                "model": name,
                "predicted_signal": predicted,
                "actual_signal": actual,
                "actual_return_7d": query.future_returns.get("7d"),
                "confidence": confidence,
                "same_count_at_k": sum(_candidate_label(row, label_threshold) == actual for row in ranked),
                "k": top_k,
            }
        )
    metrics = summarize_predictions(name, rows_out, label_threshold=label_threshold)
    summary = {
        **asdict(metrics),
        "buy_da": _class_da(metrics, "BUY"),
        "hold_da": _class_da(metrics, "HOLD"),
        "sell_da": _class_da(metrics, "SELL"),
    }
    return summary, rows_out


def _objective(summary: dict[str, Any]) -> float:
    return (
        0.25 * float(summary["overall_acc"])
        + 0.20 * float(summary["active_acc"])
        + 0.20 * float(summary["coverage"])
        + 0.20 * float(summary["sell_da"])
        + 0.10 * float(summary["buy_da"])
        + 0.05 * float(summary["hold_da"])
    )


def _rank_cache(
    rows: list[LearnedRow],
    *,
    retriever: str,
    split: str,
    max_k: int,
    fixed_weights: tuple[float, float, float],
    learned_metric: LearnedDiagonalMetric,
) -> list[tuple[LearnedRow, list[LearnedRow]]]:
    cached: list[tuple[LearnedRow, list[LearnedRow]]] = []
    for query in rows:
        if query.split != split:
            continue
        pool = _matured_pool(rows, query)
        if retriever == "fixed_knn":
            scored = [(_fixed_score(query, candidate, fixed_weights), candidate) for candidate in pool]
        elif retriever == "learned_finbert":
            scored = [(learned_metric.score(query.blocks, candidate.blocks), candidate) for candidate in pool]
        else:
            raise ValueError(f"Unknown retriever: {retriever}")
        scored.sort(key=lambda item: item[0], reverse=True)
        cached.append((query, [row for _, row in scored[:max_k]]))
    return cached


def _candidate_heads() -> list[tuple[str, Callable[[list[LearnedRow], float], tuple[str, float]]]]:
    heads: list[tuple[str, Callable[[list[LearnedRow], float], tuple[str, float]]]] = []
    return_weight_sets = {
        "old_fixed_weights": {"1d": 0.0344, "3d": 0.1381, "7d": 0.1629, "15d": 0.3234, "30d": 0.3411},
        "old_learned_weights": {"1d": 0.0161, "3d": 0.1459, "7d": 0.4549, "15d": 0.1005, "30d": 0.2827},
        "d7_only": {"7d": 1.0},
    }
    for weight_name, weights in return_weight_sets.items():
        for threshold in (0.5, 1.0, 1.5, 2.0):
            heads.append((f"mean_return_{weight_name}_th{threshold:.1f}", _make_return_average_head(threshold=threshold, return_weights=weights)))
    for threshold in (0.5, 1.0, 1.5, 2.0):
        heads.append((f"median_return_th{threshold:.1f}", _make_median_return_head(threshold=threshold)))
    for buy_votes in range(3, 7):
        for sell_votes in range(3, 7):
            heads.append((f"count_vote_buy{buy_votes}_sell{sell_votes}", _make_count_vote_head(buy_votes=buy_votes, sell_votes=sell_votes)))
    for buy_weight in (0.30, 0.35, 0.40, 0.45, 0.50):
        for sell_weight in (0.30, 0.35, 0.40, 0.45, 0.50):
            heads.append((f"rank_weighted_vote_buy{buy_weight:.2f}_sell{sell_weight:.2f}", _make_rank_weighted_vote_head(buy_weight=buy_weight, sell_weight=sell_weight)))
    return heads


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Top-k Consensus Head Experiment",
        "",
        f"- Data: `{payload['data_path']}`",
        f"- Label threshold: `±{payload['label_threshold']:.2f}%` on `future_return_7d`",
        f"- Top-k: `{payload['top_k']}`",
        "- Head selection split: validation",
        "- Final metrics split: held-out test",
        "",
        "## Evidence Density",
        "",
        "| Retriever | k | Hit@k >=1 | Majority same | Mean same count | Weighted same |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["evidence_test"]:
        summary = item["summary"]
        lines.append(
            f"| {item['retriever']} | {summary['top_k']} | {summary['hit_at_k']:.4f} | "
            f"{summary['majority_same_at_k']:.4f} | {summary['mean_same_count']:.4f} | "
            f"{summary['weighted_same_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Heads",
            "",
            "| Retriever | Head | Val score | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | BUY% | HOLD% | SELL% |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["selected_test"]:
        metrics = item["test_metrics"]
        lines.append(
            f"| {item['retriever']} | `{item['head_name']}` | {item['validation_score']:.4f} | "
            f"{metrics['overall_acc']:.4f} | {metrics['active_acc']:.4f} | {metrics['coverage']:.4f} | "
            f"{metrics['buy_da']:.4f} | {metrics['hold_da']:.4f} | {metrics['sell_da']:.4f} | "
            f"{metrics['buy_rate']:.4f} | {metrics['hold_rate']:.4f} | {metrics['sell_rate']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--out-dir", default="artifacts/topk_consensus_heads")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label-threshold", type=float, default=2.0)
    args = parser.parse_args()

    rows = _load_rows(Path(args.data))
    fixed_weights = _load_knn_weights(Path(args.weights))
    learned_metric = LearnedDiagonalMetric.load(args.artifact)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "data_path": args.data,
        "weights_path": args.weights,
        "artifact_path": args.artifact,
        "top_k": args.top_k,
        "label_threshold": args.label_threshold,
        "evidence_test": [],
        "selected_test": [],
    }

    heads = _candidate_heads()
    for retriever in ("fixed_knn", "learned_finbert"):
        val_cache = _rank_cache(
            rows,
            retriever=retriever,
            split="val",
            max_k=args.top_k,
            fixed_weights=fixed_weights,
            learned_metric=learned_metric,
        )
        test_cache = _rank_cache(
            rows,
            retriever=retriever,
            split="test",
            max_k=args.top_k,
            fixed_weights=fixed_weights,
            learned_metric=learned_metric,
        )
        payload["evidence_test"].append(
            {
                "retriever": retriever,
                "summary": _evidence_summary(test_cache, top_k=args.top_k, label_threshold=args.label_threshold),
            }
        )
        scored_heads = []
        for head_name, head in heads:
            val_metrics, _ = _evaluate_head(
                head_name,
                val_cache,
                top_k=args.top_k,
                label_threshold=args.label_threshold,
                head=head,
            )
            scored_heads.append((_objective(val_metrics), head_name, head, val_metrics))
        scored_heads.sort(key=lambda item: item[0], reverse=True)
        best_score, best_name, best_head, best_val = scored_heads[0]
        test_metrics, test_rows = _evaluate_head(
            f"{retriever}_{best_name}",
            test_cache,
            top_k=args.top_k,
            label_threshold=args.label_threshold,
            head=best_head,
        )
        with (out_dir / f"{retriever}_{best_name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in test_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload["selected_test"].append(
            {
                "retriever": retriever,
                "head_name": best_name,
                "validation_score": best_score,
                "validation_metrics": best_val,
                "test_metrics": test_metrics,
                "top_candidates": [
                    {
                        "rank": rank,
                        "score": score,
                        "head_name": head_name,
                        "validation_metrics": val_metrics,
                    }
                    for rank, (score, head_name, _head, val_metrics) in enumerate(scored_heads[:10], start=1)
                ],
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(out_dir / "summary.md", payload)
    print(f"wrote top-k consensus experiment to {out_dir}")


if __name__ == "__main__":
    main()
