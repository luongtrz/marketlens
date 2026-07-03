from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stockmem.scripts.evaluate_learned_strict_test import (
    HeadConfig,
    LearnedRow,
    _configured_head_signal,
    _fixed_score,
    _load_knn_weights,
    _load_rows,
    _matured_pool,
)
from stockmem.scripts.ndjson_eval_common import actual_signal, summarize_predictions

HORIZONS = ("1d", "3d", "7d", "15d", "30d")


@dataclass(frozen=True)
class CandidateResult:
    score: float
    head: HeadConfig
    metrics: dict[str, Any]


def _class_da(metrics: Any, label: str) -> float:
    count = metrics.actual_counts.get(label, 0)
    if count <= 0:
        return 0.0
    return metrics.confusion[label][label] / count


def _evaluate_cached(
    name: str,
    cached: list[tuple[LearnedRow, list[LearnedRow]]],
    *,
    head: HeadConfig,
    label_threshold: float,
) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    for query, ranked in cached:
        selected = ranked[: head.k]
        predicted, confidence = _configured_head_signal(selected, head=head)
        actual = actual_signal(query.future_returns.get("7d"), label_threshold)
        top5_same = any(
            actual_signal(candidate.future_returns.get("7d"), label_threshold) == actual
            for candidate in selected[:5]
        )
        rows_out.append(
            {
                "date": query.date.isoformat(),
                "model": name,
                "predicted_signal": predicted,
                "actual_signal": actual,
                "actual_return_7d": query.future_returns.get("7d"),
                "confidence": confidence,
                "top5_same_sign": top5_same,
            }
        )
    metrics = summarize_predictions(name, rows_out, label_threshold=label_threshold)
    buy_da = _class_da(metrics, "BUY")
    hold_da = _class_da(metrics, "HOLD")
    sell_da = _class_da(metrics, "SELL")
    return {
        "rows": rows_out,
        "summary": {
            **asdict(metrics),
            "buy_da": buy_da,
            "hold_da": hold_da,
            "sell_da": sell_da,
        },
    }


def _objective(summary: dict[str, Any], *, min_coverage: float) -> float:
    coverage = float(summary["coverage"])
    sell_da = float(summary["sell_da"])
    buy_da = float(summary["buy_da"])
    hold_da = float(summary["hold_da"])
    overall = float(summary["overall_acc"])
    active = float(summary["active_acc"])
    sell_rate = float(summary["sell_rate"])
    actual_sell_rate = float(summary["actual_counts"].get("SELL", 0)) / max(float(summary["n"]), 1.0)

    coverage_penalty = max(0.0, min_coverage - coverage) * 0.50
    sell_rate_penalty = abs(sell_rate - actual_sell_rate) * 0.08
    hold_penalty = max(0.0, 0.08 - hold_da) * 0.05
    return (
        0.25 * overall
        + 0.20 * active
        + 0.30 * sell_da
        + 0.15 * buy_da
        + 0.10 * coverage
        - coverage_penalty
        - sell_rate_penalty
        - hold_penalty
    )


def _normalise(weights: np.ndarray) -> dict[str, float]:
    weights = np.maximum(weights, 1e-9)
    weights = weights / weights.sum()
    return {horizon: round(float(value), 6) for horizon, value in zip(HORIZONS, weights)}


def _candidate_weights(seed: int, samples: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    anchors = [
        np.asarray([0.0344, 0.1381, 0.1629, 0.3234, 0.3411]),
        np.asarray([0.0161, 0.1459, 0.4549, 0.1005, 0.2827]),
        np.asarray([0.05, 0.10, 0.25, 0.25, 0.35]),
        np.asarray([0.05, 0.15, 0.45, 0.15, 0.20]),
        np.asarray([0.10, 0.20, 0.30, 0.20, 0.20]),
    ]
    candidates = [_normalise(item) for item in anchors]
    for _ in range(samples):
        candidates.append(_normalise(rng.dirichlet(np.asarray([1.2, 1.5, 2.5, 1.6, 1.8]))))
    deduped: dict[tuple[float, ...], dict[str, float]] = {}
    for item in candidates:
        key = tuple(item[horizon] for horizon in HORIZONS)
        deduped[key] = item
    return list(deduped.values())


def _cache_fixed_rankings(
    rows: list[LearnedRow],
    *,
    split: str,
    weights: tuple[float, float, float],
    max_k: int,
) -> list[tuple[LearnedRow, list[LearnedRow]]]:
    cached: list[tuple[LearnedRow, list[LearnedRow]]] = []
    for query in rows:
        if query.split != split:
            continue
        pool = _matured_pool(rows, query)
        scored = [(_fixed_score(query, candidate, weights), candidate) for candidate in pool]
        scored.sort(key=lambda item: item[0], reverse=True)
        cached.append((query, [row for _, row in scored[:max_k]]))
    return cached


def _write_head_config(path: Path, *, result: CandidateResult, args: argparse.Namespace) -> None:
    payload = {
        "name": "fixed_knn_downside_sensitive_head_v1",
        "source": "stockmem/scripts/experimental/train_downside_sensitive_head.py",
        "retriever": "fixed_knn",
        "selection_protocol": {
            "split": "chronological",
            "train_split": "unused_for_parametric_fit",
            "validation_split": "2025-01-01_to_2025-06-23",
            "test_split": "left_untouched_for_downstream_evaluation",
            "objective": "cost_sensitive_validation_score",
            "min_coverage": args.min_coverage,
            "seed": args.seed,
            "random_weight_samples": args.weight_samples,
        },
        "head": {
            "k": result.head.k,
            "buy_threshold": result.head.buy_threshold,
            "sell_threshold": result.head.sell_threshold,
            "return_weights": result.head.return_weights,
        },
        "validation_summary": result.metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_markdown(path: Path, *, best: CandidateResult, top: list[CandidateResult], args: argparse.Namespace) -> None:
    lines = [
        "# Downside-Sensitive Fixed-kNN Head Training",
        "",
        f"- Data: `{args.data}`",
        f"- Retriever: fixed kNN using `{args.weights}`",
        "- Selection split: validation only (`2025-01-01` to `2025-06-23`)",
        "- Test split is not used by this trainer.",
        f"- Label threshold: `±{args.label_threshold:.2f}%` on `future_return_7d`",
        "",
        "## Selected Head",
        "",
        f"- k: `{best.head.k}`",
        f"- buy_threshold: `{best.head.buy_threshold:.2f}`",
        f"- sell_threshold: `{best.head.sell_threshold:.2f}`",
        f"- return_weights: `{best.head.return_weights}`",
        f"- validation objective: `{best.score:.6f}`",
        "",
        "## Validation Metrics",
        "",
        "| Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | BUY% | HOLD% | SELL% |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = best.metrics
    lines.append(
        f"| {summary['overall_acc']:.4f} | {summary['active_acc']:.4f} | {summary['coverage']:.4f} | "
        f"{summary['buy_da']:.4f} | {summary['hold_da']:.4f} | {summary['sell_da']:.4f} | "
        f"{summary['buy_rate']:.4f} | {summary['hold_rate']:.4f} | {summary['sell_rate']:.4f} |"
    )
    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| Rank | Score | k | Buy Th | Sell Th | Overall | Active | Coverage | BUY DA | SELL DA |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, item in enumerate(top, start=1):
        summary = item.metrics
        lines.append(
            f"| {rank} | {item.score:.6f} | {item.head.k} | {item.head.buy_threshold:.2f} | "
            f"{item.head.sell_threshold:.2f} | {summary['overall_acc']:.4f} | "
            f"{summary['active_acc']:.4f} | {summary['coverage']:.4f} | "
            f"{summary['buy_da']:.4f} | {summary['sell_da']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--out-dir", default="artifacts/retrained_heads/downside_sensitive_fixed_knn")
    parser.add_argument("--label-threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--weight-samples", type=int, default=300)
    parser.add_argument("--min-coverage", type=float, default=0.65)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(Path(args.data))
    weights = _load_knn_weights(Path(args.weights))
    k_values = [3, 5, 7, 10]
    max_k = max(k_values)
    val_cache = _cache_fixed_rankings(rows, split="val", weights=weights, max_k=max_k)

    buy_thresholds = np.round(np.arange(0.8, 2.41, 0.2), 2)
    sell_thresholds = np.round(np.arange(0.4, 2.21, 0.2), 2)
    weight_candidates = _candidate_weights(args.seed, args.weight_samples)

    candidates: list[CandidateResult] = []
    checked = 0
    for k in k_values:
        for return_weights in weight_candidates:
            for buy_threshold in buy_thresholds:
                for sell_threshold in sell_thresholds:
                    head = HeadConfig(
                        name="fixed_knn_downside_sensitive_head_candidate",
                        retriever="fixed_knn",
                        k=k,
                        buy_threshold=float(buy_threshold),
                        sell_threshold=float(sell_threshold),
                        return_weights=return_weights,
                    )
                    result = _evaluate_cached(
                        "fixed_knn_downside_sensitive_head_candidate",
                        val_cache,
                        head=head,
                        label_threshold=args.label_threshold,
                    )
                    summary = result["summary"]
                    score = _objective(summary, min_coverage=args.min_coverage)
                    candidates.append(CandidateResult(score=score, head=head, metrics=summary))
                    checked += 1
                    if checked % 1000 == 0:
                        print(f"checked={checked} best={max(item.score for item in candidates):.6f}", flush=True)

    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    top = candidates[:20]
    _write_head_config(out_dir / "knn_head.fixed_knn_downside_sensitive.json", result=best, args=args)
    (out_dir / "top_candidates.json").write_text(
        json.dumps(
            [
                {
                    "rank": rank,
                    "score": item.score,
                    "head": asdict(item.head),
                    "validation_summary": item.metrics,
                }
                for rank, item in enumerate(top, start=1)
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "summary.md", best=best, top=top, args=args)
    print(f"wrote selected head to {out_dir / 'knn_head.fixed_knn_downside_sensitive.json'}")
    print(f"validation score={best.score:.6f} metrics={best.metrics}")


if __name__ == "__main__":
    main()
