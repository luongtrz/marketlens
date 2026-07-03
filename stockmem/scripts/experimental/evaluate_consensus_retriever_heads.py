from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from stockmem.scripts.experimental.train_majority_consensus_retriever import (
    ConsensusConfig,
    QueryCache,
    _fixed_scores,
    _learned_scores,
    _load_rows,
    _matured_pool,
    _minmax,
    _rank_scores,
    _regime_scores,
    _top_indices,
)
from stockmem.scripts.ndjson_eval_common import actual_signal, load_knn_weights, summarize_predictions
from stockmem.src.search.learned_metric import LearnedDiagonalMetric

HORIZONS = ("1d", "3d", "7d", "15d", "30d")


@dataclass(frozen=True)
class RankedQuery:
    date: str
    actual_return_7d: float
    ranked_records: list[Any]
    same_count_at_10: int


@dataclass(frozen=True)
class HeadSpec:
    name: str
    head: Callable[[list[Any]], tuple[str, float]]


def _class_da(metrics: Any, label: str) -> float:
    count = metrics.actual_counts.get(label, 0)
    if count <= 0:
        return 0.0
    return metrics.confusion[label][label] / count


def _load_config(path: Path) -> ConsensusConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("config", payload)
    return ConsensusConfig(
        w_fixed=float(source["w_fixed"]),
        w_learned=float(source["w_learned"]),
        w_recency=float(source["w_recency"]),
        w_regime=float(source["w_regime"]),
        recency_half_life_days=float(source["recency_half_life_days"]),
    )


def _weighted_return(record: Any, weights: dict[str, float]) -> float | None:
    total_w = total_v = 0.0
    for horizon, weight in weights.items():
        value = getattr(record, f"future_return_{horizon}", None)
        if value is None:
            continue
        total_w += float(weight)
        total_v += float(value) * float(weight)
    if total_w <= 0:
        return None
    return total_v / total_w


def _mean_return_head(name: str, *, weights: dict[str, float], buy_threshold: float, sell_threshold: float) -> HeadSpec:
    def head(records: list[Any]) -> tuple[str, float]:
        values = [value for record in records if (value := _weighted_return(record, weights)) is not None]
        if not values:
            return "HOLD", 0.5
        avg = float(np.mean(values))
        if avg > buy_threshold:
            return "BUY", round(min(0.55 + min((avg - buy_threshold) / 15.0, 0.35), 0.95), 3)
        if avg < -sell_threshold:
            return "SELL", round(min(0.55 + min((abs(avg) - sell_threshold) / 15.0, 0.35), 0.95), 3)
        return "HOLD", 0.5

    return HeadSpec(name=name, head=head)


def _median_d7_head(name: str, *, threshold: float) -> HeadSpec:
    def head(records: list[Any]) -> tuple[str, float]:
        values = [float(record.future_return_7d) for record in records if record.future_return_7d is not None]
        if not values:
            return "HOLD", 0.5
        median = float(np.median(values))
        if median > threshold:
            return "BUY", round(min(0.55 + min((median - threshold) / 15.0, 0.35), 0.95), 3)
        if median < -threshold:
            return "SELL", round(min(0.55 + min((abs(median) - threshold) / 15.0, 0.35), 0.95), 3)
        return "HOLD", 0.5

    return HeadSpec(name=name, head=head)


def _count_vote_head(name: str, *, buy_votes: int, sell_votes: int, label_threshold: float) -> HeadSpec:
    def head(records: list[Any]) -> tuple[str, float]:
        counts = Counter(actual_signal(record.future_return_7d, label_threshold) for record in records)
        if counts["SELL"] >= sell_votes and counts["SELL"] >= counts["BUY"]:
            return "SELL", min(0.95, round(0.50 + counts["SELL"] / (2.0 * max(len(records), 1)), 3))
        if counts["BUY"] >= buy_votes and counts["BUY"] > counts["SELL"]:
            return "BUY", min(0.95, round(0.50 + counts["BUY"] / (2.0 * max(len(records), 1)), 3))
        return "HOLD", 0.5

    return HeadSpec(name=name, head=head)


def _candidate_heads(label_threshold: float) -> list[HeadSpec]:
    fixed_weights = {"1d": 0.0344, "3d": 0.1381, "7d": 0.1629, "15d": 0.3234, "30d": 0.3411}
    learned_weights = {"1d": 0.0161, "3d": 0.1459, "7d": 0.4549, "15d": 0.1005, "30d": 0.2827}
    d7_only = {"7d": 1.0}
    heads: list[HeadSpec] = []
    for weight_name, weights in (
        ("fixed_weights", fixed_weights),
        ("learned_weights", learned_weights),
        ("d7_only", d7_only),
    ):
        for buy_threshold in np.arange(0.5, 2.55, 0.25):
            for sell_threshold in np.arange(0.5, 2.55, 0.25):
                heads.append(
                    _mean_return_head(
                        f"mean_{weight_name}_buy{buy_threshold:.2f}_sell{sell_threshold:.2f}",
                        weights=weights,
                        buy_threshold=float(buy_threshold),
                        sell_threshold=float(sell_threshold),
                    )
                )
    for threshold in np.arange(0.5, 2.55, 0.25):
        heads.append(_median_d7_head(f"median_d7_th{threshold:.2f}", threshold=float(threshold)))
    for buy_votes in range(3, 7):
        for sell_votes in range(3, 7):
            heads.append(
                _count_vote_head(
                    f"count_vote_buy{buy_votes}_sell{sell_votes}",
                    buy_votes=buy_votes,
                    sell_votes=sell_votes,
                    label_threshold=label_threshold,
                )
            )
    return heads


def _build_ranked_queries(
    *,
    rows: list[Any],
    split: str,
    config: ConsensusConfig,
    fixed_weights: tuple[float, float, float],
    learned_metric: LearnedDiagonalMetric,
    top_k: int,
    label_threshold: float,
) -> list[RankedQuery]:
    out: list[RankedQuery] = []
    for index, query in enumerate([row for row in rows if row.split == split], start=1):
        pool = _matured_pool(rows, query)
        cache = QueryCache(
            query_date=query.date,
            actual_id=query.label_id,
            candidate_labels=np.asarray([candidate.label_id for candidate in pool], dtype=np.int8),
            fixed=_minmax(_fixed_scores(query, pool, fixed_weights)),
            learned=_minmax(_learned_scores(query, pool, learned_metric)),
            age_days=np.asarray([(query.date - candidate.date).days for candidate in pool], dtype=np.float64),
            regime=_regime_scores(query, pool),
        )
        if cache.candidate_labels.size == 0:
            ranked = []
            same_count = 0
        else:
            scores = _rank_scores(cache, config)
            top = _top_indices(scores, min(top_k, scores.size))
            ranked = [pool[int(i)].record for i in top]
            same_count = int(np.sum(cache.candidate_labels[top] == cache.actual_id))
        out.append(
            RankedQuery(
                date=query.date.isoformat(),
                actual_return_7d=float(query.record.future_return_7d),
                ranked_records=ranked,
                same_count_at_10=same_count,
            )
        )
        if index % 500 == 0:
            print(f"ranked {split} queries={index}", flush=True)
    return out


def _evaluate_head(name: str, ranked_queries: list[RankedQuery], head: HeadSpec, *, label_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for query in ranked_queries:
        predicted, confidence = head.head(query.ranked_records)
        actual = actual_signal(query.actual_return_7d, label_threshold)
        rows.append(
            {
                "date": query.date,
                "model": name,
                "head": head.name,
                "predicted_signal": predicted,
                "actual_signal": actual,
                "actual_return_7d": query.actual_return_7d,
                "confidence": confidence,
                "same_count_at_10": query.same_count_at_10,
                "top10_majority_same": query.same_count_at_10 >= 5,
            }
        )
    metrics = summarize_predictions(name, rows, label_threshold=label_threshold)
    summary = {
        **asdict(metrics),
        "buy_da": _class_da(metrics, "BUY"),
        "hold_da": _class_da(metrics, "HOLD"),
        "sell_da": _class_da(metrics, "SELL"),
        "mean_same_at_10": float(np.mean([row["same_count_at_10"] for row in rows])) if rows else 0.0,
        "majority_same_at_10": float(np.mean([row["top10_majority_same"] for row in rows])) if rows else 0.0,
    }
    return summary, rows


def _selection_score(summary: dict[str, Any]) -> float:
    return (
        0.30 * float(summary["overall_acc"])
        + 0.20 * float(summary["active_acc"])
        + 0.15 * float(summary["coverage"])
        + 0.15 * float(summary["sell_da"])
        + 0.10 * float(summary["buy_da"])
        + 0.05 * float(summary["hold_da"])
        + 0.05 * float(summary["majority_same_at_10"])
    )


def _fmt_metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Consensus Retriever Decision Head Evaluation",
        "",
        f"- Data: `{payload['data_path']}`",
        f"- Retriever config: `{payload['config_path']}`",
        f"- Top-k: `{payload['top_k']}`",
        f"- Label threshold: `±{payload['label_threshold']:.2f}%`",
        "",
        "## Selected Head",
        "",
        f"- Head: `{payload['selected_head']}`",
        f"- Validation score: `{payload['selected_validation_score']:.4f}`",
        "",
        "## Comparison",
        "",
        "| Model | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 | Mean Same@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["comparison"]:
        metrics = item["metrics"]
        lines.append(
            f"| `{item['model']}` | {item['split']} | {metrics['n']} | "
            f"{metrics['overall_acc']:.4f} | {metrics['active_acc']:.4f} | {metrics['coverage']:.4f} | "
            f"{metrics['buy_da']:.4f} | {metrics['hold_da']:.4f} | {metrics['sell_da']:.4f} | "
            f"{_fmt_metric(metrics, 'majority_same_at_10')} | {_fmt_metric(metrics, 'mean_same_at_10')} |"
        )
    lines.extend(
        [
            "",
            "## Top Validation Heads",
            "",
            "| Rank | Head | Score | Val Overall | Val Active | Val Coverage | Val BUY DA | Val SELL DA | Test Overall | Test Active | Test Coverage | Test BUY DA | Test SELL DA |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["top_validation_heads"]:
        metrics = item["metrics"]
        test_metrics = item["test_metrics"]
        lines.append(
            f"| {item['rank']} | `{item['head']}` | {item['score']:.4f} | "
            f"{metrics['overall_acc']:.4f} | {metrics['active_acc']:.4f} | {metrics['coverage']:.4f} | "
            f"{metrics['buy_da']:.4f} | {metrics['sell_da']:.4f} | "
            f"{test_metrics['overall_acc']:.4f} | {test_metrics['active_acc']:.4f} | {test_metrics['coverage']:.4f} | "
            f"{test_metrics['buy_da']:.4f} | {test_metrics['sell_da']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--config", default="stockmem/config/majority_consensus_retriever.learned_recency_50_50.json")
    parser.add_argument("--out-dir", default="artifacts/consensus_retriever_heads")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label-threshold", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(Path(args.data), label_threshold=args.label_threshold)
    fixed_weights = load_knn_weights(Path(args.weights))
    learned_metric = LearnedDiagonalMetric.load(args.artifact)
    config = _load_config(Path(args.config))

    val_ranked = _build_ranked_queries(
        rows=rows,
        split="val",
        config=config,
        fixed_weights=fixed_weights,
        learned_metric=learned_metric,
        top_k=args.top_k,
        label_threshold=args.label_threshold,
    )
    test_ranked = _build_ranked_queries(
        rows=rows,
        split="test",
        config=config,
        fixed_weights=fixed_weights,
        learned_metric=learned_metric,
        top_k=args.top_k,
        label_threshold=args.label_threshold,
    )

    scored: list[tuple[float, HeadSpec, dict[str, Any]]] = []
    for head in _candidate_heads(args.label_threshold):
        metrics, _rows = _evaluate_head(
            f"consensus_{head.name}",
            val_ranked,
            head,
            label_threshold=args.label_threshold,
        )
        scored.append((_selection_score(metrics), head, metrics))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_head, best_val = scored[0]
    best_test, best_rows = _evaluate_head(
        f"learned_recency_50_50_{best_head.name}",
        test_ranked,
        best_head,
        label_threshold=args.label_threshold,
    )

    comparison = [
        {
            "model": f"learned_recency_50_50 + {best_head.name}",
            "split": "val",
            "metrics": best_val,
        },
        {
            "model": f"learned_recency_50_50 + {best_head.name}",
            "split": "test",
            "metrics": best_test,
        },
    ]

    old_summary_path = Path("artifacts/audit_runs/stockmem_audit_20260702_232452/learned_strict_test/summary.json")
    if old_summary_path.exists():
        old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
        for model_name in ("fixed_knn_rolling_stable", "fixed_retriever_learned_head", "learned_finbert_rolling_stable"):
            model = next(item for item in old_summary["models"] if item["name"] == model_name)
            confusion = model["confusion"]
            metrics = {
                **model,
                "buy_da": confusion["BUY"]["BUY"] / model["actual_counts"]["BUY"],
                "hold_da": confusion["HOLD"]["HOLD"] / model["actual_counts"]["HOLD"],
                "sell_da": confusion["SELL"]["SELL"] / model["actual_counts"]["SELL"],
                "majority_same_at_10": None,
                "mean_same_at_10": None,
            }
            comparison.append({"model": model_name, "split": "test_old_strict", "metrics": metrics})

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in best_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "data_path": args.data,
        "weights_path": args.weights,
        "artifact_path": args.artifact,
        "config_path": args.config,
        "top_k": args.top_k,
        "label_threshold": args.label_threshold,
        "selected_head": best_head.name,
        "selected_validation_score": best_score,
        "comparison": comparison,
        "top_validation_heads": [],
    }
    for rank, (score, head, metrics) in enumerate(scored[:20], start=1):
        test_metrics, _ = _evaluate_head(
            f"learned_recency_50_50_{head.name}",
            test_ranked,
            head,
            label_threshold=args.label_threshold,
        )
        payload["top_validation_heads"].append(
            {
                "rank": rank,
                "head": head.name,
                "score": score,
                "metrics": metrics,
                "test_metrics": test_metrics,
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(out_dir / "summary.md", payload)
    print(f"wrote consensus retriever head evaluation to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
