from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from stockmem.scripts.ndjson_eval_common import (
    PredictionMetrics,
    actual_signal,
    summarize_predictions,
)
from stockmem.src.search.learned_metric import LearnedDiagonalMetric

TRAIN_END = date(2024, 12, 24)
VAL_START = date(2025, 1, 1)
VAL_END = date(2025, 6, 23)
TEST_START = date(2025, 7, 1)
TEST_END = date(2026, 5, 1)
DEFAULT_KNN_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)


@dataclass(frozen=True)
class HeadConfig:
    name: str
    retriever: str
    k: int
    buy_threshold: float
    sell_threshold: float
    return_weights: dict[str, float]


@dataclass(frozen=True)
class LearnedRow:
    date: date
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray
    event_vec: np.ndarray
    future_returns: dict[str, float | None]
    split: str

    @property
    def blocks(self) -> tuple[np.ndarray, ...]:
        return (self.event_vec, self.factor_vec, self.indicator_vec, self.price_vec)


def _overall_correct(row: dict[str, object], *, label_threshold: float) -> bool:
    predicted = str(row["predicted_signal"])
    actual_return = float(row["actual_return_7d"] or 0.0)
    actual = actual_signal(actual_return, label_threshold)
    return predicted == actual


def _active_correct(row: dict[str, object]) -> bool | None:
    predicted = str(row["predicted_signal"])
    actual_return = float(row["actual_return_7d"] or 0.0)
    if predicted == "BUY":
        return actual_return > 0.0
    if predicted == "SELL":
        return actual_return < 0.0
    return None


def _bootstrap_metric_delta(
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    *,
    label_threshold: float,
    metric_name: str,
    samples: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    if len(left_rows) != len(right_rows):
        raise ValueError("Bootstrap inputs must have same length")
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n = len(left_rows)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        left_sample = [left_rows[i] for i in idx]
        right_sample = [right_rows[i] for i in idx]
        left_metrics = summarize_predictions("left", left_sample, label_threshold=label_threshold)
        right_metrics = summarize_predictions("right", right_sample, label_threshold=label_threshold)
        deltas.append(getattr(right_metrics, metric_name) - getattr(left_metrics, metric_name))
    arr = np.asarray(deltas, dtype=np.float64)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _mcnemar_exact(
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    *,
    label_threshold: float,
) -> dict[str, float | int]:
    left_only = right_only = 0
    for left_row, right_row in zip(left_rows, right_rows):
        left_correct = _overall_correct(left_row, label_threshold=label_threshold)
        right_correct = _overall_correct(right_row, label_threshold=label_threshold)
        if left_correct and not right_correct:
            left_only += 1
        elif right_correct and not left_correct:
            right_only += 1
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def _l2(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _assign_split(day: date) -> str:
    if day <= TRAIN_END:
        return "train"
    if VAL_START <= day <= VAL_END:
        return "val"
    if TEST_START <= day <= TEST_END:
        return "test"
    return "embargo"


def _matured_pool(rows: list[LearnedRow], query: LearnedRow, *, horizon_days: int = 7) -> list[LearnedRow]:
    cutoff = query.date
    return [
        candidate
        for candidate in rows
        if candidate.date < cutoff and candidate.date + timedelta(days=horizon_days) <= cutoff
    ]


def _load_rows(path: Path) -> list[LearnedRow]:
    rows: list[LearnedRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            payload = raw.get("payload", raw)
            factor_raw = payload.get("factor_vec") or payload.get("factor_vector") or []
            indicator_raw = payload.get("indicator_vec") or []
            price_raw = payload.get("price_vec") or []
            event_raw = payload.get("event_vec") or []
            if len(factor_raw) != 75 or len(indicator_raw) != 5 or len(price_raw) != 60:
                continue
            if len(event_raw) != 85:
                event_raw = [0.0] * 85
            day = date.fromisoformat(str(payload["date"]))
            if payload.get("future_return_7d") is None:
                continue
            rows.append(
                LearnedRow(
                    date=day,
                    factor_vec=_l2(factor_raw),
                    indicator_vec=_l2(indicator_raw),
                    price_vec=_l2(price_raw),
                    event_vec=_l2(event_raw),
                    future_returns={
                        "1d": payload.get("future_return_1d"),
                        "3d": payload.get("future_return_3d"),
                        "7d": payload.get("future_return_7d"),
                        "15d": payload.get("future_return_15d"),
                        "30d": payload.get("future_return_30d"),
                    },
                    split=_assign_split(day),
                )
            )
    rows.sort(key=lambda row: row.date)
    return rows


def _load_head_config(path: Path) -> HeadConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    head = payload["head"]
    return HeadConfig(
        name=str(payload.get("name", path.stem)),
        retriever=str(payload.get("retriever", "fixed_knn")),
        k=int(head["k"]),
        buy_threshold=float(head["buy_threshold"]),
        sell_threshold=float(head["sell_threshold"]),
        return_weights={str(k): float(v) for k, v in head["return_weights"].items()},
    )


def _load_knn_weights(path: Path) -> tuple[float, float, float]:
    if not path.exists():
        return DEFAULT_KNN_WEIGHTS
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("weights", payload)
    return (
        float(source["w1_factor"]),
        float(source["w2_indicator"]),
        float(source["w3_price"]),
    )


def _fixed_score(query: LearnedRow, candidate: LearnedRow, weights: tuple[float, float, float]) -> float:
    return (
        weights[0] * float(np.dot(query.factor_vec, candidate.factor_vec))
        + weights[1] * float(np.dot(query.indicator_vec, candidate.indicator_vec))
        + weights[2] * float(np.dot(query.price_vec, candidate.price_vec))
    )


def _configured_head_signal(similar_rows: list[LearnedRow], *, head: HeadConfig) -> tuple[str, float]:
    per_record_avgs: list[float] = []
    for row in similar_rows[: head.k]:
        total_w = total_v = 0.0
        for horizon, weight in head.return_weights.items():
            value = row.future_returns.get(horizon)
            if value is not None:
                total_v += float(value) * weight
                total_w += weight
        if total_w > 0:
            per_record_avgs.append(total_v / total_w)
    if not per_record_avgs:
        return "HOLD", 0.5
    avg = float(np.mean(per_record_avgs))
    if avg > head.buy_threshold:
        signal = "BUY"
        confidence = min(0.55 + min((avg - head.buy_threshold) / 15.0, 0.35), 0.95)
    elif avg < -head.sell_threshold:
        signal = "SELL"
        confidence = min(0.55 + min((abs(avg) - head.sell_threshold) / 15.0, 0.35), 0.95)
    else:
        signal = "HOLD"
        confidence = 0.5
    return signal, round(confidence, 3)


def _evaluate_model(
    name: str,
    rows: list[LearnedRow],
    *,
    rank_fn,
    head: HeadConfig,
    label_threshold: float,
) -> tuple[PredictionMetrics, list[dict[str, object]]]:
    out_rows: list[dict[str, object]] = []
    test_rows = [row for row in rows if row.split == "test"]
    for query in test_rows:
        pool = _matured_pool(rows, query)
        ranked = rank_fn(query, pool)[: head.k]
        predicted, confidence = _configured_head_signal(ranked, head=head)
        actual = actual_signal(query.future_returns.get("7d"), label_threshold)
        top5_same = any(
            actual_signal(candidate.future_returns.get("7d"), label_threshold) == actual
            for candidate in ranked[:5]
        )
        out_rows.append(
            {
                "date": query.date.isoformat(),
                "model": name,
                "predicted_signal": predicted,
                "actual_signal": actual,
                "actual_return_7d": query.future_returns.get("7d"),
                "confidence": confidence,
                "top5_same_sign": top5_same,
                "retrieval_count_reference": len(ranked),
            }
        )
    return summarize_predictions(name, out_rows, label_threshold=label_threshold), out_rows


def _write_markdown(
    metrics: list[PredictionMetrics],
    out_path: Path,
    *,
    data_path: Path,
    label_threshold: float,
    paired_stats: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# Strict Test: Learned Retriever vs Fixed-kNN",
        "",
        f"- Data source: `{data_path}`",
        f"- Test split: `2025-07-01` to `2026-05-01`",
        f"- Label threshold: `±{label_threshold:.2f}%` on `future_return_7d`",
        "",
        "| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate | Avg conf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        lines.append(
            "| "
            f"{item.name} | {item.n} | {item.overall_acc:.4f} | {item.active_acc:.4f} | "
            f"{item.coverage:.4f} | {item.hit_at_5_same_sign:.4f} | {item.buy_rate:.4f} | "
            f"{item.hold_rate:.4f} | {item.sell_rate:.4f} | {item.avg_confidence:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Comparison",
            "",
            "Primary pair: `fixed_knn_rolling_stable` vs `fixed_retriever_learned_head`",
            "",
            "| Metric | Delta learned-fixed | 95% bootstrap CI |",
            "| --- | ---: | ---: |",
            f"| overall_acc | {paired_stats['best_vs_fixed']['overall_acc']['mean_delta']:+.4f} | [{paired_stats['best_vs_fixed']['overall_acc']['ci_low']:+.4f}, {paired_stats['best_vs_fixed']['overall_acc']['ci_high']:+.4f}] |",
            f"| active_acc | {paired_stats['best_vs_fixed']['active_acc']['mean_delta']:+.4f} | [{paired_stats['best_vs_fixed']['active_acc']['ci_low']:+.4f}, {paired_stats['best_vs_fixed']['active_acc']['ci_high']:+.4f}] |",
            f"| coverage | {paired_stats['best_vs_fixed']['coverage']['mean_delta']:+.4f} | [{paired_stats['best_vs_fixed']['coverage']['ci_low']:+.4f}, {paired_stats['best_vs_fixed']['coverage']['ci_high']:+.4f}] |",
            f"| hit_at_5_same_sign | {paired_stats['best_vs_fixed']['hit_at_5_same_sign']['mean_delta']:+.4f} | [{paired_stats['best_vs_fixed']['hit_at_5_same_sign']['ci_low']:+.4f}, {paired_stats['best_vs_fixed']['hit_at_5_same_sign']['ci_high']:+.4f}] |",
            "",
            f"- McNemar exact test p-value: `{paired_stats['best_vs_fixed']['mcnemar']['p_value']:.6f}`",
            f"- Discordant pairs: `{paired_stats['best_vs_fixed']['mcnemar']['discordant']}`",
            f"- Fixed-only correct: `{paired_stats['best_vs_fixed']['mcnemar']['left_only_correct']}`",
            f"- Learned-only correct: `{paired_stats['best_vs_fixed']['mcnemar']['right_only_correct']}`",
            "",
            "Secondary pair: `fixed_knn_rolling_stable` vs `learned_finbert_rolling_stable`",
            "",
            f"- overall_acc delta: `{paired_stats['full_learned_vs_fixed']['overall_acc']['mean_delta']:+.4f}` "
            f"(95% CI [{paired_stats['full_learned_vs_fixed']['overall_acc']['ci_low']:+.4f}, "
            f"{paired_stats['full_learned_vs_fixed']['overall_acc']['ci_high']:+.4f}])",
            f"- McNemar exact p-value: `{paired_stats['full_learned_vs_fixed']['mcnemar']['p_value']:.6f}`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--fixed-head", default="stockmem/config/knn_head.fixed_knn_rolling_stable.json")
    parser.add_argument("--learned-head", default="stockmem/config/knn_head.learned_finbert_rolling_stable.json")
    parser.add_argument("--label-threshold", type=float, default=2.0)
    parser.add_argument("--out-dir", default="artifacts/learned_strict_test")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(data_path)
    weights = _load_knn_weights(Path(args.weights))
    metric = LearnedDiagonalMetric.load(args.artifact)
    fixed_head = _load_head_config(Path(args.fixed_head))
    learned_head = _load_head_config(Path(args.learned_head))

    def rank_fixed(query: LearnedRow, pool: list[LearnedRow]) -> list[LearnedRow]:
        scored = [(_fixed_score(query, cand, weights), cand) for cand in pool]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored]

    def rank_learned(query: LearnedRow, pool: list[LearnedRow]) -> list[LearnedRow]:
        scored = [(metric.score(query.blocks, cand.blocks), cand) for cand in pool]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored]

    results: list[tuple[PredictionMetrics, list[dict[str, object]]]] = [
        _evaluate_model(
            "fixed_knn_rolling_stable",
            rows,
            rank_fn=rank_fixed,
            head=fixed_head,
            label_threshold=args.label_threshold,
        ),
        _evaluate_model(
            "fixed_retriever_learned_head",
            rows,
            rank_fn=rank_fixed,
            head=learned_head,
            label_threshold=args.label_threshold,
        ),
        _evaluate_model(
            "learned_retriever_fixed_head",
            rows,
            rank_fn=rank_learned,
            head=fixed_head,
            label_threshold=args.label_threshold,
        ),
        _evaluate_model(
            "learned_finbert_rolling_stable",
            rows,
            rank_fn=rank_learned,
            head=learned_head,
            label_threshold=args.label_threshold,
        ),
    ]

    metrics = [metric_result for metric_result, _ in results]
    rows_by_name = {metric_result.name: rows_out for metric_result, rows_out in results}
    paired_stats: dict[str, dict[str, object]] = {}
    pair_specs = {
        "best_vs_fixed": "fixed_retriever_learned_head",
        "full_learned_vs_fixed": "learned_finbert_rolling_stable",
    }
    for pair_name, challenger in pair_specs.items():
        pair_payload: dict[str, object] = {
            "mcnemar": _mcnemar_exact(
                rows_by_name["fixed_knn_rolling_stable"],
                rows_by_name[challenger],
                label_threshold=args.label_threshold,
            )
        }
        for metric_name in ("overall_acc", "active_acc", "coverage", "hit_at_5_same_sign"):
            mean_delta, ci_low, ci_high = _bootstrap_metric_delta(
                rows_by_name["fixed_knn_rolling_stable"],
                rows_by_name[challenger],
                label_threshold=args.label_threshold,
                metric_name=metric_name,
            )
            pair_payload[metric_name] = {
                "mean_delta": mean_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        paired_stats[pair_name] = pair_payload
    for metric_result, rows_out in results:
        path = out_dir / f"{metric_result.name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "data_path": str(data_path),
        "weights_path": args.weights,
        "artifact_path": args.artifact,
        "fixed_head_path": args.fixed_head,
        "learned_head_path": args.learned_head,
        "label_threshold": args.label_threshold,
        "models": [asdict(item) for item in metrics],
        "paired_stats": paired_stats,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        metrics,
        out_dir / "summary.md",
        data_path=data_path,
        label_threshold=args.label_threshold,
        paired_stats=paired_stats,
    )


if __name__ == "__main__":
    main()
