from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Callable

from stockmem.scripts.evaluate_learned_strict_test import (
    _bootstrap_metric_delta,
    _configured_head_signal,
    _fixed_score,
    _load_head_config,
    _load_knn_weights,
    _load_rows,
    _matured_pool,
    _mcnemar_exact,
    HeadConfig,
    LearnedRow,
)
from stockmem.scripts.ndjson_eval_common import PredictionMetrics, actual_signal, summarize_predictions
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


DEFAULT_START_DATE = date(2018, 1, 1)


def _parse_date(value: str | None) -> date | None:
    if value is None or value == "":
        return None
    return date.fromisoformat(value)


def _select_eval_rows(
    rows: list[LearnedRow],
    *,
    start_date: date | None,
    end_date: date | None,
    min_pool_size: int,
) -> tuple[list[LearnedRow], int]:
    selected: list[LearnedRow] = []
    skipped_insufficient_pool = 0
    for row in rows:
        if start_date is not None and row.date < start_date:
            continue
        if end_date is not None and row.date > end_date:
            continue
        pool = _matured_pool(rows, row)
        if len(pool) < min_pool_size:
            skipped_insufficient_pool += 1
            continue
        selected.append(row)
    return selected, skipped_insufficient_pool


def _evaluate_model(
    name: str,
    *,
    all_rows: list[LearnedRow],
    eval_rows: list[LearnedRow],
    rank_fn: Callable[[LearnedRow, list[LearnedRow]], list[LearnedRow]],
    head: HeadConfig,
    label_threshold: float,
) -> tuple[PredictionMetrics, list[dict[str, object]]]:
    out_rows: list[dict[str, object]] = []
    for query in eval_rows:
        pool = _matured_pool(all_rows, query)
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
                "split": query.split,
                "model": name,
                "predicted_signal": predicted,
                "actual_signal": actual,
                "actual_return_7d": query.future_returns.get("7d"),
                "confidence": confidence,
                "top5_same_sign": top5_same,
                "retrieval_count_reference": len(ranked),
                "matured_pool_size": len(pool),
            }
        )
    return summarize_predictions(name, out_rows, label_threshold=label_threshold), out_rows


def _paired_stats(
    baseline_rows: list[dict[str, object]],
    challenger_rows: list[dict[str, object]],
    *,
    label_threshold: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "mcnemar": _mcnemar_exact(
            baseline_rows,
            challenger_rows,
            label_threshold=label_threshold,
        )
    }
    for metric_name in ("overall_acc", "active_acc", "coverage", "hit_at_5_same_sign"):
        mean_delta, ci_low, ci_high = _bootstrap_metric_delta(
            baseline_rows,
            challenger_rows,
            label_threshold=label_threshold,
            metric_name=metric_name,
            samples=bootstrap_samples,
        )
        payload[metric_name] = {
            "mean_delta": mean_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return payload


def _write_metrics_csv(metrics: list[PredictionMetrics], path: Path) -> None:
    columns = [
        "name",
        "n",
        "overall_acc",
        "active_acc",
        "coverage",
        "hit_at_5_same_sign",
        "buy_rate",
        "hold_rate",
        "sell_rate",
        "avg_confidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in metrics:
            raw = asdict(item)
            writer.writerow({column: raw[column] for column in columns})


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# Full-History Structured StockMem Evaluation",
        "",
        "This is an exploratory full-history run, not the official held-out paper",
        "result. It evaluates current structured StockMem models over every eligible",
        "dataset row in the requested date range. Naive AI is intentionally excluded.",
        "",
        f"- Dataset: `{payload['data_path']}`",
        f"- Requested range: `{payload['requested_start_date']}` to `{payload['requested_end_date']}`",
        f"- Actual evaluated range: `{payload['actual_start_date']}` to `{payload['actual_end_date']}`",
        f"- Eligible rows: `{payload['eligible_rows']}`",
        f"- Skipped rows with insufficient matured pool: `{payload['skipped_insufficient_pool']}`",
        f"- Minimum matured pool size: `{payload['min_pool_size']}`",
        f"- Label threshold: `+/-{float(payload['label_threshold']):.2f}%` on `future_return_7d`",
        f"- Bootstrap samples for paired CIs: `{payload['bootstrap_samples']}`",
        "",
        "## Main Table",
        "",
        "| Model | n | Overall | Active | Coverage | Hit@5 | BUY% | HOLD% | SELL% | Avg Conf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["models"]:  # type: ignore[index]
        lines.append(
            "| "
            f"`{item['name']}` | {item['n']} | {item['overall_acc']:.4f} | "
            f"{item['active_acc']:.4f} | {item['coverage']:.4f} | "
            f"{item['hit_at_5_same_sign']:.4f} | {item['buy_rate']:.4f} | "
            f"{item['hold_rate']:.4f} | {item['sell_rate']:.4f} | "
            f"{item['avg_confidence']:.4f} |"
        )

    lines.extend(["", "## Paired Against Fixed kNN", ""])
    paired = payload["paired_stats"]  # type: ignore[assignment]
    for challenger, stats in paired.items():  # type: ignore[union-attr]
        lines.extend(
            [
                f"### `{challenger}`",
                "",
                "| Metric | Delta vs fixed | 95% bootstrap CI |",
                "| --- | ---: | ---: |",
            ]
        )
        for metric_name in ("overall_acc", "active_acc", "coverage", "hit_at_5_same_sign"):
            item = stats[metric_name]
            lines.append(
                f"| {metric_name} | {item['mean_delta']:+.4f} | "
                f"[{item['ci_low']:+.4f}, {item['ci_high']:+.4f}] |"
            )
        mcnemar = stats["mcnemar"]
        lines.extend(
            [
                "",
                f"- McNemar p: `{mcnemar['p_value']:.6f}`",
                f"- Discordant pairs: `{mcnemar['discordant']}`",
                f"- Fixed-only correct: `{mcnemar['left_only_correct']}`",
                f"- Challenger-only correct: `{mcnemar['right_only_correct']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Guardrail",
            "",
            "This run mixes train, validation, and test eras, so it is useful for",
            "sanity-checking long-history behavior but should not replace the official",
            "chronological held-out test. Use it to inspect stability and regime behavior,",
            "not as the primary claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_full_history(
    *,
    data_path: Path,
    weights_path: Path,
    learned_artifact_path: Path,
    fixed_head_path: Path,
    learned_head_path: Path,
    out_dir: Path,
    start_date: date | None,
    end_date: date | None,
    min_pool_size: int,
    label_threshold: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    rows = _load_rows(data_path)
    if not rows:
        raise SystemExit("no rows loaded")

    requested_start = start_date or DEFAULT_START_DATE
    requested_end = end_date or rows[-1].date
    eval_rows, skipped_insufficient_pool = _select_eval_rows(
        rows,
        start_date=requested_start,
        end_date=requested_end,
        min_pool_size=min_pool_size,
    )
    if not eval_rows:
        raise SystemExit("no eligible eval rows")

    weights = _load_knn_weights(weights_path)
    learned_metric = LearnedDiagonalMetric.load(learned_artifact_path)
    fixed_head = _load_head_config(fixed_head_path)
    learned_head = _load_head_config(learned_head_path)

    def rank_fixed(query: LearnedRow, pool: list[LearnedRow]) -> list[LearnedRow]:
        scored = [(_fixed_score(query, candidate, weights), candidate) for candidate in pool]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored]

    def rank_learned(query: LearnedRow, pool: list[LearnedRow]) -> list[LearnedRow]:
        scored = [(learned_metric.score(query.blocks, candidate.blocks), candidate) for candidate in pool]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored]

    results = [
        _evaluate_model(
            "fixed_knn_rolling_stable",
            all_rows=rows,
            eval_rows=eval_rows,
            rank_fn=rank_fixed,
            head=fixed_head,
            label_threshold=label_threshold,
        ),
        _evaluate_model(
            "fixed_retriever_learned_head",
            all_rows=rows,
            eval_rows=eval_rows,
            rank_fn=rank_fixed,
            head=learned_head,
            label_threshold=label_threshold,
        ),
        _evaluate_model(
            "learned_retriever_fixed_head",
            all_rows=rows,
            eval_rows=eval_rows,
            rank_fn=rank_learned,
            head=fixed_head,
            label_threshold=label_threshold,
        ),
        _evaluate_model(
            "learned_finbert_rolling_stable",
            all_rows=rows,
            eval_rows=eval_rows,
            rank_fn=rank_learned,
            head=learned_head,
            label_threshold=label_threshold,
        ),
    ]

    metrics = [item for item, _ in results]
    rows_by_name = {item.name: rows_out for item, rows_out in results}
    paired_stats = {
        name: _paired_stats(
            rows_by_name["fixed_knn_rolling_stable"],
            rows_out,
            label_threshold=label_threshold,
            bootstrap_samples=bootstrap_samples,
        )
        for name, rows_out in rows_by_name.items()
        if name != "fixed_knn_rolling_stable"
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, rows_out in results:
        with (out_dir / f"{metric.name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload: dict[str, object] = {
        "data_path": str(data_path),
        "weights_path": str(weights_path),
        "learned_artifact_path": str(learned_artifact_path),
        "fixed_head_path": str(fixed_head_path),
        "learned_head_path": str(learned_head_path),
        "requested_start_date": requested_start.isoformat(),
        "requested_end_date": requested_end.isoformat(),
        "actual_start_date": eval_rows[0].date.isoformat(),
        "actual_end_date": eval_rows[-1].date.isoformat(),
        "eligible_rows": len(eval_rows),
        "skipped_insufficient_pool": skipped_insufficient_pool,
        "min_pool_size": min_pool_size,
        "label_threshold": label_threshold,
        "bootstrap_samples": bootstrap_samples,
        "models": [asdict(item) for item in metrics],
        "paired_stats": paired_stats,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(metrics, out_dir / "summary.csv")
    _write_markdown(payload, out_dir / "summary.md")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structured StockMem models over full history")
    parser.add_argument("--data", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--learned-artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--fixed-head", default="stockmem/config/knn_head.fixed_knn_rolling_stable.json")
    parser.add_argument("--learned-head", default="stockmem/config/knn_head.learned_finbert_rolling_stable.json")
    parser.add_argument("--out-dir", default="artifacts/full_history_structured_models")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-pool-size", type=int, default=5)
    parser.add_argument("--label-threshold", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    payload = evaluate_full_history(
        data_path=Path(args.data),
        weights_path=Path(args.weights),
        learned_artifact_path=Path(args.learned_artifact),
        fixed_head_path=Path(args.fixed_head),
        learned_head_path=Path(args.learned_head),
        out_dir=Path(args.out_dir),
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        min_pool_size=args.min_pool_size,
        label_threshold=args.label_threshold,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        "wrote full-history structured evaluation to "
        f"{args.out_dir} ({payload['eligible_rows']} eligible rows)"
    )


if __name__ == "__main__":
    main()
