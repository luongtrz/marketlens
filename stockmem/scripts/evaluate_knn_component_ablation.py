"""Run Experiment 2: fixed-kNN block ablation on the current pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from stockmem.scripts.ndjson_eval_common import (
    HeadConfig,
    PredictionMetrics,
    actual_signal,
    configured_head_signal,
    load_head_config,
    load_historical_rows,
    load_knn_weights,
    matured_pool,
    retrieve_fixed_knn,
    summarize_predictions,
)

logger = logging.getLogger(__name__)


def _variant_weights(
    weights: tuple[float, float, float],
) -> list[tuple[str, tuple[float, float, float]]]:
    w1, w2, w3 = weights
    return [
        ("full_fixed_knn", (w1, w2, w3)),
        ("no_factor_block", (0.0, w2, w3)),
        ("no_indicator_block", (w1, 0.0, w3)),
        ("no_price_block", (w1, w2, 0.0)),
        ("factor_only", (w1, 0.0, 0.0)),
        ("indicator_only", (0.0, w2, 0.0)),
        ("price_only", (0.0, 0.0, w3)),
    ]


def _write_markdown(
    metrics: list[PredictionMetrics],
    out_path: Path,
    *,
    data_path: Path,
    k: int,
    label_threshold: float,
) -> None:
    baseline = next((item for item in metrics if item.name == "full_fixed_knn"), None)
    lines = [
        "# Fixed-kNN Component Ablation",
        "",
        f"- Data source: `{data_path}`",
        f"- Test split: `2025-07-01` to `2026-05-01`",
        f"- Label threshold: `±{label_threshold:.2f}%` on `future_return_7d`",
        f"- Retrieval depth: `k={k}`",
        "",
        "| Variant | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | Delta overall | Delta active |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        base_overall = baseline.overall_acc if baseline else 0.0
        base_active = baseline.active_acc if baseline else 0.0
        lines.append(
            "| "
            f"{item.name} | {item.n} | {item.overall_acc:.4f} | {item.active_acc:.4f} | "
            f"{item.coverage:.4f} | {item.hit_at_5_same_sign:.4f} | "
            f"{item.overall_acc - base_overall:+.4f} | {item.active_acc - base_active:+.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--out-dir", default="artifacts/fixed_knn_component_ablation")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--fixed-head", default="stockmem/config/knn_head.fixed_knn_rolling_stable.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label-threshold", type=float, default=2.0)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_historical_rows(data_path)
    test_rows = [row for row in rows if row.split == "test"]
    if args.max_queries is not None:
        test_rows = test_rows[:args.max_queries]

    base_weights = load_knn_weights(Path(args.weights))
    fixed_head: HeadConfig = load_head_config(Path(args.fixed_head))
    variant_rows: dict[str, list[dict[str, object]]] = {
        name: [] for name, _ in _variant_weights(base_weights)
    }

    for index, query in enumerate(test_rows, start=1):
        pool = matured_pool(rows, query)
        actual = actual_signal(query.record.future_return_7d, args.label_threshold)

        for name, weights in _variant_weights(base_weights):
            similar = retrieve_fixed_knn(query, pool, weights=weights, k=fixed_head.k)
            predicted, confidence = configured_head_signal(similar, head=fixed_head)
            top5_same = any(
                actual_signal(item.record.future_return_7d, args.label_threshold) == actual
                for item in similar[:5]
            )
            variant_rows[name].append(
                {
                    "date": query.record.date.isoformat(),
                    "variant": name,
                    "predicted_signal": predicted,
                    "actual_signal": actual,
                    "actual_return_7d": query.record.future_return_7d,
                    "confidence": confidence,
                    "top5_same_sign": top5_same,
                    "retrieval_count_reference": len(similar),
                }
            )

        if args.progress_every > 0 and index % args.progress_every == 0:
            baseline_partial = summarize_predictions(
                "full_fixed_knn",
                variant_rows["full_fixed_knn"],
                label_threshold=args.label_threshold,
            )
            logger.info(
                "[full_fixed_knn] %d/%d overall_acc=%.4f active_acc=%.4f coverage=%.4f",
                index,
                len(test_rows),
                baseline_partial.overall_acc,
                baseline_partial.active_acc,
                baseline_partial.coverage,
            )

    metrics: list[PredictionMetrics] = []
    for name, rows_out in variant_rows.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics.append(summarize_predictions(name, rows_out, label_threshold=args.label_threshold))

    summary_json = {
        "data_path": str(data_path),
        "weights_path": args.weights,
        "fixed_head_path": args.fixed_head,
        "k": args.k,
        "label_threshold": args.label_threshold,
        "variants": [asdict(item) for item in metrics],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        metrics,
        out_dir / "summary.md",
        data_path=data_path,
        k=args.k,
        label_threshold=args.label_threshold,
    )
    logger.info("summary json: %s", out_dir / "summary.json")
    logger.info("summary md:   %s", out_dir / "summary.md")


if __name__ == "__main__":
    main()
