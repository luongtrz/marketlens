from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from stockmem.scripts.experimental.train_majority_consensus_retriever import (
    ConsensusConfig,
    EvalRow,
    QueryCache,
    _evaluate_cache,
    _fixed_scores,
    _learned_scores,
    _load_rows,
    _matured_pool,
    _minmax,
    _regime_scores,
)
from stockmem.scripts.ndjson_eval_common import load_knn_weights
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _config_from_payload(payload: dict[str, Any]) -> ConsensusConfig:
    source = payload.get("config", payload)
    return ConsensusConfig(
        w_fixed=float(source["w_fixed"]),
        w_learned=float(source["w_learned"]),
        w_recency=float(source["w_recency"]),
        w_regime=float(source["w_regime"]),
        recency_half_life_days=float(source["recency_half_life_days"]),
    )


def _load_selected_config(path: Path) -> ConsensusConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _config_from_payload(payload)


def _build_query_cache(
    rows: list[EvalRow],
    *,
    queries: list[EvalRow],
    fixed_weights: tuple[float, float, float],
    learned_metric: LearnedDiagonalMetric,
    exclude_recent_days: int,
    min_pool_size: int,
) -> tuple[list[QueryCache], int]:
    caches: list[QueryCache] = []
    skipped = 0
    for index, query in enumerate(queries, start=1):
        pool = [
            candidate
            for candidate in _matured_pool(rows, query)
            if (query.date - candidate.date).days > exclude_recent_days
        ]
        if len(pool) < min_pool_size:
            skipped += 1
            continue
        labels = np.asarray([candidate.label_id for candidate in pool], dtype=np.int8)
        ages = np.asarray([(query.date - candidate.date).days for candidate in pool], dtype=np.float64)
        caches.append(
            QueryCache(
                query_date=query.date,
                actual_id=query.label_id,
                candidate_labels=labels,
                fixed=_minmax(_fixed_scores(query, pool, fixed_weights)),
                learned=_minmax(_learned_scores(query, pool, learned_metric)),
                age_days=ages,
                regime=_regime_scores(query, pool),
            )
        )
        if index % 500 == 0:
            print(f"cached queries={index}/{len(queries)} kept={len(caches)} skipped={skipped}", flush=True)
    return caches, skipped


def _query_rows(
    rows: list[EvalRow],
    *,
    split: str,
    start_date: date | None,
    end_date: date | None,
) -> list[EvalRow]:
    out: list[EvalRow] = []
    for row in rows:
        if split != "all" and row.split != split:
            continue
        if start_date is not None and row.date < start_date:
            continue
        if end_date is not None and row.date > end_date:
            continue
        out.append(row)
    return out


def _baseline_configs(half_life: float) -> dict[str, ConsensusConfig]:
    return {
        "fixed_only": ConsensusConfig(1.0, 0.0, 0.0, 0.0, half_life),
        "learned_only": ConsensusConfig(0.0, 1.0, 0.0, 0.0, half_life),
        "recency_only": ConsensusConfig(0.0, 0.0, 1.0, 0.0, half_life),
        "learned_recency_50_50": ConsensusConfig(0.0, 0.5, 0.5, 0.0, half_life),
        "fixed_recency_50_50": ConsensusConfig(0.5, 0.0, 0.5, 0.0, half_life),
    }


def _metric_row(split_name: str, model_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    by_actual = summary["by_actual"]
    return {
        "split": split_name,
        "model": model_name,
        "n": summary["n"],
        "hit_at_10": summary["hit_at_k"],
        "majority_at_10": summary["majority_same_at_k"],
        "mean_same_at_10": summary["mean_same_count"],
        "weighted_same_at_10": summary["weighted_same_score"],
        "buy_majority_at_10": by_actual["BUY"]["majority_same_at_k"],
        "hold_majority_at_10": by_actual["HOLD"]["majority_same_at_k"],
        "sell_majority_at_10": by_actual["SELL"]["majority_same_at_k"],
        "buy_mean_same_at_10": by_actual["BUY"]["mean_same_count"],
        "hold_mean_same_at_10": by_actual["HOLD"]["mean_same_count"],
        "sell_mean_same_at_10": by_actual["SELL"]["mean_same_count"],
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Majority-Consensus Retriever Evaluation",
        "",
        f"- Data: `{payload['data_path']}`",
        f"- Top-k: `{payload['top_k']}`",
        f"- Label threshold: `±{payload['label_threshold']:.2f}%` on `future_return_7d`",
        f"- Min pool size: `{payload['min_pool_size']}`",
        f"- Full-history date range: `{payload['full_start_date']}` to `{payload['full_end_date']}`",
        "",
    ]
    for split_name in payload["split_order"]:
        lines.extend(
            [
                f"## {split_name}",
                "",
                "| Model | n | Hit@10 | Majority@10 | Mean Same | Weighted Same | BUY Maj | HOLD Maj | SELL Maj |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in payload["rows"]:
            if row["split"] != split_name:
                continue
            lines.append(
                f"| `{row['model']}` | {row['n']} | {row['hit_at_10']:.4f} | "
                f"{row['majority_at_10']:.4f} | {row['mean_same_at_10']:.4f} | "
                f"{row['weighted_same_at_10']:.4f} | {row['buy_majority_at_10']:.4f} | "
                f"{row['hold_majority_at_10']:.4f} | {row['sell_majority_at_10']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Configs",
            "",
            "```json",
            json.dumps(payload["configs"], indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--out-dir", default="artifacts/majority_consensus_retriever_eval")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-pool-size", type=int, default=10)
    parser.add_argument("--label-threshold", type=float, default=2.0)
    parser.add_argument("--full-start-date", default="2018-01-01")
    parser.add_argument("--full-end-date", default=None)
    parser.add_argument("--half-life", type=float, default=21.0)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config spec as name:path/to/majority_consensus_retriever.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(Path(args.data), label_threshold=args.label_threshold)
    fixed_weights = load_knn_weights(Path(args.weights))
    learned_metric = LearnedDiagonalMetric.load(args.artifact)
    full_start = _parse_date(args.full_start_date)
    full_end = _parse_date(args.full_end_date) or rows[-1].date

    configs = _baseline_configs(args.half_life)
    for spec in args.config:
        if ":" not in spec:
            raise ValueError(f"Config must be name:path, got {spec!r}")
        name, raw_path = spec.split(":", 1)
        configs[name] = _load_selected_config(Path(raw_path))

    split_specs = [
        ("val", _query_rows(rows, split="val", start_date=None, end_date=None), 0),
        ("test", _query_rows(rows, split="test", start_date=None, end_date=None), 0),
        ("full_2018_now", _query_rows(rows, split="all", start_date=full_start, end_date=full_end), 0),
    ]

    result_rows: list[dict[str, Any]] = []
    split_meta: dict[str, Any] = {}
    for split_name, queries, exclude_recent_days in split_specs:
        print(f"building cache split={split_name} queries={len(queries)}", flush=True)
        cache, skipped = _build_query_cache(
            rows,
            queries=queries,
            fixed_weights=fixed_weights,
            learned_metric=learned_metric,
            exclude_recent_days=exclude_recent_days,
            min_pool_size=args.min_pool_size,
        )
        split_meta[split_name] = {
            "requested_queries": len(queries),
            "evaluated_queries": len(cache),
            "skipped_insufficient_pool": skipped,
        }
        for model_name, config in configs.items():
            summary = _evaluate_cache(cache, config=config, top_k=args.top_k)
            result_rows.append(_metric_row(split_name, model_name, summary))

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    payload = {
        "data_path": args.data,
        "weights_path": args.weights,
        "learned_artifact_path": args.artifact,
        "top_k": args.top_k,
        "min_pool_size": args.min_pool_size,
        "label_threshold": args.label_threshold,
        "full_start_date": full_start.isoformat() if full_start else None,
        "full_end_date": full_end.isoformat() if full_end else None,
        "split_order": [item[0] for item in split_specs],
        "split_meta": split_meta,
        "configs": {name: config.as_dict() for name, config in configs.items()},
        "rows": result_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "summary.md", payload)
    print(f"wrote majority consensus evaluation to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
