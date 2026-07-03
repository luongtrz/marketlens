from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stockmem.scripts.evaluate_learned_strict_test import _assign_split, _l2
from stockmem.scripts.ndjson_eval_common import actual_signal, load_knn_weights
from stockmem.src.models import StockMemRecord
from stockmem.src.search.learned_metric import LearnedDiagonalMetric
from stockmem.src.search.searcher import _get_regime

CLASSES = ("BUY", "HOLD", "SELL")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class EvalRow:
    record: StockMemRecord
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray
    event_vec: np.ndarray
    split: str
    label_id: int
    regime: str

    @property
    def date(self) -> date:
        return self.record.date

    @property
    def blocks(self) -> tuple[np.ndarray, ...]:
        return (self.event_vec, self.factor_vec, self.indicator_vec, self.price_vec)


@dataclass(frozen=True)
class QueryCache:
    query_date: date
    actual_id: int
    candidate_labels: np.ndarray
    fixed: np.ndarray
    learned: np.ndarray
    age_days: np.ndarray
    regime: np.ndarray


@dataclass(frozen=True)
class ExperimentConstraints:
    min_memory_weight: float
    max_recency_weight: float
    min_learned_weight: float
    exclude_recent_days: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "min_memory_weight": self.min_memory_weight,
            "max_recency_weight": self.max_recency_weight,
            "min_learned_weight": self.min_learned_weight,
            "exclude_recent_days": self.exclude_recent_days,
        }


@dataclass(frozen=True)
class ConsensusConfig:
    w_fixed: float
    w_learned: float
    w_recency: float
    w_regime: float
    recency_half_life_days: float

    def as_dict(self) -> dict[str, float]:
        return {
            "w_fixed": self.w_fixed,
            "w_learned": self.w_learned,
            "w_recency": self.w_recency,
            "w_regime": self.w_regime,
            "recency_half_life_days": self.recency_half_life_days,
        }


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values.astype(np.float64) - lo) / (hi - lo)


def _load_rows(path: Path, *, label_threshold: float) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            payload = raw.get("payload", raw)
            record = StockMemRecord.model_validate(payload)
            if record.future_return_7d is None:
                continue
            factor_raw = payload.get("factor_vec") or payload.get("factor_vector") or []
            indicator_raw = payload.get("indicator_vec") or []
            price_raw = payload.get("price_vec") or []
            event_raw = payload.get("event_vec") or []
            if len(factor_raw) != 75 or len(indicator_raw) != 5 or len(price_raw) != 60:
                continue
            if len(event_raw) != 85:
                event_raw = [0.0] * 85
            label = actual_signal(record.future_return_7d, label_threshold)
            rows.append(
                EvalRow(
                    record=record,
                    factor_vec=_l2(factor_raw),
                    indicator_vec=_l2(indicator_raw),
                    price_vec=_l2(price_raw),
                    event_vec=_l2(event_raw),
                    split=_assign_split(record.date),
                    label_id=CLASS_TO_ID[label],
                    regime=_get_regime(record),
                )
            )
    rows.sort(key=lambda row: row.date)
    return rows


def _matured_pool(rows: list[EvalRow], query: EvalRow, *, horizon_days: int = 7) -> list[EvalRow]:
    return [
        candidate
        for candidate in rows
        if candidate.date < query.date and candidate.date + timedelta(days=horizon_days) <= query.date
    ]


def _fixed_scores(query: EvalRow, pool: list[EvalRow], weights: tuple[float, float, float]) -> np.ndarray:
    if not pool:
        return np.asarray([], dtype=np.float64)
    factors = np.vstack([row.factor_vec for row in pool])
    indicators = np.vstack([row.indicator_vec for row in pool])
    prices = np.vstack([row.price_vec for row in pool])
    return (
        weights[0] * (factors @ query.factor_vec)
        + weights[1] * (indicators @ query.indicator_vec)
        + weights[2] * (prices @ query.price_vec)
    )


def _learned_scores(query: EvalRow, pool: list[EvalRow], metric: LearnedDiagonalMetric) -> np.ndarray:
    if not pool:
        return np.asarray([], dtype=np.float64)
    stacked = (
        np.vstack([row.event_vec for row in pool]),
        np.vstack([row.factor_vec for row in pool]),
        np.vstack([row.indicator_vec for row in pool]),
        np.vstack([row.price_vec for row in pool]),
    )
    return metric.score_batch(query.blocks, stacked)


def _regime_scores(query: EvalRow, pool: list[EvalRow]) -> np.ndarray:
    values: list[float] = []
    for candidate in pool:
        if candidate.regime == query.regime:
            values.append(1.0)
        elif candidate.regime == "neutral" or query.regime == "neutral":
            values.append(0.5)
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float64)


def _build_cache(
    rows: list[EvalRow],
    *,
    split: str,
    fixed_weights: tuple[float, float, float],
    learned_metric: LearnedDiagonalMetric,
    exclude_recent_days: int = 0,
) -> list[QueryCache]:
    caches: list[QueryCache] = []
    queries = [row for row in rows if row.split == split]
    for index, query in enumerate(queries, start=1):
        pool = [
            candidate
            for candidate in _matured_pool(rows, query)
            if (query.date - candidate.date).days > exclude_recent_days
        ]
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
        if index % 50 == 0:
            print(f"cached {split} queries={index}/{len(queries)}", flush=True)
    return caches


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size <= k:
        return np.argsort(-scores)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])]


def _rank_scores(cache: QueryCache, config: ConsensusConfig) -> np.ndarray:
    recency = np.exp(-cache.age_days / config.recency_half_life_days)
    return (
        config.w_fixed * cache.fixed
        + config.w_learned * cache.learned
        + config.w_recency * recency
        + config.w_regime * cache.regime
    )


def _top_indices_for_config(
    cache: QueryCache,
    *,
    config: ConsensusConfig,
    top_k: int,
) -> np.ndarray:
    if cache.candidate_labels.size == 0:
        return np.asarray([], dtype=np.int64)
    scores = _rank_scores(cache, config)
    return _top_indices(scores, min(top_k, scores.size))


def _top_indices_composite(
    cache: QueryCache,
    *,
    memory_config: ConsensusConfig,
    recent_config: ConsensusConfig,
    top_k: int,
    recent_slots: int,
) -> np.ndarray:
    if cache.candidate_labels.size == 0:
        return np.asarray([], dtype=np.int64)
    recent_slots = max(0, min(recent_slots, top_k))
    memory_slots = top_k - recent_slots
    selected: list[int] = []
    if recent_slots:
        recent_scores = _rank_scores(cache, recent_config)
        selected.extend(int(index) for index in _top_indices(recent_scores, min(recent_slots, recent_scores.size)))
    if memory_slots:
        memory_scores = _rank_scores(cache, memory_config)
        order = _top_indices(memory_scores, min(top_k + len(selected) + 20, memory_scores.size))
        selected_set = set(selected)
        for index in order:
            if int(index) in selected_set:
                continue
            selected.append(int(index))
            selected_set.add(int(index))
            if len(selected) >= top_k:
                break
    return np.asarray(selected[:top_k], dtype=np.int64)


def _summarize_same_counts(
    same_counts: list[tuple[int, int]],
    *,
    top_k: int,
) -> dict[str, Any]:
    majority_threshold = (top_k + 1) // 2
    dist: Counter[int] = Counter()
    by_actual: dict[str, Counter[int]] = defaultdict(Counter)
    for actual_id, same_count in same_counts:
        actual = CLASSES[actual_id]
        dist[same_count] += 1
        by_actual[actual][same_count] += 1

    def summarize(counter: Counter[int]) -> dict[str, Any]:
        n = sum(counter.values())
        if n == 0:
            return {
                "n": 0,
                "distribution": {},
                "hit_at_k": 0.0,
                "majority_same_at_k": 0.0,
                "mean_same_count": 0.0,
                "weighted_same_score": 0.0,
            }
        return {
            "n": n,
            "distribution": dict(sorted(counter.items())),
            "hit_at_k": sum(v for k, v in counter.items() if k >= 1) / n,
            "majority_same_at_k": sum(v for k, v in counter.items() if k >= majority_threshold) / n,
            "mean_same_count": sum(k * v for k, v in counter.items()) / n,
            "weighted_same_score": sum((k / top_k) * v for k, v in counter.items()) / n,
        }

    summary = summarize(dist)
    summary["top_k"] = top_k
    summary["majority_threshold"] = majority_threshold
    summary["by_actual"] = {label: summarize(by_actual[label]) for label in CLASSES}
    return summary


def _evaluate_cache(
    caches: list[QueryCache],
    *,
    config: ConsensusConfig,
    top_k: int,
) -> dict[str, Any]:
    same_counts: list[tuple[int, int]] = []
    for cache in caches:
        if cache.candidate_labels.size == 0:
            same_count = 0
        else:
            top = _top_indices_for_config(cache, config=config, top_k=top_k)
            same_count = int(np.sum(cache.candidate_labels[top] == cache.actual_id))
        same_counts.append((cache.actual_id, same_count))
    return _summarize_same_counts(same_counts, top_k=top_k)


def _evaluate_composite_cache(
    caches: list[QueryCache],
    *,
    memory_config: ConsensusConfig,
    recent_config: ConsensusConfig,
    top_k: int,
    recent_slots: int,
) -> dict[str, Any]:
    same_counts: list[tuple[int, int]] = []
    for cache in caches:
        top = _top_indices_composite(
            cache,
            memory_config=memory_config,
            recent_config=recent_config,
            top_k=top_k,
            recent_slots=recent_slots,
        )
        same_count = int(np.sum(cache.candidate_labels[top] == cache.actual_id)) if top.size else 0
        same_counts.append((cache.actual_id, same_count))
    return _summarize_same_counts(same_counts, top_k=top_k)


def _objective(summary: dict[str, Any]) -> float:
    by_actual = summary["by_actual"]
    return (
        0.35 * float(summary["majority_same_at_k"])
        + 0.20 * float(summary["weighted_same_score"])
        + 0.20 * float(by_actual["SELL"]["majority_same_at_k"])
        + 0.10 * float(by_actual["BUY"]["majority_same_at_k"])
        + 0.10 * float(by_actual["SELL"]["weighted_same_score"])
        + 0.05 * float(by_actual["HOLD"]["weighted_same_score"])
    )


def _weight_grid(step: float) -> Iterable[tuple[float, float, float, float]]:
    units = int(round(1.0 / step))
    for fixed in range(units + 1):
        for learned in range(units - fixed + 1):
            for recency in range(units - fixed - learned + 1):
                regime = units - fixed - learned - recency
                yield (
                    round(fixed * step, 10),
                    round(learned * step, 10),
                    round(recency * step, 10),
                    round(regime * step, 10),
                )


def _candidate_configs(
    *,
    step: float,
    half_lives: list[float],
    constraints: ExperimentConstraints,
) -> list[ConsensusConfig]:
    configs: list[ConsensusConfig] = []
    for half_life in half_lives:
        for w_fixed, w_learned, w_recency, w_regime in _weight_grid(step):
            if w_fixed + w_learned < constraints.min_memory_weight:
                continue
            if w_learned < constraints.min_learned_weight:
                continue
            if w_recency > constraints.max_recency_weight:
                continue
            configs.append(
                ConsensusConfig(
                    w_fixed=w_fixed,
                    w_learned=w_learned,
                    w_recency=w_recency,
                    w_regime=w_regime,
                    recency_half_life_days=half_life,
                )
            )
    return configs


def _named_baselines(half_life: float) -> dict[str, ConsensusConfig]:
    return {
        "fixed_only": ConsensusConfig(1.0, 0.0, 0.0, 0.0, half_life),
        "learned_only": ConsensusConfig(0.0, 1.0, 0.0, 0.0, half_life),
        "recency_only": ConsensusConfig(0.0, 0.0, 1.0, 0.0, half_life),
        "regime_only": ConsensusConfig(0.0, 0.0, 0.0, 1.0, half_life),
        "fixed_recency_50_50": ConsensusConfig(0.5, 0.0, 0.5, 0.0, half_life),
        "learned_recency_50_50": ConsensusConfig(0.0, 0.5, 0.5, 0.0, half_life),
        "memory_first_learned_recency_60_30_10": ConsensusConfig(0.0, 0.6, 0.3, 0.1, half_life),
        "memory_first_fixed_recency_60_30_10": ConsensusConfig(0.6, 0.0, 0.3, 0.1, half_life),
        "balanced_fixed_learned_recency_regime": ConsensusConfig(0.3, 0.3, 0.3, 0.1, half_life),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Majority-Consensus Retriever Training",
        "",
        f"- Data: `{payload['data_path']}`",
        f"- Learned metric: `{payload['learned_artifact_path']}`",
        f"- Fixed weights: `{payload['fixed_weights_path']}`",
        f"- Top-k: `{payload['top_k']}`",
        f"- Label threshold: `±{payload['label_threshold']:.2f}%` on `future_return_7d`",
        "- Selection split: validation",
        "- Test split: held out",
        f"- Constraints: `{payload['constraints']}`",
        "",
        "## Selected Config",
        "",
        "```json",
        json.dumps(payload["selected_config"], indent=2),
        "```",
        "",
        "## Validation",
        "",
        "| Objective | Hit@k | Majority | Mean same | SELL majority | BUY majority | HOLD weighted |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    val = payload["selected_validation"]
    lines.append(
        f"| {payload['selected_validation_objective']:.4f} | {val['hit_at_k']:.4f} | "
        f"{val['majority_same_at_k']:.4f} | {val['mean_same_count']:.4f} | "
        f"{val['by_actual']['SELL']['majority_same_at_k']:.4f} | "
        f"{val['by_actual']['BUY']['majority_same_at_k']:.4f} | "
        f"{val['by_actual']['HOLD']['weighted_same_score']:.4f} |"
    )
    lines.extend(
        [
            "",
            "## Test Comparison",
            "",
            "| Retriever | Hit@k | Majority | Mean same | Weighted same | BUY majority | HOLD majority | SELL majority |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["test_comparison"]:
        test = item["test"]
        lines.append(
            f"| `{item['name']}` | {test['hit_at_k']:.4f} | {test['majority_same_at_k']:.4f} | "
            f"{test['mean_same_count']:.4f} | {test['weighted_same_score']:.4f} | "
            f"{test['by_actual']['BUY']['majority_same_at_k']:.4f} | "
            f"{test['by_actual']['HOLD']['majority_same_at_k']:.4f} | "
            f"{test['by_actual']['SELL']['majority_same_at_k']:.4f} |"
        )
    if payload.get("composite_test_comparison"):
        lines.extend(
            [
                "",
                "## Composite Test Comparison",
                "",
                "| Retriever | Recent slots | Hit@k | Majority | Mean same | BUY majority | HOLD majority | SELL majority |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in payload["composite_test_comparison"]:
            test = item["test"]
            lines.append(
                f"| `{item['name']}` | {item['recent_slots']} | {test['hit_at_k']:.4f} | "
                f"{test['majority_same_at_k']:.4f} | {test['mean_same_count']:.4f} | "
                f"{test['by_actual']['BUY']['majority_same_at_k']:.4f} | "
                f"{test['by_actual']['HOLD']['majority_same_at_k']:.4f} | "
                f"{test['by_actual']['SELL']['majority_same_at_k']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Top Validation Candidates",
            "",
            "| Rank | Objective | Config | Val majority | Val SELL majority | Val mean same |",
            "| ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for item in payload["top_validation_candidates"]:
        metrics = item["validation"]
        lines.append(
            f"| {item['rank']} | {item['objective']:.4f} | `{item['config']}` | "
            f"{metrics['majority_same_at_k']:.4f} | "
            f"{metrics['by_actual']['SELL']['majority_same_at_k']:.4f} | "
            f"{metrics['mean_same_count']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--out-dir", default="artifacts/majority_consensus_retriever")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label-threshold", type=float, default=2.0)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--min-memory-weight", type=float, default=0.0)
    parser.add_argument("--max-recency-weight", type=float, default=1.0)
    parser.add_argument("--min-learned-weight", type=float, default=0.0)
    parser.add_argument(
        "--exclude-recent-days",
        type=int,
        default=0,
        help="Exclude candidates whose age is less than or equal to this value.",
    )
    parser.add_argument(
        "--half-lives",
        default="7,14,21,30,45,60,90,120,180,365",
        help="Comma-separated recency half-life values in days.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    half_lives = [float(item) for item in args.half_lives.split(",") if item.strip()]
    constraints = ExperimentConstraints(
        min_memory_weight=args.min_memory_weight,
        max_recency_weight=args.max_recency_weight,
        min_learned_weight=args.min_learned_weight,
        exclude_recent_days=args.exclude_recent_days,
    )
    rows = _load_rows(Path(args.data), label_threshold=args.label_threshold)
    fixed_weights = load_knn_weights(Path(args.weights))
    learned_metric = LearnedDiagonalMetric.load(args.artifact)

    print("building validation cache", flush=True)
    val_cache = _build_cache(
        rows,
        split="val",
        fixed_weights=fixed_weights,
        learned_metric=learned_metric,
        exclude_recent_days=args.exclude_recent_days,
    )
    print("building test cache", flush=True)
    test_cache = _build_cache(
        rows,
        split="test",
        fixed_weights=fixed_weights,
        learned_metric=learned_metric,
        exclude_recent_days=args.exclude_recent_days,
    )

    configs = _candidate_configs(
        step=args.grid_step,
        half_lives=half_lives,
        constraints=constraints,
    )
    if not configs:
        raise ValueError(f"No candidate configs satisfy constraints: {constraints.as_dict()}")
    print(f"evaluating configs={len(configs)}", flush=True)
    scored: list[tuple[float, ConsensusConfig, dict[str, Any]]] = []
    for index, config in enumerate(configs, start=1):
        summary = _evaluate_cache(val_cache, config=config, top_k=args.top_k)
        scored.append((_objective(summary), config, summary))
        if index % 500 == 0:
            best = max(item[0] for item in scored)
            print(f"checked={index}/{len(configs)} best_objective={best:.6f}", flush=True)
    scored.sort(key=lambda item: item[0], reverse=True)
    best_objective, best_config, best_val = scored[0]

    baseline_half_life = best_config.recency_half_life_days
    comparisons: list[dict[str, Any]] = []
    for name, config in _named_baselines(baseline_half_life).items():
        comparisons.append(
            {
                "name": name,
                "config": config.as_dict(),
                "validation": _evaluate_cache(val_cache, config=config, top_k=args.top_k),
                "test": _evaluate_cache(test_cache, config=config, top_k=args.top_k),
            }
        )
    comparisons.append(
        {
            "name": "selected_majority_consensus",
            "config": best_config.as_dict(),
            "validation": best_val,
            "test": _evaluate_cache(test_cache, config=best_config, top_k=args.top_k),
        }
    )

    selected_artifact = {
        "version": "majority_consensus_retriever_v1",
        "type": "time_aware_consensus",
        "top_k": args.top_k,
        "label_threshold": args.label_threshold,
        "components": ["fixed_similarity", "learned_similarity", "recency_decay", "regime_match"],
        "normalization": "per_query_minmax_for_similarity_components",
        "config": best_config.as_dict(),
        "selection_protocol": {
            "selection_split": "validation",
            "objective": "majority_same_at_10_weighted_with_sell_consensus",
            "grid_step": args.grid_step,
            "half_lives": half_lives,
            "constraints": constraints.as_dict(),
        },
        "validation_objective": best_objective,
        "validation_summary": best_val,
    }
    (out_dir / "majority_consensus_retriever.json").write_text(
        json.dumps(selected_artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    payload = {
        "data_path": args.data,
        "fixed_weights_path": args.weights,
        "learned_artifact_path": args.artifact,
        "top_k": args.top_k,
        "label_threshold": args.label_threshold,
        "constraints": constraints.as_dict(),
        "selected_config": best_config.as_dict(),
        "selected_validation_objective": best_objective,
        "selected_validation": best_val,
        "test_comparison": comparisons,
        "composite_test_comparison": [],
        "top_validation_candidates": [
            {
                "rank": rank,
                "objective": objective,
                "config": config.as_dict(),
                "validation": validation,
            }
            for rank, (objective, config, validation) in enumerate(scored[:20], start=1)
        ],
    }

    recent_config = ConsensusConfig(0.0, 0.0, 1.0, 0.0, best_config.recency_half_life_days)
    composite_specs = [
        ("selected_memory_plus_2_recent", best_config, 2),
        ("selected_memory_plus_3_recent", best_config, 3),
        ("learned_memory_plus_2_recent", ConsensusConfig(0.0, 1.0, 0.0, 0.0, best_config.recency_half_life_days), 2),
        ("learned_memory_plus_3_recent", ConsensusConfig(0.0, 1.0, 0.0, 0.0, best_config.recency_half_life_days), 3),
        ("balanced_memory_plus_2_recent", ConsensusConfig(0.4, 0.4, 0.0, 0.2, best_config.recency_half_life_days), 2),
        ("balanced_memory_plus_3_recent", ConsensusConfig(0.4, 0.4, 0.0, 0.2, best_config.recency_half_life_days), 3),
    ]
    for name, memory_config, recent_slots in composite_specs:
        payload["composite_test_comparison"].append(
            {
                "name": name,
                "recent_slots": recent_slots,
                "memory_config": memory_config.as_dict(),
                "recent_config": recent_config.as_dict(),
                "validation": _evaluate_composite_cache(
                    val_cache,
                    memory_config=memory_config,
                    recent_config=recent_config,
                    top_k=args.top_k,
                    recent_slots=recent_slots,
                ),
                "test": _evaluate_composite_cache(
                    test_cache,
                    memory_config=memory_config,
                    recent_config=recent_config,
                    top_k=args.top_k,
                    recent_slots=recent_slots,
                ),
            }
        )
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "summary.md", payload)
    print(f"wrote majority consensus retriever training to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
