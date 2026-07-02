from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from stockmem.scripts.cem_dataset import (
    LabeledRow,
    label_rows,
    matured_pool,
    ndcg_at_k,
    snapshot_similarity,
    teacher_relevance,
)
from stockmem.scripts.hybrid_reranking import (
    HybridComponentScores,
    HybridRankedCandidate,
    HybridRerankWeights,
    d7_label,
    rerank_knn_candidates,
)
from stockmem.scripts.optimize_weights import load_rows, validate_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


DEFAULT_SEARCH_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)
RETURN_WEIGHTS_DEFAULT = {"1d": 0.40, "3d": 0.30, "7d": 0.15, "15d": 0.10, "30d": 0.05}
DEFAULT_HYBRID_WEIGHTS = HybridRerankWeights(
    w_knn=0.5,
    w_learned=0.3,
    w_regime=0.1,
    w_prior=0.1,
)


@dataclass(frozen=True)
class RetrievedEvidence:
    date: str
    future_return_7d: float
    d7_label: str
    score: float
    components: dict[str, float]
    original_rank: int | None = None


@dataclass(frozen=True)
class MethodEvaluation:
    name: str
    top5_same_d7_sign_rate: float
    ndcg_at_5: float
    downstream_da: float
    active_acc: float
    coverage: float
    evidence_coverage: float
    queries_evaluated: int
    directional_queries: int
    examples: list[dict]


def _fixed_score(
    query: LabeledRow,
    candidate: LabeledRow,
    weights: tuple[float, float, float],
) -> float:
    return snapshot_similarity(query, candidate, weights=weights)


def _multi_horizon_average(candidate: LabeledRow) -> float | None:
    horizons = {
        "1d": candidate.row.future_return_1d,
        "3d": getattr(candidate.row, "future_return_3d", None),
        "7d": candidate.row.future_return_7d,
        "15d": getattr(candidate.row, "future_return_15d", None),
        "30d": candidate.row.future_return_30d,
    }
    total_weight = 0.0
    total_value = 0.0
    for horizon, weight in RETURN_WEIGHTS_DEFAULT.items():
        value = horizons.get(horizon)
        if value is None:
            continue
        total_value += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight <= 1e-12:
        return None
    return total_value / total_weight


def _predicted_return_7d(
    ranked: Sequence[HybridRankedCandidate],
    *,
    top_k: int,
    use_production_head: bool,
) -> float:
    top_candidates = ranked[:top_k]
    if use_production_head:
        values = [
            avg
            for item in top_candidates
            if (avg := _multi_horizon_average(item.candidate)) is not None
        ]
    else:
        values = [float(item.candidate.row.future_return_7d) for item in top_candidates]
    if not values:
        return 0.0
    return float(np.mean(values))


def _ideal_relevances(
    query: LabeledRow,
    pool: Sequence[LabeledRow],
    *,
    search_weights: tuple[float, float, float],
) -> list[float]:
    return [
        teacher_relevance(
            query,
            candidate,
            baseline_similarity=snapshot_similarity(
                query,
                candidate,
                weights=search_weights,
            ),
        )
        for candidate in pool
    ]


def _evidence_rows(
    ranked: Sequence[HybridRankedCandidate],
    top_k: int,
    *,
    buy_threshold: float,
    sell_threshold: float,
) -> list[RetrievedEvidence]:
    rows: list[RetrievedEvidence] = []
    for item in ranked[:top_k]:
        rows.append(
            RetrievedEvidence(
                date=item.candidate.row.date,
                future_return_7d=float(item.candidate.row.future_return_7d),
                d7_label=d7_label(
                    item.candidate.row.future_return_7d,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                ),
                score=float(item.score),
                components=item.components.as_dict(),
                original_rank=item.original_rank,
            )
        )
    return rows


def _fixed_ranked(
    query: LabeledRow,
    pool: Sequence[LabeledRow],
    *,
    search_weights: tuple[float, float, float],
) -> list[HybridRankedCandidate]:
    scored = sorted(
        (
            (_fixed_score(query, candidate, search_weights), candidate)
            for candidate in pool
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    return [
        HybridRankedCandidate(
            candidate=candidate,
            score=float(score),
            components=HybridComponentScores(0.0, 0.0, 0.0, 0.0),
            original_rank=index + 1,
        )
        for index, (score, candidate) in enumerate(scored)
    ]


def _learned_ranked(
    query: LabeledRow,
    pool: Sequence[LabeledRow],
    *,
    learned_metric: LearnedDiagonalMetric,
) -> list[HybridRankedCandidate]:
    scored = sorted(
        (
            (learned_metric.score(query.blocks, candidate.blocks), candidate)
            for candidate in pool
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    return [
        HybridRankedCandidate(
            candidate=candidate,
            score=float(score),
            components=HybridComponentScores(0.0, 0.0, 0.0, 0.0),
            original_rank=index + 1,
        )
        for index, (score, candidate) in enumerate(scored)
    ]


def _with_basic_components(
    ranked: Sequence[HybridRankedCandidate],
    *,
    component_key: str,
) -> list[HybridRankedCandidate]:
    out: list[HybridRankedCandidate] = []
    for item in ranked:
        value = max(0.0, min(1.0, (float(item.score) + 1.0) / 2.0))
        components = HybridComponentScores(
            knn_market_score=value if component_key == "knn_market_score" else 0.0,
            learned_finbert_score=value if component_key == "learned_finbert_score" else 0.0,
            regime_score=value if component_key == "regime_score" else 0.0,
            signal_prior_score=value if component_key == "signal_prior_score" else 0.0,
        )
        out.append(
            HybridRankedCandidate(
                candidate=item.candidate,
                score=item.score,
                components=components,
                original_rank=item.original_rank,
            )
        )
    return out


def evaluate_ranker(
    labeled: Sequence[LabeledRow],
    *,
    method_name: str,
    ranker: Callable[[LabeledRow, Sequence[LabeledRow]], list[HybridRankedCandidate]],
    search_weights: tuple[float, float, float],
    split: str,
    top_k: int,
    buy_threshold: float,
    sell_threshold: float,
    use_production_head: bool = False,
    example_limit: int = 10,
) -> MethodEvaluation:
    examples: list[dict] = []
    hit_total = hit_count = 0
    ndcg_values: list[float] = []
    total_queries = evaluated_queries = correct = active = active_correct = evidence_ready = 0

    for query in labeled:
        if query.split != split:
            continue
        total_queries += 1
        pool = matured_pool(labeled, query, guard=True)
        ranked = ranker(query, pool)
        top_ranked = ranked[:top_k]
        if len(top_ranked) < top_k:
            continue

        evaluated_queries += 1
        evidence_ready += 1
        actual_label = d7_label(
            query.row.future_return_7d,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        predicted_return = _predicted_return_7d(
            top_ranked,
            top_k=top_k,
            use_production_head=use_production_head,
        )
        predicted_label = d7_label(
            predicted_return,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        is_correct = predicted_label == actual_label
        correct += int(is_correct)
        if predicted_label != "HOLD":
            active += 1
            active_correct += int(is_correct)

        if query.direction != 0:
            hit_total += 1
            hit = any(candidate.candidate.direction == query.direction for candidate in top_ranked)
            hit_count += int(hit)
            ideal = _ideal_relevances(query, pool, search_weights=search_weights)
            retrieved = [
                teacher_relevance(
                    query,
                    candidate.candidate,
                    baseline_similarity=snapshot_similarity(
                        query,
                        candidate.candidate,
                        weights=search_weights,
                    ),
                )
                for candidate in top_ranked
            ]
            ndcg_values.append(ndcg_at_k(retrieved, ideal, top_k))
        else:
            hit = False

        if len(examples) < example_limit:
            examples.append(
                {
                    "query_date": query.row.date,
                    "query_future_return_7d": float(query.row.future_return_7d),
                    "actual_d7_label": actual_label,
                    "predicted_return_7d": predicted_return,
                    "predicted_d7_label": predicted_label,
                    "top5_same_d7_sign": hit,
                    "evidence": [
                        asdict(row)
                        for row in _evidence_rows(
                            top_ranked,
                            top_k,
                            buy_threshold=buy_threshold,
                            sell_threshold=sell_threshold,
                        )
                    ],
                }
            )

    return MethodEvaluation(
        name=method_name,
        top5_same_d7_sign_rate=(hit_count / hit_total) if hit_total else 0.0,
        ndcg_at_5=float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        downstream_da=(correct / evaluated_queries) if evaluated_queries else 0.0,
        active_acc=(active_correct / active) if active else 0.0,
        coverage=(active / evaluated_queries) if evaluated_queries else 0.0,
        evidence_coverage=(evidence_ready / total_queries) if total_queries else 0.0,
        queries_evaluated=evaluated_queries,
        directional_queries=hit_total,
        examples=examples,
    )


def _markdown_summary(
    evaluations: Sequence[MethodEvaluation],
    *,
    config: dict[str, object],
) -> str:
    lines = [
        "# Hybrid Retrieval D7-Consistency Evaluation",
        "",
        f"- data: `{config['data']}`",
        f"- split: `{config['split']}`",
        f"- top_k: `{config['top_k']}`",
        f"- candidate_pool_size: `{config['candidate_pool_size']}`",
        f"- buy_threshold: `{config['buy_threshold']}`",
        f"- sell_threshold: `{config['sell_threshold']}`",
        f"- hybrid_weights: `{json.dumps(config['hybrid_weights'], sort_keys=True)}`",
        "",
        "| Method | Top5 same D7 sign | nDCG@5 | Downstream DA | Active Acc | Coverage | Evidence Coverage | Queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in evaluations:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.name,
                    f"{item.top5_same_d7_sign_rate:.4f}",
                    f"{item.ndcg_at_5:.4f}",
                    f"{item.downstream_da:.4f}",
                    f"{item.active_acc:.4f}",
                    f"{item.coverage:.4f}",
                    f"{item.evidence_coverage:.4f}",
                    str(item.queries_evaluated),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed, learned, and hybrid retrieval for D7-consistent StockMem evidence."
    )
    parser.add_argument("--data", default="stockmem/data/real_optimizer_finbert.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--output-dir", default="artifacts/hybrid_retrieval")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=30)
    parser.add_argument("--buy-threshold", type=float, default=2.0)
    parser.add_argument("--sell-threshold", type=float, default=2.0)
    parser.add_argument("--w-knn", type=float, default=DEFAULT_HYBRID_WEIGHTS.w_knn)
    parser.add_argument("--w-learned", type=float, default=DEFAULT_HYBRID_WEIGHTS.w_learned)
    parser.add_argument("--w-regime", type=float, default=DEFAULT_HYBRID_WEIGHTS.w_regime)
    parser.add_argument("--w-prior", type=float, default=DEFAULT_HYBRID_WEIGHTS.w_prior)
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    validate_rows(rows)
    labeled = label_rows(rows, band="fixed", fixed_band=args.buy_threshold)
    learned_metric = LearnedDiagonalMetric.load(args.artifact)
    search_weights = DEFAULT_SEARCH_WEIGHTS
    weights_path = Path(args.weights)
    if weights_path.exists():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        source = payload.get("weights", payload)
        search_weights = (
            float(source["w1_factor"]),
            float(source["w2_indicator"]),
            float(source["w3_price"]),
        )

    hybrid_weights = HybridRerankWeights(
        w_knn=args.w_knn,
        w_learned=args.w_learned,
        w_regime=args.w_regime,
        w_prior=args.w_prior,
    )

    def fixed_ranker(query: LabeledRow, pool: Sequence[LabeledRow]) -> list[HybridRankedCandidate]:
        return _with_basic_components(
            _fixed_ranked(query, pool, search_weights=search_weights),
            component_key="knn_market_score",
        )

    def learned_ranker(query: LabeledRow, pool: Sequence[LabeledRow]) -> list[HybridRankedCandidate]:
        return _with_basic_components(
            _learned_ranked(query, pool, learned_metric=learned_metric),
            component_key="learned_finbert_score",
        )

    def hybrid_ranker(query: LabeledRow, pool: Sequence[LabeledRow]) -> list[HybridRankedCandidate]:
        fixed_ranked = _fixed_ranked(query, pool, search_weights=search_weights)
        candidate_slice = fixed_ranked[: args.candidate_pool_size]
        candidate_rows = [item.candidate for item in candidate_slice]
        baseline_scores = [float(item.score) for item in candidate_slice]
        return rerank_knn_candidates(
            query,
            candidate_rows,
            learned_metric=learned_metric,
            baseline_scores=baseline_scores,
            weights=hybrid_weights,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )

    evaluations = [
        evaluate_ranker(
            labeled,
            method_name="fixed_knn",
            ranker=fixed_ranker,
            search_weights=search_weights,
            split=args.split,
            top_k=args.top_k,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        ),
        evaluate_ranker(
            labeled,
            method_name="learned_only",
            ranker=learned_ranker,
            search_weights=search_weights,
            split=args.split,
            top_k=args.top_k,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        ),
        evaluate_ranker(
            labeled,
            method_name="hybrid_reranker",
            ranker=hybrid_ranker,
            search_weights=search_weights,
            split=args.split,
            top_k=args.top_k,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        ),
        evaluate_ranker(
            labeled,
            method_name="fixed_knn_production_head",
            ranker=fixed_ranker,
            search_weights=search_weights,
            split=args.split,
            top_k=args.top_k,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            use_production_head=True,
        ),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "data": args.data,
            "artifact": args.artifact,
            "weights": args.weights,
            "split": args.split,
            "top_k": args.top_k,
            "candidate_pool_size": args.candidate_pool_size,
            "buy_threshold": args.buy_threshold,
            "sell_threshold": args.sell_threshold,
            "hybrid_weights": hybrid_weights.as_dict(),
        },
        "methods": [asdict(item) for item in evaluations],
    }
    json_path = output_dir / "d7_consistency_eval.json"
    md_path = output_dir / "d7_consistency_eval.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        _markdown_summary(evaluations, config=payload["config"]),
        encoding="utf-8",
    )

    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
