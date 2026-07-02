from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from stockmem.scripts.cem_dataset import LabeledRow
from stockmem.scripts.experimental.evaluate_hybrid_retrieval import (
    _fixed_ranked,
    _with_basic_components,
    evaluate_ranker,
)
from stockmem.scripts.experimental.hybrid_reranking import (
    HybridRerankWeights,
    d7_label,
    rerank_knn_candidates,
)
from stockmem.scripts.optimize_weights import Row
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


def _labeled(
    day: str,
    vector: np.ndarray,
    future_return_7d: float,
    *,
    split: str,
) -> LabeledRow:
    factor = np.zeros(75, dtype=np.float32)
    indicator = np.zeros(5, dtype=np.float32)
    price = np.zeros(60, dtype=np.float32)
    indicator[: vector.size] = vector
    row = Row(
        date=day,
        factor_vec=factor,
        indicator_vec=indicator,
        price_vec=price,
        future_return_1d=future_return_7d,
        future_return_7d=future_return_7d,
        future_return_30d=future_return_7d,
    )
    direction = 0
    if future_return_7d > 2.0:
        direction = 1
    elif future_return_7d < -2.0:
        direction = -1
    return LabeledRow(
        row=row,
        parsed_date=date.fromisoformat(day),
        split=split,
        direction=direction,  # type: ignore[arg-type]
        band_value=2.0,
        causal_volatility=2.0,
    )


def test_d7_label_threshold_edges() -> None:
    assert d7_label(2.0) == "HOLD"
    assert d7_label(-2.0) == "HOLD"
    assert d7_label(2.01) == "UP"
    assert d7_label(-2.01) == "DOWN"


def test_hybrid_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        HybridRerankWeights(0.5, 0.3, 0.2, 0.1)


def test_hybrid_reranker_stays_within_knn_candidate_pool() -> None:
    query = _labeled("2025-07-10", np.array([1.0, 0.0], dtype=np.float32), 4.0, split="test")
    candidate_a = _labeled("2025-06-01", np.array([1.0, 0.0], dtype=np.float32), 5.0, split="train")
    candidate_b = _labeled("2025-06-02", np.array([0.8, 0.2], dtype=np.float32), 4.5, split="train")
    candidate_c = _labeled("2025-06-03", np.array([0.0, 1.0], dtype=np.float32), -5.0, split="train")
    metric = LearnedDiagonalMetric(
        block_dims=(75, 5, 60),
        diagonal=np.ones(140, dtype=np.float64),
        block_scales=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )

    ranked = rerank_knn_candidates(
        query,
        [candidate_a, candidate_b],
        learned_metric=metric,
        baseline_scores=[0.95, 0.90],
        weights=HybridRerankWeights(0.5, 0.3, 0.1, 0.1),
    )

    assert {item.candidate.row.date for item in ranked} == {"2025-06-01", "2025-06-02"}
    assert candidate_c.row.date not in {item.candidate.row.date for item in ranked}


def test_evaluate_ranker_computes_d7_metrics_on_small_fixture() -> None:
    candidate_up = _labeled("2024-12-01", np.array([1.0, 0.0], dtype=np.float32), 4.0, split="train")
    candidate_hold = _labeled("2024-12-02", np.array([0.0, 1.0], dtype=np.float32), 0.5, split="train")
    query_up = _labeled("2025-07-10", np.array([1.0, 0.0], dtype=np.float32), 5.0, split="test")
    query_hold = _labeled("2025-07-11", np.array([0.0, 1.0], dtype=np.float32), 0.1, split="test")
    labeled = [candidate_up, candidate_hold, query_up, query_hold]

    def ranker(query: LabeledRow, pool: list[LabeledRow]) -> list:
        return _with_basic_components(
            _fixed_ranked(query, pool, search_weights=(0.0, 1.0, 0.0)),
            component_key="knn_market_score",
        )

    result = evaluate_ranker(
        labeled,
        method_name="fixed_knn",
        ranker=ranker,
        search_weights=(0.0, 1.0, 0.0),
        split="test",
        top_k=1,
        buy_threshold=2.0,
        sell_threshold=2.0,
    )

    assert result.top5_same_d7_sign_rate == 1.0
    assert result.ndcg_at_5 > 0.0
    assert result.evidence_coverage == 1.0
    assert result.coverage == 0.5
    assert result.downstream_da == 1.0
