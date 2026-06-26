from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from stockmem.scripts.cem_dataset import (
    LabeledRow,
    hybrid_selection_score,
    is_mature,
    label_rows,
    mine_candidates,
    ndcg_at_k,
    teacher_relevance,
)
from stockmem.scripts.evaluate_retriever import _zero_block_scale, evaluate
from stockmem.scripts.optimize_weights import Row
from stockmem.scripts.optimize_weights import _evaluate_query_against_pool
from stockmem.scripts.train_learned_retriever import (
    BLOCK_DIMS,
    info_nce_loss_and_grad,
)
from stockmem.src.search.embedder import SplitEmbedding
from stockmem.src.search.learned_metric import LearnedDiagonalMetric, load_learned_metric


def _labeled(
    day: str,
    vector: np.ndarray,
    direction: int,
    *,
    split: str = "train",
    future_return: float | None = None,
    volatility: float = 2.0,
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
        future_return_1d=0.0,
        future_return_7d=3.0 * direction if future_return is None else future_return,
        future_return_30d=0.0,
    )
    return LabeledRow(
        row=row,
        parsed_date=date.fromisoformat(day),
        split=split,
        direction=direction,  # type: ignore[arg-type]
        band_value=0.01,
        causal_volatility=volatility,
    )


def test_maturity_uses_calendar_days() -> None:
    assert is_mature(date(2025, 1, 1), date(2025, 1, 8))
    assert not is_mature(date(2025, 1, 2), date(2025, 1, 8))


def test_volatility_band_uses_fixed_fallback_before_history_matures() -> None:
    row = replace(_labeled("2023-01-01", np.array([1.0, 0.0]), 1).row, future_return_7d=0.5)
    labeled = label_rows([row], band="0.5sigma", fixed_band=1.0)
    assert labeled[0].band_value == 1.0
    assert labeled[0].direction == 0


def test_evaluation_includes_embargo_rows_after_they_mature() -> None:
    old_negative = _labeled("2025-06-20", np.array([0.5, 0.0]), -1, split="val")
    embargo_positive = _labeled("2025-06-25", np.array([1.0, 0.0]), 1, split="embargo")
    query = _labeled("2025-07-10", np.array([1.0, 0.0]), 1, split="test")

    result = evaluate(
        [old_negative, embargo_positive, query],
        lambda left, right: float(np.dot(left.row.indicator_vec, right.row.indicator_vec)),
        k=1,
        guard=True,
    )

    assert result.metrics["da"] == 1.0


def test_hit_at_5_is_independent_of_downstream_k() -> None:
    candidates = [
        _labeled("2025-06-01", np.array([1.0, 0.0]), -1, split="val"),
        _labeled("2025-06-02", np.array([0.9, 0.1]), 1, split="val"),
    ]
    query = _labeled("2025-07-10", np.array([1.0, 0.0]), 1, split="test")

    result = evaluate(
        [*candidates, query],
        lambda left, right: float(np.dot(left.row.indicator_vec, right.row.indicator_vec)),
        k=1,
        guard=True,
    )

    assert result.metrics["da"] == 0.0
    assert result.metrics["hit_at_5"] == 1.0


def test_evaluation_uses_thresholded_signals_and_true_coverage() -> None:
    buy_neighbor = _labeled("2025-06-01", np.array([1.0, 0.0]), 1, split="val")
    hold_neighbor = _labeled("2025-06-02", np.array([0.0, 1.0]), 0, split="val")
    hold_neighbor = replace(
        hold_neighbor,
        row=replace(hold_neighbor.row, future_return_7d=0.5),
    )
    buy_query = _labeled("2025-07-10", np.array([1.0, 0.0]), 1, split="test")
    hold_query = _labeled("2025-07-11", np.array([0.0, 1.0]), 0, split="test")
    hold_query = replace(hold_query, row=replace(hold_query.row, future_return_7d=1.0))

    result = evaluate(
        [buy_neighbor, hold_neighbor, buy_query, hold_query],
        lambda left, right: float(np.dot(left.row.indicator_vec, right.row.indicator_vec)),
        k=1,
        guard=True,
        buy_threshold=2.0,
        sell_threshold=2.0,
    )

    assert result.metrics["coverage"] == 0.5
    assert result.metrics["buy_da"] == 1.0
    assert result.metrics["hold_da"] == 1.0
    assert result.metrics["da"] == 1.0


def test_empty_learned_artifact_falls_back_to_fixed(tmp_path) -> None:
    artifact = tmp_path / "learned.json"
    artifact.write_text("", encoding="utf-8")
    assert load_learned_metric(artifact) is None


@pytest.mark.parametrize(
    ("diagonal", "scales"),
    [
        (np.array([np.nan, 1.0]), np.array([1.0])),
        (np.ones(2), np.array([np.inf])),
        (np.ones(2), np.array([0.0])),
    ],
)
def test_learned_metric_rejects_invalid_artifact_parameters(
    diagonal: np.ndarray,
    scales: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        LearnedDiagonalMetric(
            block_dims=(2, 1, 1),
            diagonal=np.concatenate([diagonal, np.ones(2)]),
            block_scales=np.concatenate([scales, np.zeros(2)]),
        )


def test_ablation_preserves_total_score_scale() -> None:
    scales = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    ablated = _zero_block_scale(scales, 1)
    assert ablated[1] == 0.0
    assert ablated.sum() == pytest.approx(scales.sum())


def test_optimizer_excludes_unmatured_candidate_outcomes() -> None:
    query = _labeled("2025-01-08", np.array([1.0, 0.0]), 1).row
    matured = _labeled("2025-01-01", np.array([0.0, 1.0]), 1).row
    immature = _labeled("2025-01-07", np.array([1.0, 0.0]), -1).row
    guarded, _ = _evaluate_query_against_pool(
        query,
        [matured, immature],
        0.0,
        1.0,
        0.0,
        1,
        "7d",
    )
    leaky, _ = _evaluate_query_against_pool(
        query,
        [matured, immature],
        0.0,
        1.0,
        0.0,
        1,
        "7d",
        maturity_guard=False,
    )
    assert guarded == 1
    assert leaky == 0


def test_identity_diagonal_reproduces_weighted_block_cosine() -> None:
    weights = np.array([0.54, 0.31, 0.15], dtype=np.float64)
    query = SplitEmbedding(
        event_vec=np.zeros(85, dtype=np.float32),
        factor_vec=np.array([1.0, 0.0], dtype=np.float32),
        indicator_vec=np.array([0.6, 0.8], dtype=np.float32),
        price_vec=np.array([0.0, 1.0], dtype=np.float32),
    )
    candidate = SplitEmbedding(
        event_vec=np.zeros(85, dtype=np.float32),
        factor_vec=np.array([0.0, 1.0], dtype=np.float32),
        indicator_vec=np.array([1.0, 0.0], dtype=np.float32),
        price_vec=np.array([0.0, 1.0], dtype=np.float32),
    )
    metric = LearnedDiagonalMetric(
        block_dims=(2, 2, 2),
        diagonal=np.ones(6, dtype=np.float64),
        block_scales=weights,
    )
    expected = (
        weights[0] * float(np.dot(query.factor_vec, candidate.factor_vec))
        + weights[1] * float(np.dot(query.indicator_vec, candidate.indicator_vec))
        + weights[2] * float(np.dot(query.price_vec, candidate.price_vec))
    )
    assert metric.score_split(query, candidate) == pytest.approx(expected, abs=1e-8)


def test_info_nce_gradient_reduces_synthetic_loss() -> None:
    anchor = _labeled("2024-01-20", np.array([1.0, 1.0]), 1)
    positive = _labeled("2024-01-01", np.array([1.0, 0.1]), 1)
    negative = _labeled("2024-01-02", np.array([0.1, 1.0]), -1)
    diagonal = np.ones(sum(BLOCK_DIMS), dtype=np.float64)
    scales = np.array([0.1, 0.8, 0.1], dtype=np.float64)
    before, grad_d, grad_s = info_nce_loss_and_grad(
        anchor,
        [positive, negative],
        diagonal,
        scales,
        temperature=0.2,
        ridge=0.0,
    )
    diagonal = np.clip(diagonal - 0.1 * grad_d, 0.05, None)
    scales = np.clip(scales - 0.05 * grad_s, 1e-4, None)
    scales /= scales.sum()
    after, _, _ = info_nce_loss_and_grad(
        anchor,
        [positive, negative],
        diagonal,
        scales,
        temperature=0.2,
        ridge=0.0,
    )
    assert after < before


def test_teacher_relevance_prefers_matching_outcome_and_volatility() -> None:
    anchor = _labeled(
        "2024-02-01",
        np.array([1.0, 0.0]),
        1,
        future_return=6.0,
        volatility=4.0,
    )
    matched = _labeled(
        "2024-01-01",
        np.array([0.8, 0.2]),
        1,
        future_return=5.5,
        volatility=4.2,
    )
    mismatched = _labeled(
        "2024-01-02",
        np.array([1.0, 0.0]),
        1,
        future_return=15.0,
        volatility=12.0,
    )

    assert teacher_relevance(
        anchor,
        matched,
        baseline_similarity=0.8,
    ) > teacher_relevance(
        anchor,
        mismatched,
        baseline_similarity=1.0,
    )


def test_mining_retains_flat_candidates_and_uses_current_metric_for_hardness() -> None:
    anchor = _labeled("2024-02-01", np.array([1.0, 0.0]), 1)
    positives = [
        _labeled("2024-01-01", np.array([0.9, 0.1]), 1),
        _labeled("2024-01-02", np.array([0.8, 0.2]), 1),
    ]
    fixed_hardest = _labeled("2024-01-03", np.array([1.0, 0.0]), -1)
    learned_hardest = _labeled("2024-01-04", np.array([0.9, 0.1]), -1)
    flat = _labeled("2024-01-05", np.array([0.95, 0.05]), 0)

    mined = mine_candidates(
        anchor,
        [*positives, fixed_hardest, learned_hardest, flat],
        weights=(0.0, 1.0, 0.0),
        hard_negs=1,
        positive_count=2,
        flat_negs=1,
        learned_score=lambda _anchor, candidate: (
            10.0 if candidate.row.date == learned_hardest.row.date else 0.0
        ),
    )

    assert mined is not None
    assert len(mined.positives) == 2
    assert mined.negatives[0] is learned_hardest
    assert flat in mined.negatives


def test_soft_teacher_targets_reduce_distillation_loss() -> None:
    anchor = _labeled("2024-02-01", np.array([1.0, 1.0]), 1)
    best_positive = _labeled("2024-01-01", np.array([1.0, 0.1]), 1)
    weaker_positive = _labeled("2024-01-02", np.array([0.7, 0.7]), 1)
    negative = _labeled("2024-01-03", np.array([0.1, 1.0]), -1)
    diagonal = np.ones(sum(BLOCK_DIMS), dtype=np.float64)
    scales = np.array([0.1, 0.8, 0.1], dtype=np.float64)

    before, grad_d, grad_s = info_nce_loss_and_grad(
        anchor,
        [best_positive, weaker_positive, negative],
        diagonal,
        scales,
        temperature=0.2,
        teacher_temperature=0.1,
        ridge=0.0,
        positive_weights=[0.9, 0.5],
    )
    updated_d = np.clip(diagonal - 0.05 * grad_d, 0.05, None)
    updated_s = np.clip(scales - 0.02 * grad_s, 1e-4, None)
    updated_s /= updated_s.sum()
    after, _, _ = info_nce_loss_and_grad(
        anchor,
        [best_positive, weaker_positive, negative],
        updated_d,
        updated_s,
        temperature=0.2,
        teacher_temperature=0.1,
        ridge=0.0,
        positive_weights=[0.9, 0.5],
    )

    assert after < before


def test_ndcg_at_k_rewards_better_ranking_order() -> None:
    ranked_good = [0.9, 0.6, 0.1]
    ranked_bad = [0.1, 0.6, 0.9]
    ideal = [0.9, 0.6, 0.1]

    assert ndcg_at_k(ranked_good, ideal, 3) > ndcg_at_k(ranked_bad, ideal, 3)
    assert ndcg_at_k(ranked_good, ideal, 3) <= 1.0


def test_ndcg_at_k_uses_global_top_k_ideal_not_prefix() -> None:
    ranked = [0.8, 0.7, 0.6]
    unsorted_pool = [0.1, 0.8, 0.2, 0.7, 0.6]

    assert ndcg_at_k(ranked, unsorted_pool, 3) == pytest.approx(1.0)


def test_hybrid_selection_score_balances_ndcg_and_combined() -> None:
    stronger_ranking = hybrid_selection_score(0.9, 0.25)
    stronger_trading = hybrid_selection_score(0.7, 0.45)

    assert stronger_ranking != pytest.approx(stronger_trading)
    assert hybrid_selection_score(0.9, 0.45) > stronger_ranking
