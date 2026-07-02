from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from stockmem.scripts.cem_dataset import LabeledRow, _regime_similarity
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


_EPS = 1e-9


@dataclass(frozen=True)
class HybridRerankWeights:
    w_knn: float
    w_learned: float
    w_regime: float
    w_prior: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.w_knn, self.w_learned, self.w_regime, self.w_prior],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Hybrid reranking weights must be finite")
        if np.any(values < 0.0):
            raise ValueError("Hybrid reranking weights must be non-negative")
        if abs(float(values.sum()) - 1.0) > 1e-6:
            raise ValueError("Hybrid reranking weights must sum to 1")

    def as_dict(self) -> dict[str, float]:
        return {
            "w_knn": float(self.w_knn),
            "w_learned": float(self.w_learned),
            "w_regime": float(self.w_regime),
            "w_prior": float(self.w_prior),
        }


@dataclass(frozen=True)
class HybridComponentScores:
    knn_market_score: float
    learned_finbert_score: float
    regime_score: float
    signal_prior_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "knn_market_score": float(self.knn_market_score),
            "learned_finbert_score": float(self.learned_finbert_score),
            "regime_score": float(self.regime_score),
            "signal_prior_score": float(self.signal_prior_score),
        }


@dataclass(frozen=True)
class HybridRankedCandidate:
    candidate: LabeledRow
    score: float
    components: HybridComponentScores
    original_rank: int


def d7_label(
    value: float,
    *,
    buy_threshold: float = 2.0,
    sell_threshold: float = 2.0,
) -> str:
    if value > buy_threshold:
        return "UP"
    if value < -sell_threshold:
        return "DOWN"
    return "HOLD"


def normalize_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Scores must be finite before normalization")
    minimum = float(arr.min())
    maximum = float(arr.max())
    if maximum - minimum <= _EPS:
        return [0.5] * arr.size
    normalized = (arr - minimum) / (maximum - minimum)
    return [float(v) for v in normalized]


def signal_prior_scores(
    candidates: Sequence[LabeledRow],
    *,
    buy_threshold: float = 2.0,
    sell_threshold: float = 2.0,
) -> list[float]:
    if not candidates:
        return []
    labels = [
        d7_label(
            candidate.row.future_return_7d,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        for candidate in candidates
    ]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = float(len(labels))
    return [counts[label] / total for label in labels]


def rerank_knn_candidates(
    query: LabeledRow,
    candidates: Sequence[LabeledRow],
    *,
    learned_metric: LearnedDiagonalMetric,
    baseline_scores: Sequence[float],
    weights: HybridRerankWeights,
    buy_threshold: float = 2.0,
    sell_threshold: float = 2.0,
) -> list[HybridRankedCandidate]:
    if len(candidates) != len(baseline_scores):
        raise ValueError("Candidate count must match baseline score count")
    if not candidates:
        return []

    learned_scores = [
        learned_metric.score(query.blocks, candidate.blocks)
        for candidate in candidates
    ]
    regime_scores = [
        _regime_similarity(query, candidate)
        for candidate in candidates
    ]
    prior_scores = signal_prior_scores(
        candidates,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    normalized_knn = normalize_scores(baseline_scores)
    normalized_learned = normalize_scores(learned_scores)
    normalized_regime = normalize_scores(regime_scores)
    normalized_prior = normalize_scores(prior_scores)

    ranked: list[HybridRankedCandidate] = []
    for index, candidate in enumerate(candidates):
        components = HybridComponentScores(
            knn_market_score=normalized_knn[index],
            learned_finbert_score=normalized_learned[index],
            regime_score=normalized_regime[index],
            signal_prior_score=normalized_prior[index],
        )
        score = (
            weights.w_knn * components.knn_market_score
            + weights.w_learned * components.learned_finbert_score
            + weights.w_regime * components.regime_score
            + weights.w_prior * components.signal_prior_score
        )
        ranked.append(
            HybridRankedCandidate(
                candidate=candidate,
                score=float(score),
                components=components,
                original_rank=index + 1,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.original_rank))
    return ranked
