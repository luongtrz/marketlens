from __future__ import annotations

from datetime import date
import numpy as np

from ..config import SearchWeights
from ..models import SimilarRecord, StockMemRecord
from .embedder import RecordEmbedder, SplitEmbedding
from .index import MemoryVectorIndex
from .learned_metric import LearnedDiagonalMetric

# Regime bonus added to the weighted score when query and candidate are in the
# same market regime. Opposite regimes get a symmetric penalty.
# With max weighted score = 1.0, ±0.15 shifts priority meaningfully without dominating.
_REGIME_SAME_BONUS: float = 0.15
_REGIME_OPP_PENALTY: float = -0.15


def _get_regime(record: StockMemRecord) -> str:
    """Classify record's market regime as 'bull', 'bear', or 'neutral'.

    Uses 14-day price return from recent_candles and RSI as secondary signal.
    Regime-aware search prevents matching 2022 bear cases with 2024 bull cases.
    """
    candles = list(
        getattr(record.market_snapshot, "recent_candles", None)
        or getattr(record.market_snapshot, "candles", None)
        or []
    )
    ret_14d = 0.0
    if len(candles) >= 15:
        close_now = getattr(candles[-1], "close", None)
        close_14d = getattr(candles[-15], "close", None)
        if close_now and close_14d and float(close_14d) > 0:
            ret_14d = (float(close_now) - float(close_14d)) / float(close_14d) * 100.0

    indicators = getattr(record.market_snapshot, "indicators", None) or {}
    rsi = float(indicators.get("rsi") or getattr(record.market_snapshot, "rsi", None) or 50)

    if ret_14d < -5.0 or (ret_14d < -2.0 and rsi < 45):
        return "bear"
    if ret_14d > 5.0 or (ret_14d > 2.0 and rsi > 55):
        return "bull"
    return "neutral"


class RecordSearcher:
    """
    Weighted similarity search:
        score = w1 * sim(factor) + w2 * sim(indicator) + w3 * sim(price)

    All three sub-vectors are L2-normalized by the embedder, so each sim term
    equals cosine similarity in [-1, 1]. Final similarity is mapped to [0, 1].
    """

    def __init__(
        self,
        embedder: RecordEmbedder,
        index: MemoryVectorIndex,
        record_cache: dict[str, StockMemRecord],
        weights: SearchWeights,
        learned_metric: LearnedDiagonalMetric | None = None,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._record_cache = record_cache
        self._weights = weights
        self._learned_metric = learned_metric

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape[0] != b.shape[0]:
            return 0.0
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a <= 1e-12 and norm_b <= 1e-12:
            return 1.0  # both zero vectors → identical
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return 0.0
        return float(np.dot(a, b))

    def _weighted_score(self, query: SplitEmbedding, candidate: SplitEmbedding) -> float:
        sim_factor = self._cosine(query.factor_vec, candidate.factor_vec)
        sim_indicator = self._cosine(query.indicator_vec, candidate.indicator_vec)
        sim_price = self._cosine(query.price_vec, candidate.price_vec)
        return (
            self._weights.w1_factor * sim_factor
            + self._weights.w2_indicator * sim_indicator
            + self._weights.w3_price * sim_price
        )

    def _score(
        self,
        query: SplitEmbedding,
        candidate: SplitEmbedding,
        retriever_type: str,
    ) -> float:
        if retriever_type == "learned_linear" and self._learned_metric is not None:
            return self._learned_metric.score_split(query, candidate)
        return self._weighted_score(query, candidate)

    @staticmethod
    def _event_match(query: SplitEmbedding, candidate: SplitEmbedding) -> float:
        if np.linalg.norm(query.event_vec) <= 1e-12:
            return 0.0
        if np.linalg.norm(candidate.event_vec) <= 1e-12:
            return 0.0
        return float(np.dot(query.event_vec, candidate.event_vec))

    def search(
        self,
        query: StockMemRecord,
        k: int = 5,
        before_date: date | None = None,
        retriever_type: str = "fixed_knn",
    ) -> list[SimilarRecord]:
        scored: list[tuple[float, StockMemRecord, float]] = []
        query_split = self._embedder.embed_split(query)
        use_learned = retriever_type == "learned_linear" and self._learned_metric is not None
        if use_learned:
            # Exact scan keeps production behavior aligned with the offline learned-metric evaluation.
            candidate_records = list(self._record_cache.values())
        else:
            query_joint = self._embedder.embed(query)
            pre_k = max(k * 30, 300)
            pre_candidates = self._index.search(query_joint, pre_k)
            candidate_ids = [c.record_id for c in pre_candidates]
            if not candidate_ids:
                candidate_records = list(self._record_cache.values())
            else:
                candidate_records = [
                    self._record_cache[rid]
                    for rid in candidate_ids
                    if rid in self._record_cache
                ]

        query_regime = _get_regime(query)

        for rec in candidate_records:
            if rec.symbol.upper() != query.symbol.upper():
                continue
            if before_date is not None and rec.date >= before_date:
                continue
            cand_split = self._embedder.embed_split(rec)
            score = self._score(query_split, cand_split, retriever_type)
            cand_regime = _get_regime(rec)
            if cand_regime == query_regime:
                score += _REGIME_SAME_BONUS
            elif query_regime != "neutral" and cand_regime != "neutral":
                score += _REGIME_OPP_PENALTY
            scored.append((score, rec, self._event_match(query_split, cand_split)))

        if len(scored) < k:
            seen_ids = {rec.id for _, rec, _ in scored}
            for rec in self._record_cache.values():
                if rec.id in seen_ids:
                    continue
                if rec.symbol.upper() != query.symbol.upper():
                    continue
                if before_date is not None and rec.date >= before_date:
                    continue
                cand_split = self._embedder.embed_split(rec)
                score = self._score(query_split, cand_split, retriever_type)
                cand_regime = _get_regime(rec)
                if cand_regime == query_regime:
                    score += _REGIME_SAME_BONUS
                elif query_regime != "neutral" and cand_regime != "neutral":
                    score += _REGIME_OPP_PENALTY
                scored.append((score, rec, self._event_match(query_split, cand_split)))

        scored.sort(key=lambda x: x[0], reverse=True)
        k_eff = max(1, min(k, len(scored)))

        results: list[SimilarRecord] = []
        retriever_version = (
            self._learned_metric.version
            if use_learned and self._learned_metric is not None
            else "fixed_knn_v1"
        )
        for score, rec, event_similarity in scored[:k_eff]:
            similarity = max(0.0, min(1.0, (score + 1.0) / 2.0))
            results.append(
                SimilarRecord(
                    record=rec,
                    similarity=round(similarity, 6),
                    outcome=None,
                    event_match={
                        "event_vector_cosine": round(event_similarity, 6),
                    },
                    retriever_version=retriever_version,
                )
            )
        return results
