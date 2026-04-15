from __future__ import annotations

import numpy as np

from ..config import SearchWeights
from ..models import SimilarRecord, StockMemRecord
from .embedder import RecordEmbedder, SplitEmbedding
from .index import MemoryVectorIndex


class RecordSearcher:
    """k-NN similarity search over vector index."""

    def __init__(
        self,
        embedder: RecordEmbedder,
        index: MemoryVectorIndex,
        record_cache: dict[str, StockMemRecord],
        weights: SearchWeights,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._record_cache = record_cache
        self._weights = weights

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape[0] != b.shape[0]:
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

    def search(self, query: StockMemRecord, k: int = 5) -> list[SimilarRecord]:
        _ = self._index  # preserved for API/backend compatibility
        scored: list[tuple[float, StockMemRecord]] = []
        query_split = self._embedder.embed_split(query)
        for rec in self._record_cache.values():
            cand_split = self._embedder.embed_split(rec)
            score = self._weighted_score(query_split, cand_split)
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        k_eff = max(1, min(k, len(scored)))

        results: list[SimilarRecord] = []
        for score, rec in scored[:k_eff]:
            similarity = max(0.0, min(1.0, (score + 1.0) / 2.0))
            results.append(
                SimilarRecord(
                    record=rec,
                    similarity=round(similarity, 6),
                    outcome=None,
                )
            )
        return results
