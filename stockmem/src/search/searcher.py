from __future__ import annotations

from datetime import date
import numpy as np

from ..config import SearchWeights
from ..models import SimilarRecord, StockMemRecord
from .embedder import RecordEmbedder, SplitEmbedding
from .index import MemoryVectorIndex


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
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._record_cache = record_cache
        self._weights = weights

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

    def search(
        self,
        query: StockMemRecord,
        k: int = 5,
        before_date: date | None = None,
    ) -> list[SimilarRecord]:
        scored: list[tuple[float, StockMemRecord]] = []
        query_split = self._embedder.embed_split(query)
        query_joint = self._embedder.embed(query)

        # ANN prefilter via joint vector index, then weighted rerank.
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

        for rec in candidate_records:
            if rec.symbol.upper() != query.symbol.upper():
                continue
            if before_date is not None and rec.date >= before_date:
                continue
            cand_split = self._embedder.embed_split(rec)
            score = self._weighted_score(query_split, cand_split)
            scored.append((score, rec))

        if len(scored) < k:
            seen_ids = {rec.id for _, rec in scored}
            for rec in self._record_cache.values():
                if rec.id in seen_ids:
                    continue
                if rec.symbol.upper() != query.symbol.upper():
                    continue
                if before_date is not None and rec.date >= before_date:
                    continue
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
