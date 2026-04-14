from __future__ import annotations

from ..models import SimilarRecord, StockMemRecord
from .embedder import RecordEmbedder
from .index import MemoryVectorIndex


class RecordSearcher:
    """k-NN similarity search over vector index."""

    def __init__(
        self,
        embedder: RecordEmbedder,
        index: MemoryVectorIndex,
        record_cache: dict[str, StockMemRecord],
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._record_cache = record_cache

    def search(self, query: StockMemRecord, k: int = 5) -> list[SimilarRecord]:
        query_vec = self._embedder.embed(query)
        scored = self._index.search(query_vec, k)

        results: list[SimilarRecord] = []
        for item in scored:
            rec = self._record_cache.get(item.record_id)
            if rec is None:
                continue
            similarity = max(0.0, min(1.0, (item.score + 1.0) / 2.0))
            results.append(
                SimilarRecord(
                    record=rec,
                    similarity=round(similarity, 6),
                    outcome=None,
                )
            )
        return results
