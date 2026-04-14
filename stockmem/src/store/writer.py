from __future__ import annotations

import uuid

from ..models import StockMemRecord
from ..search.embedder import RecordEmbedder
from ..search.index import MemoryVectorIndex
from .repository import RecordRepository


class RecordWriter:
    def __init__(
        self,
        repository: RecordRepository,
        embedder: RecordEmbedder,
        index: MemoryVectorIndex,
        record_cache: dict[str, StockMemRecord],
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._index = index
        self._record_cache = record_cache

    async def save(self, record: StockMemRecord) -> str:
        """
        Persists record to relational DB.
        Also triggers embedder to compute and store vector in vector index.
        Returns record id (UUID).
        """
        existing_id = await self._repository.get_id_by_date_symbol(
            record.date.isoformat(),
            record.symbol,
        )
        rid = record.id or existing_id or str(uuid.uuid4())
        normalized_symbol = record.symbol.upper()
        to_save = record.model_copy(update={"id": rid, "symbol": normalized_symbol})

        # Keep cache aligned with one-record-per-day/symbol semantics.
        for cached_id, cached_record in list(self._record_cache.items()):
            if (
                cached_id != rid
                and cached_record.date == to_save.date
                and cached_record.symbol.upper() == normalized_symbol
            ):
                del self._record_cache[cached_id]

        await self._repository.upsert(to_save)
        self._record_cache[rid] = to_save

        # TF-IDF depends on corpus statistics, so refresh embedding stats and index.
        all_records = list(self._record_cache.values())
        self._embedder.rebuild_corpus(all_records)
        vectors = [(r.id, self._embedder.embed(r)) for r in all_records if r.id is not None]
        self._index.rebuild(vectors)

        return rid
