from __future__ import annotations

import uuid

from ..models import StockMemRecord
from ..search.embedder import RecordEmbedder
from ..search.event_memory import build_daily_event_state
from ..search.index import MemoryVectorIndex
from .base import Repository


class RecordWriter:
    def __init__(
        self,
        repository: Repository,
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
        history = [
            cached
            for cached in self._record_cache.values()
            if cached.symbol.upper() == normalized_symbol and cached.date < record.date
        ]
        base_record = record.model_copy(update={"id": rid, "symbol": normalized_symbol})
        event_state = build_daily_event_state(base_record, history)
        to_save = base_record.model_copy(update={"event_state": event_state})

        # Keep cache aligned with one-record-per-day/symbol semantics.
        removed_ids: list[str] = []
        for cached_id, cached_record in list(self._record_cache.items()):
            if (
                cached_id != rid
                and cached_record.date == to_save.date
                and cached_record.symbol.upper() == normalized_symbol
            ):
                del self._record_cache[cached_id]
                removed_ids.append(cached_id)

        await self._repository.upsert(to_save)
        self._record_cache[rid] = to_save

        # Incremental update to avoid O(n) rebuild on every save.
        # Note: if we replaced an existing key or removed ids, index is patched accordingly.
        for old_id in removed_ids:
            self._index.remove(old_id)
        self._embedder.update_corpus_with_record(to_save)
        self._index.upsert(rid, self._embedder.embed(to_save))

        return rid
