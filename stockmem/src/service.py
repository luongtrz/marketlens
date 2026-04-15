from __future__ import annotations

from .config import SearchWeights
from .models import SimilarRecord, StockMemRecord
from .search.embedder import RecordEmbedder
from .search.index import MemoryVectorIndex
from .search.searcher import RecordSearcher
from .store.repository import RecordRepository
from .store.writer import RecordWriter


class StockMemService:
    def __init__(self, db_url: str, vector_backend: str, weights: SearchWeights) -> None:
        self.vector_backend = vector_backend
        self.weights = weights
        self.repository = RecordRepository(db_url)
        self.embedder = RecordEmbedder()
        self.index = MemoryVectorIndex()
        self.records_by_id: dict[str, StockMemRecord] = {}
        self.writer = RecordWriter(
            repository=self.repository,
            embedder=self.embedder,
            index=self.index,
            record_cache=self.records_by_id,
        )
        self.searcher = RecordSearcher(
            embedder=self.embedder,
            index=self.index,
            record_cache=self.records_by_id,
            weights=self.weights,
        )

    async def startup(self) -> None:
        await self.repository.init()
        records = await self.repository.list_all()
        self.records_by_id = {r.id: r for r in records if r.id is not None}
        self.writer = RecordWriter(
            repository=self.repository,
            embedder=self.embedder,
            index=self.index,
            record_cache=self.records_by_id,
        )
        self.searcher = RecordSearcher(
            embedder=self.embedder,
            index=self.index,
            record_cache=self.records_by_id,
            weights=self.weights,
        )

        self.embedder.rebuild_corpus(self.records_by_id.values())
        vectors = [(rid, self.embedder.embed(rec)) for rid, rec in self.records_by_id.items()]
        self.index.rebuild(vectors)

    async def save_record(self, record: StockMemRecord) -> str:
        return await self.writer.save(record)

    async def get_record(self, record_id: str) -> StockMemRecord | None:
        cached = self.records_by_id.get(record_id)
        if cached is not None:
            return cached
        record = await self.repository.get(record_id)
        if record is not None and record.id is not None:
            self.records_by_id[record.id] = record
        return record

    async def search(self, query: StockMemRecord, k: int = 5) -> list[SimilarRecord]:
        return self.searcher.search(query, k)
