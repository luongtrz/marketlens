from __future__ import annotations

from datetime import date
import logging

from .config import SearchWeights, load_learned_retriever_from_config
from .models import SimilarRecord, StockMemRecord
from .search.embedder import RecordEmbedder
from .search.event_memory import build_daily_event_state
from .search.index import MemoryVectorIndex
from .search.learned_metric import LearnedDiagonalMetric
from .search.searcher import RecordSearcher
from .store.base import Repository
from .store.pg_repository import PGRepository
from .store.writer import RecordWriter
from .weights_retrainer import retrain_weights, write_weights_snapshot

logger = logging.getLogger(__name__)


def _build_repository(db_url: str) -> Repository:
    return PGRepository(db_url)


class StockMemService:
    def __init__(
        self,
        db_url: str,
        vector_backend: str,
        weights: SearchWeights,
        learned_retriever_file: str | None = None,
    ) -> None:
        self.vector_backend = vector_backend
        self.weights = weights
        self.learned_retriever_file = learned_retriever_file
        self.learned_metric: LearnedDiagonalMetric | None = None
        self.repository: Repository = _build_repository(db_url)
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
            learned_metric=self.learned_metric,
        )

    async def startup(self) -> None:
        await self.repository.init()
        self.learned_metric = load_learned_retriever_from_config(self.learned_retriever_file)
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
            learned_metric=self.learned_metric,
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

    async def search(
        self,
        query: StockMemRecord,
        k: int = 5,
        before_date: date | None = None,
        retriever_type: str = "fixed_knn",
    ) -> list[SimilarRecord]:
        effective_query = query
        if query.event_state is None:
            persisted = self.records_by_id.get(query.id) if query.id else None
            if (
                persisted is not None
                and persisted.date == query.date
                and persisted.symbol.upper() == query.symbol.upper()
                and persisted.event_state is not None
            ):
                event_state = persisted.event_state
                event_vector = query.event_vector or persisted.event_vector
            else:
                normalized_symbol = query.symbol.upper()
                history = [
                    record
                    for record in self.records_by_id.values()
                    if record.symbol.upper() == normalized_symbol
                    and record.date < query.date
                ]
                event_state = build_daily_event_state(query, history)
                event_vector = query.event_vector
            effective_query = query.model_copy(
                update={
                    "event_state": event_state,
                    "event_vector": event_vector,
                }
            )
        return self.searcher.search(
            effective_query,
            k,
            before_date=before_date,
            retriever_type=retriever_type,
        )

    async def list_missing_returns(self, symbol: str | None = None) -> list[StockMemRecord]:
        return await self.repository.list_missing_returns(symbol)

    async def update_future_returns(
        self,
        record_id: str,
        future_return_1d: float | None = None,
        future_return_3d: float | None = None,
        future_return_7d: float | None = None,
        future_return_15d: float | None = None,
        future_return_30d: float | None = None,
    ) -> bool:
        ok = await self.repository.update_future_returns(
            record_id,
            future_return_1d=future_return_1d,
            future_return_3d=future_return_3d,
            future_return_7d=future_return_7d,
            future_return_15d=future_return_15d,
            future_return_30d=future_return_30d,
        )
        if ok and record_id in self.records_by_id:
            rec = self.records_by_id[record_id]
            self.records_by_id[record_id] = rec.model_copy(update={
                k: v for k, v in {
                    "future_return_1d": future_return_1d,
                    "future_return_3d": future_return_3d,
                    "future_return_7d": future_return_7d,
                    "future_return_15d": future_return_15d,
                    "future_return_30d": future_return_30d,
                }.items() if v is not None
            })
        return ok

    def set_weights(self, weights: SearchWeights) -> None:
        self.weights = weights
        self.searcher = RecordSearcher(
            embedder=self.embedder,
            index=self.index,
            record_cache=self.records_by_id,
            weights=self.weights,
            learned_metric=self.learned_metric,
        )

    async def auto_retrain_weights(
        self,
        *,
        horizon: str,
        k: int,
        warmup: int,
        trials: int,
        min_records: int,
        output_path: str,
    ) -> dict:
        records = await self.repository.list_all()
        labeled = [
            r for r in records
            if r.future_return_1d is not None
            and r.future_return_7d is not None
            and r.future_return_30d is not None
        ]
        if len(labeled) < min_records:
            raise ValueError(
                f"Insufficient labeled records: {len(labeled)} < min_records({min_records})"
            )

        new_weights, payload = retrain_weights(
            labeled,
            horizon=horizon,
            k=k,
            warmup=warmup,
            trials=trials,
        )
        self.set_weights(new_weights)
        write_weights_snapshot(output_path, payload)
        logger.info(
            "StockMem auto-retrained weights: w1=%.4f w2=%.4f w3=%.4f",
            new_weights.w1_factor,
            new_weights.w2_indicator,
            new_weights.w3_price,
        )
        return payload
