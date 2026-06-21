from __future__ import annotations

import asyncio
from datetime import date

from stockmem.src.models import MarketSnapshot, StockMemRecord
from stockmem.src.search.embedder import RecordEmbedder
from stockmem.src.search.index import MemoryVectorIndex
from stockmem.src.store.pg_repository import PGRepository
from stockmem.src.store.reader import RecordReader
from stockmem.src.store.writer import RecordWriter

PG_URL = "postgresql+asyncpg://postgres:pass@localhost:5432/postgres"
_TEST_DATES = ["2099-12-29", "2099-12-30", "2099-12-31"]


async def _cleanup(repo: PGRepository) -> None:
    pool = repo._require_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM stockmem_records WHERE record_date = ANY($1) AND symbol = $2",
            _TEST_DATES,
            "BTC",
        )
    await repo.close()


def test_store_save_and_read_roundtrip() -> None:
    async def scenario() -> None:
        repo = PGRepository(PG_URL)
        await repo.init()
        try:
            cache: dict[str, StockMemRecord] = {}
            writer = RecordWriter(repo, RecordEmbedder(), MemoryVectorIndex(), cache)
            reader = RecordReader(repo)

            rec1 = StockMemRecord(
                date=date(2099, 12, 31),
                symbol="btc",
                sentiment_score=0.3,
                factors=["macro"],
                market_snapshot=MarketSnapshot(rsi=52.0, macd_hist=0.01),
                summary="first",
                article_ids=["n1"],
                future_return_1d=1.11,
                future_return_3d=2.00,
                future_return_7d=2.22,
                future_return_15d=3.00,
                future_return_30d=3.33,
            )
            # Upsert again with None returns — upsert must preserve existing values
            rec2 = rec1.model_copy(
                update={
                    "summary": "updated",
                    "symbol": "BTC",
                    "future_return_1d": None,
                    "future_return_3d": None,
                    "future_return_7d": None,
                    "future_return_15d": None,
                    "future_return_30d": None,
                }
            )

            rid1 = await writer.save(rec1)
            rid2 = await writer.save(rec2)

            by_id = await reader.get_by_id(rid2)
            by_date = await reader.get_by_date(date(2099, 12, 31), "BTC")
            count = await repo.count()

            assert rid1 == rid2
            assert by_id is not None
            assert by_date is not None
            assert by_id.summary == "updated"
            assert by_date.id == rid2
            assert by_date.future_return_1d == 1.11
            assert by_date.future_return_3d == 2.00
            assert by_date.future_return_7d == 2.22
            assert by_date.future_return_15d == 3.00
            assert by_date.future_return_30d == 3.33
            assert by_date.event_state is not None
            assert by_date.event_state.article_count == 1
            assert count >= 1
        finally:
            await _cleanup(repo)

    asyncio.run(scenario())


def test_update_future_returns_all_horizons() -> None:
    async def scenario() -> None:
        repo = PGRepository(PG_URL)
        await repo.init()
        try:
            cache: dict[str, StockMemRecord] = {}
            writer = RecordWriter(repo, RecordEmbedder(), MemoryVectorIndex(), cache)
            rec = StockMemRecord(
                date=date(2099, 12, 30),
                symbol="BTC",
                sentiment_score=0.0,
                factors=[],
                market_snapshot=MarketSnapshot(rsi=50.0),
                summary="returns-test",
                article_ids=[],
            )
            rid = await writer.save(rec)

            ok = await repo.update_future_returns(
                rid,
                future_return_1d=1.0,
                future_return_3d=3.0,
                future_return_7d=7.0,
                future_return_15d=15.0,
                future_return_30d=30.0,
            )
            assert ok

            loaded = await repo.get(rid)
            assert loaded is not None
            assert loaded.future_return_1d == 1.0
            assert loaded.future_return_3d == 3.0
            assert loaded.future_return_7d == 7.0
            assert loaded.future_return_15d == 15.0
            assert loaded.future_return_30d == 30.0
        finally:
            await _cleanup(repo)

    asyncio.run(scenario())


def test_list_missing_returns_detects_all_horizons() -> None:
    async def scenario() -> None:
        repo = PGRepository(PG_URL)
        await repo.init()
        try:
            cache: dict[str, StockMemRecord] = {}
            writer = RecordWriter(repo, RecordEmbedder(), MemoryVectorIndex(), cache)
            rec = StockMemRecord(
                date=date(2099, 12, 29),
                symbol="BTC",
                sentiment_score=0.0,
                factors=[],
                market_snapshot=MarketSnapshot(rsi=50.0),
                summary="missing-test",
                article_ids=[],
            )
            rid = await writer.save(rec)

            # Fill only 1d and 7d — 3d, 15d, 30d must still show as missing
            await repo.update_future_returns(rid, future_return_1d=1.0, future_return_7d=7.0)

            missing = await repo.list_missing_returns(symbol="BTC")
            missing_ids = [r.id for r in missing]
            assert rid in missing_ids
        finally:
            await _cleanup(repo)

    asyncio.run(scenario())
