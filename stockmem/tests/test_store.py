from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

from stockmem.src.models import MarketSnapshot, StockMemRecord
from stockmem.src.search.embedder import RecordEmbedder
from stockmem.src.search.index import MemoryVectorIndex
from stockmem.src.store.reader import RecordReader
from stockmem.src.store.repository import RecordRepository
from stockmem.src.store.writer import RecordWriter


def test_store_save_and_read_roundtrip() -> None:
    db_path = Path("test_store.db")
    if db_path.exists():
        db_path.unlink()

    async def scenario() -> None:
        repo = RecordRepository("sqlite+aiosqlite:///test_store.db")
        await repo.init()

        cache: dict[str, StockMemRecord] = {}
        writer = RecordWriter(repo, RecordEmbedder(), MemoryVectorIndex(), cache)
        reader = RecordReader(repo)

        rec1 = StockMemRecord(
            date=date(2026, 4, 14),
            symbol="btc",
            sentiment_score=0.3,
            factors=["macro"],
            market_snapshot=MarketSnapshot(rsi=52.0, macd_hist=0.01),
            summary="first",
            article_ids=["n1"],
            future_return_1d=1.11,
            future_return_7d=2.22,
            future_return_30d=3.33,
        )
        rec2 = rec1.model_copy(
            update={
                "summary": "updated",
                "symbol": "BTC",
                "future_return_1d": None,
                "future_return_7d": None,
                "future_return_30d": None,
            }
        )

        rid1 = await writer.save(rec1)
        rid2 = await writer.save(rec2)

        by_id = await reader.get_by_id(rid2)
        by_date = await reader.get_by_date(date(2026, 4, 14), "BTC")
        count = await repo.count()

        assert rid1 == rid2
        assert by_id is not None
        assert by_date is not None
        assert by_id.summary == "updated"
        assert by_date.id == rid2
        assert by_date.future_return_1d == 1.11
        assert by_date.future_return_7d == 2.22
        assert by_date.future_return_30d == 3.33
        assert count == 1

    asyncio.run(scenario())
    if db_path.exists():
        db_path.unlink()
