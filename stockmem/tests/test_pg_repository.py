"""Integration tests for PGRepository — require a live PostgreSQL instance.

Run with:
    DB_URL=postgresql+asyncpg://postgres:pass@localhost:5432/stockmem \
    PYTHONPATH=. pytest -q stockmem/tests/test_pg_repository.py -m integration
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from stockmem.src.models import MarketSnapshot, StockMemRecord
from stockmem.src.store.pg_repository import PGRepository

_DB_URL = os.getenv(
    "DB_URL",
    "postgresql+asyncpg://postgres:pass@localhost:5432/stockmem",
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def repo():
    """Fresh PGRepository, cleaned up after each test."""
    r = PGRepository(_DB_URL)
    await r.init()
    # wipe slate before test
    import asyncpg
    pool = r._pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM stockmem_records")
    yield r
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM stockmem_records")
    await r.close()


def _record(
    symbol: str = "BTCUSDT",
    record_date: date = date(2026, 1, 1),
    sentiment: float = 0.5,
    summary: str = "test record",
) -> StockMemRecord:
    return StockMemRecord(
        date=record_date,
        symbol=symbol,
        sentiment_score=sentiment,
        factors=["macro", "regulatory"],
        market_snapshot=MarketSnapshot(rsi=55.0, macd_hist=0.02, fear_greed_index=60.0),
        summary=summary,
        article_ids=["art-001"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_init_creates_table(repo: PGRepository):
    """init() must create the table without error."""
    count = await repo.count()
    assert count == 0


async def test_upsert_and_get_by_id(repo: PGRepository):
    rec = _record()
    rid = await repo.upsert(rec.model_copy(update={"id": "pg-001"}))
    assert rid == "pg-001"

    fetched = await repo.get("pg-001")
    assert fetched is not None
    assert fetched.symbol == "BTCUSDT"
    assert fetched.sentiment_score == 0.5


async def test_upsert_overwrites_same_date_symbol(repo: PGRepository):
    """Two upserts for the same (date, symbol) keep only the latest."""
    rec1 = _record(summary="first").model_copy(update={"id": "pg-001"})
    rec2 = _record(summary="updated").model_copy(update={"id": "pg-002"})

    await repo.upsert(rec1)
    await repo.upsert(rec2)

    count = await repo.count()
    assert count == 1

    fetched = await repo.get("pg-002")
    assert fetched is not None
    assert fetched.summary == "updated"


async def test_get_id_by_date_symbol(repo: PGRepository):
    rec = _record().model_copy(update={"id": "pg-001"})
    await repo.upsert(rec)

    found_id = await repo.get_id_by_date_symbol("2026-01-01", "BTCUSDT")
    assert found_id == "pg-001"

    missing = await repo.get_id_by_date_symbol("2026-01-01", "ETHUSDT")
    assert missing is None


async def test_get_by_date_symbol(repo: PGRepository):
    rec = _record(symbol="ETHUSDT", sentiment=0.8).model_copy(update={"id": "pg-eth"})
    await repo.upsert(rec)

    fetched = await repo.get_by_date_symbol("2026-01-01", "ETHUSDT")
    assert fetched is not None
    assert fetched.sentiment_score == 0.8

    missing = await repo.get_by_date_symbol("2026-01-02", "ETHUSDT")
    assert missing is None


async def test_list_all_returns_all_records(repo: PGRepository):
    await repo.upsert(_record(symbol="BTCUSDT", record_date=date(2026, 1, 1)).model_copy(update={"id": "a"}))
    await repo.upsert(_record(symbol="ETHUSDT", record_date=date(2026, 1, 1)).model_copy(update={"id": "b"}))
    await repo.upsert(_record(symbol="BTCUSDT", record_date=date(2026, 1, 2)).model_copy(update={"id": "c"}))

    records = await repo.list_all()
    assert len(records) == 3
    symbols = {r.symbol for r in records}
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols


async def test_get_missing_id_returns_none(repo: PGRepository):
    result = await repo.get("nonexistent-id")
    assert result is None


async def test_count(repo: PGRepository):
    assert await repo.count() == 0
    await repo.upsert(_record(symbol="BTCUSDT", record_date=date(2026, 1, 1)).model_copy(update={"id": "x1"}))
    await repo.upsert(_record(symbol="ETHUSDT", record_date=date(2026, 1, 1)).model_copy(update={"id": "x2"}))
    assert await repo.count() == 2


async def test_symbol_case_insensitive(repo: PGRepository):
    """Symbols stored as uppercase regardless of input case."""
    rec = _record(symbol="btcusdt").model_copy(update={"id": "lower-001"})
    await repo.upsert(rec)

    fetched = await repo.get_by_date_symbol("2026-01-01", "BTCUSDT")
    assert fetched is not None
    assert fetched.symbol == "BTCUSDT"
