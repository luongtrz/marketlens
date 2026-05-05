"""PostgreSQL repository using asyncpg."""

from __future__ import annotations

import json
import logging

import asyncpg

from ..models import StockMemRecord

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS stockmem_records (
    id TEXT PRIMARY KEY,
    record_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE (record_date, symbol)
)
"""


def _dsn(db_url: str) -> str:
    """Convert SQLAlchemy-style URL to asyncpg DSN."""
    return db_url.replace("postgresql+asyncpg://", "postgresql://")


class PGRepository:
    def __init__(self, db_url: str) -> None:
        self._dsn = _dsn(db_url)
        self._pool: asyncpg.Pool | None = None

    async def init(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PGRepository not initialised — call init() first")
        return self._pool

    async def get_id_by_date_symbol(self, record_date: str, symbol: str) -> str | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM stockmem_records WHERE record_date = $1 AND symbol = $2",
                record_date, symbol.upper(),
            )
        return str(row["id"]) if row else None

    async def count(self) -> int:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT COUNT(*) FROM stockmem_records")
        return int(val or 0)

    async def upsert(self, record: StockMemRecord) -> str:
        if record.id is None:
            raise ValueError("StockMemRecord.id must be set before upsert")
        pool = self._require_pool()
        record_date = record.date.isoformat()
        symbol = record.symbol.upper()
        payload_record = record.model_copy(update={"symbol": symbol})
        payload = json.dumps(payload_record.model_dump(mode="json"), ensure_ascii=True)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO stockmem_records (id, record_date, symbol, payload)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (record_date, symbol) DO UPDATE
                    SET id = EXCLUDED.id,
                        payload = EXCLUDED.payload
                """,
                record.id, record_date, symbol, payload,
            )
        return record.id

    async def get(self, record_id: str) -> StockMemRecord | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM stockmem_records WHERE id = $1", record_id
            )
        if row is None:
            return None
        return StockMemRecord.model_validate(json.loads(row["payload"]))

    async def get_by_date_symbol(self, record_date: str, symbol: str) -> StockMemRecord | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM stockmem_records WHERE record_date = $1 AND symbol = $2",
                record_date, symbol.upper(),
            )
        if row is None:
            return None
        return StockMemRecord.model_validate(json.loads(row["payload"]))

    async def list_all(self) -> list[StockMemRecord]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT payload FROM stockmem_records")
        return [StockMemRecord.model_validate(json.loads(r["payload"])) for r in rows]

    async def list_missing_returns(self, symbol: str | None = None) -> list[StockMemRecord]:
        """Return records that still have null future_return_1d."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if symbol:
                rows = await conn.fetch(
                    "SELECT payload FROM stockmem_records WHERE symbol = $1"
                    " AND (payload::json->>'future_return_1d') IS NULL"
                    " ORDER BY record_date",
                    symbol.upper(),
                )
            else:
                rows = await conn.fetch(
                    "SELECT payload FROM stockmem_records"
                    " WHERE (payload::json->>'future_return_1d') IS NULL"
                    " ORDER BY record_date"
                )
        return [StockMemRecord.model_validate(json.loads(r["payload"])) for r in rows]

    async def update_future_returns(
        self,
        record_id: str,
        future_return_1d: float | None = None,
        future_return_7d: float | None = None,
        future_return_30d: float | None = None,
    ) -> bool:
        """Patch future_return fields into the stored JSON payload. Returns True if found."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM stockmem_records WHERE id = $1", record_id
            )
            if row is None:
                return False
            payload = json.loads(row["payload"])
            if future_return_1d is not None:
                payload["future_return_1d"] = future_return_1d
            if future_return_7d is not None:
                payload["future_return_7d"] = future_return_7d
            if future_return_30d is not None:
                payload["future_return_30d"] = future_return_30d
            await conn.execute(
                "UPDATE stockmem_records SET payload = $1 WHERE id = $2",
                json.dumps(payload, ensure_ascii=True),
                record_id,
            )
        return True
