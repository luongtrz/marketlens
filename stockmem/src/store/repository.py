from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from ..config import sqlite_path_from_url
from ..models import StockMemRecord


class RecordRepository:
    def __init__(self, db_url: str) -> None:
        self._db_path = sqlite_path_from_url(db_url)

    @property
    def db_path(self) -> str:
        return self._db_path

    async def init(self) -> None:
        db_dir = Path(self._db_path).parent
        if str(db_dir) not in ("", "."):
            db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as conn:
            await self._ensure_schema(conn)
            await conn.commit()

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stockmem_records (
                id TEXT PRIMARY KEY,
                record_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(record_date, symbol)
            )
            """
        )

        cursor = await conn.execute("PRAGMA table_info(stockmem_records)")
        rows = await cursor.fetchall()
        columns = {str(r[1]) for r in rows}

        # Migrate legacy schema: (id, payload) -> (id, record_date, symbol, payload)
        if {"record_date", "symbol"}.issubset(columns):
            return

        legacy_rows_cursor = await conn.execute("SELECT id, payload FROM stockmem_records")
        legacy_rows = await legacy_rows_cursor.fetchall()

        await conn.execute("ALTER TABLE stockmem_records RENAME TO stockmem_records_legacy")
        await conn.execute(
            """
            CREATE TABLE stockmem_records (
                id TEXT PRIMARY KEY,
                record_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(record_date, symbol)
            )
            """
        )

        for row in legacy_rows:
            record_id = str(row[0])
            payload_raw = str(row[1])
            data: dict[str, Any] = json.loads(payload_raw)
            record_date = data.get("date")
            symbol = str(data.get("symbol", "")).upper()
            if not record_date or not symbol:
                continue

            data["symbol"] = symbol
            payload = json.dumps(data, ensure_ascii=True)

            await conn.execute(
                """
                INSERT INTO stockmem_records (id, record_date, symbol, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(record_date, symbol) DO UPDATE
                SET id = excluded.id,
                    payload = excluded.payload
                """,
                (record_id, record_date, symbol, payload),
            )

        await conn.execute("DROP TABLE stockmem_records_legacy")

    async def get_id_by_date_symbol(self, record_date: str, symbol: str) -> str | None:
        symbol_norm = symbol.upper()
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(
                "SELECT id FROM stockmem_records WHERE record_date = ? AND symbol = ?",
                (record_date, symbol_norm),
            )
            row = await cursor.fetchone()

        if row is None:
            return None
        return str(row[0])

    async def count(self) -> int:
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM stockmem_records")
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def upsert(self, record: StockMemRecord) -> str:
        assert record.id is not None
        record_date = record.date.isoformat()
        symbol = record.symbol.upper()
        payload_record = record.model_copy(update={"symbol": symbol})
        payload = json.dumps(payload_record.model_dump(mode="json"), ensure_ascii=True)

        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                """
                INSERT INTO stockmem_records (id, record_date, symbol, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(record_date, symbol) DO UPDATE
                SET id = excluded.id,
                    payload = excluded.payload
                """
                ,
                (record.id, record_date, symbol, payload),
            )
            await conn.commit()

        return record.id

    async def get(self, record_id: str) -> StockMemRecord | None:
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(
                "SELECT payload FROM stockmem_records WHERE id = ?",
                (record_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        data = json.loads(row[0])
        return StockMemRecord.model_validate(data)

    async def get_by_date_symbol(self, record_date: str, symbol: str) -> StockMemRecord | None:
        symbol_norm = symbol.upper()
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(
                "SELECT payload FROM stockmem_records WHERE record_date = ? AND symbol = ?",
                (record_date, symbol_norm),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        data = json.loads(row[0])
        return StockMemRecord.model_validate(data)

    async def list_all(self) -> list[StockMemRecord]:
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute("SELECT payload FROM stockmem_records")
            rows = await cursor.fetchall()

        return [StockMemRecord.model_validate(json.loads(r[0])) for r in rows]

    async def list_missing_returns(self, symbol: str | None = None) -> list[StockMemRecord]:
        query = "SELECT payload FROM stockmem_records WHERE json_extract(payload, '$.future_return_1d') IS NULL"
        params: tuple = ()
        if symbol:
            query += " AND symbol = ?"
            params = (symbol.upper(),)
        query += " ORDER BY record_date"
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
        return [StockMemRecord.model_validate(json.loads(r[0])) for r in rows]

    async def update_future_returns(
        self,
        record_id: str,
        future_return_1d: float | None = None,
        future_return_7d: float | None = None,
        future_return_30d: float | None = None,
    ) -> bool:
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(
                "SELECT payload FROM stockmem_records WHERE id = ?", (record_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            payload = json.loads(row[0])
            if future_return_1d is not None:
                payload["future_return_1d"] = future_return_1d
            if future_return_7d is not None:
                payload["future_return_7d"] = future_return_7d
            if future_return_30d is not None:
                payload["future_return_30d"] = future_return_30d
            await conn.execute(
                "UPDATE stockmem_records SET payload = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=True), record_id),
            )
            await conn.commit()
        return True

    async def close(self) -> None:
        # SQLite connections are opened/closed per call, nothing to release.
        return None
