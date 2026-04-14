from __future__ import annotations

from dataclasses import dataclass


TABLE_NAME = "stockmem_records"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stockmem_records (
    id TEXT PRIMARY KEY,
    record_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(record_date, symbol)
)
"""


@dataclass(frozen=True)
class StockMemRecordRow:
    id: str
    record_date: str
    symbol: str
    payload: str
