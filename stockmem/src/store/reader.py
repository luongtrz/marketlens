from __future__ import annotations

from datetime import date

from ..models import StockMemRecord
from .base import Repository


class RecordReader:
    """Read records by id or by (date, symbol)."""

    def __init__(self, repository: Repository) -> None:
        self._repository = repository

    async def get_by_id(self, record_id: str) -> StockMemRecord | None:
        return await self._repository.get(record_id)

    async def get_by_date(self, record_date: date, symbol: str) -> StockMemRecord | None:
        return await self._repository.get_by_date_symbol(record_date.isoformat(), symbol)
