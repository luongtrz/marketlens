"""Tests for StockMem record store."""

import pytest


class TestRecordStore:
    """Test suite for record writer and reader."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve(self) -> None:
        """Test saving a record and retrieving it by ID."""
        # TODO: Use in-memory SQLite
        pass

    @pytest.mark.asyncio
    async def test_get_by_date(self) -> None:
        """Test retrieving a record by date and symbol."""
        # TODO: Implement
        pass
