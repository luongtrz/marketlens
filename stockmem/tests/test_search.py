"""Tests for StockMem similarity search."""

import pytest


class TestRecordSearch:
    """Test suite for similarity search."""

    @pytest.mark.asyncio
    async def test_search_returns_similar_records(self) -> None:
        """Test that search returns k similar records."""
        # TODO: Use in-memory FAISS index
        pass

    @pytest.mark.asyncio
    async def test_search_similarity_range(self) -> None:
        """Test that similarity scores are in [0, 1]."""
        # TODO: Implement
        pass
