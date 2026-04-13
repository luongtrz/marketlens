"""Tests for RSS feed fetcher."""

import pytest


class TestRSSFetcher:
    """Test suite for RSSFetcher."""

    @pytest.mark.asyncio
    async def test_fetch_one_returns_articles(self) -> None:
        """Test that fetch_one returns a list of RawArticle objects."""
        # TODO: Implement with mock feed data
        pass

    @pytest.mark.asyncio
    async def test_fetch_one_handles_invalid_feed(self) -> None:
        """Test graceful handling of malformed feed data."""
        # TODO: Implement
        pass
