"""Tests for LLM client."""

import pytest


class TestLLMClient:
    """Test suite for LLMClient."""

    @pytest.mark.asyncio
    async def test_enrich_returns_fields(self) -> None:
        """Test that enrich returns valid EnrichedFields."""
        # TODO: Mock AIHub endpoints
        pass

    @pytest.mark.asyncio
    async def test_enrich_fallback_on_aihub_failure(self) -> None:
        """Test fallback to direct LLM when AIHub is unavailable."""
        # TODO: Implement
        pass
