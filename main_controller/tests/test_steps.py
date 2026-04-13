"""Tests for individual pipeline step functions."""

import pytest


class TestPipelineSteps:
    """Test suite for individual step functions."""

    @pytest.mark.asyncio
    async def test_step_collect_parallel(self) -> None:
        """Test that step_collect runs Crawler and MarketData in parallel."""
        # TODO: Implement
        pass

    @pytest.mark.asyncio
    async def test_step_collect_handles_failure(self) -> None:
        """Test that step_collect continues on individual failure."""
        # TODO: Implement
        pass

    @pytest.mark.asyncio
    async def test_step_ai_score(self) -> None:
        """Test AI scoring step."""
        # TODO: Implement
        pass
