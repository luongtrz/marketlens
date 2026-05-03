"""Tests for factor extraction."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from aihub.src.factors.skgp import SKGPExtractor
from shared.models.factor import Factor, FactorType

_MOCK_RESPONSE = {
    "factors": [
        {"factor": "regulatory", "type": "regulatory", "polarity": 1.0, "confidence": 90}
    ]
}


def _make_extractor() -> SKGPExtractor:
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value=_MOCK_RESPONSE)
    return SKGPExtractor(llm)


class TestSKGPExtractor:
    """Test suite for SKGPExtractor."""

    def test_extract_returns_factors(self) -> None:
        """Test that extract returns a list of Factor objects."""
        extractor = _make_extractor()
        factors = asyncio.run(extractor.extract("BTC", "Some news text"))
        assert isinstance(factors, list)
        assert all(isinstance(f, Factor) for f in factors)

    def test_factor_types_are_valid(self) -> None:
        """Test that all extracted factors have valid FactorType values."""
        extractor = _make_extractor()
        factors = asyncio.run(extractor.extract("BTC", "Some news text"))
        assert all(isinstance(f.type, FactorType) for f in factors)
