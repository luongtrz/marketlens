"""Tests for factor extraction."""

import pytest
from aihub.src.factors.skgp import SKGPExtractor
from shared.models.factor import Factor, FactorType



class TestSKGPExtractor:
    """Test suite for SKGPExtractor."""

    def test_extract_returns_factors(self) -> None:
        """Test that extract returns a list of Factor objects."""
        extractor = SKGPExtractor()
        factors = extractor.extract("BTC", "Some news text")
        assert isinstance(factors, list)
        assert all(isinstance(f, Factor) for f in factors)

    def test_factor_types_are_valid(self) -> None:
        """Test that all extracted factors have valid FactorType values."""
        extractor = SKGPExtractor()
        factors = extractor.extract("BTC", "Some news text")
        assert all(isinstance(f.type, FactorType) for f in factors)