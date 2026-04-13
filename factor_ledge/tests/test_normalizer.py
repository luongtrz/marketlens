"""Tests for factor normalizer pipeline."""

import pytest


class TestNormalizer:
    """Test suite for factor cleaning, scoring, and enrichment."""

    def test_cleaner_deduplicates(self) -> None:
        """Test that cleaner removes duplicates."""
        # TODO: Implement
        pass

    def test_scorer_assigns_weights(self) -> None:
        """Test that scorer assigns weights in [0, 1]."""
        # TODO: Implement
        pass

    def test_enricher_adds_metadata(self) -> None:
        """Test that enricher attaches sector/asset metadata."""
        # TODO: Implement
        pass
