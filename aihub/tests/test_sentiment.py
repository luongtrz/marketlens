"""Tests for sentiment analysis model."""

import pytest
from aihub.src.sentiment.model import CryptoBertModel
from aihub.src.sentiment.model import SentimentResult




class TestCryptoBertModel:
    """Test suite for CryptoBertModel."""

    def test_predict_returns_sentiment_result(self) -> None:
        """Test that predict returns a valid SentimentResult."""
        model = CryptoBertModel()
        result = model.predict("Some news text")
        assert isinstance(result, SentimentResult)

    def test_predict_score_range(self) -> None:
        """Test that sentiment score is within [-1, 1]."""
        model = CryptoBertModel()
        result = model.predict("Some news text")
        assert -1 <= result.score <= 1
