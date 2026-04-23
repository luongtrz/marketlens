"""Tests for sentiment analysis model."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aihub.src.sentiment.model import CryptoBertModel
from aihub.src.sentiment.model import SentimentResult


def _make_mock_response(score: float = 0.8, label: str = "bullish") -> MagicMock:
    """Build a mock httpx response returning a sentiment result."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"results": [{"numeric_score": score, "label": label}]}
    return mock_resp


class TestCryptoBertModel:
    """Test suite for CryptoBertModel."""

    @pytest.mark.asyncio
    async def test_predict_returns_sentiment_result(self) -> None:
        """Test that predict returns a valid SentimentResult."""
        model = CryptoBertModel(model_path="", hf_model_path="http://fake-endpoint")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_make_mock_response())
        with patch("httpx.AsyncClient", return_value=mock_client):
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            result = await model.predict("Some news text")
        assert isinstance(result, SentimentResult)

    @pytest.mark.asyncio
    async def test_predict_score_range(self) -> None:
        """Test that sentiment score is within [-1, 1]."""
        model = CryptoBertModel(model_path="", hf_model_path="http://fake-endpoint")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_make_mock_response(score=0.5))
        with patch("httpx.AsyncClient", return_value=mock_client):
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            result = await model.predict("Some news text")
        assert -1 <= result.score <= 1
