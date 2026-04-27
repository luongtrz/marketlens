"""Tests for AIHubClient."""

import pytest

from main_controller.src.clients.aihub_client import AIHubClient
from main_controller.src.clients.exceptions import AIHubClientError
from main_controller.tests.conftest import make_get_client, make_connect_error_client
from shared.models.prediction import PredictResponse, SignalType


async def test_sentiment_happy(monkeypatch):
    payload = {"score": 0.75, "label": "bullish"}
    monkeypatch.setattr("main_controller.src.clients.base.get_client", make_get_client(payload))
    client = AIHubClient()
    result = await client.sentiment("BTC ETF approved")
    assert result["score"] == 0.75
    assert result["label"] == "bullish"


async def test_factors_with_ticker(monkeypatch, sample_factor):
    payload = {"factors": [sample_factor.model_dump(mode="json")]}
    monkeypatch.setattr("main_controller.src.clients.base.get_client", make_get_client(payload))
    client = AIHubClient()
    result = await client.factors("BTC halving news", ticker="BTCUSDT")
    assert len(result) == 1
    assert result[0].name == sample_factor.name


async def test_predict_happy(monkeypatch, sample_stockmem_record, sample_predict_response):
    payload = sample_predict_response.model_dump(mode="json")
    monkeypatch.setattr("main_controller.src.clients.base.get_client", make_get_client(payload))
    client = AIHubClient()
    result = await client.predict(current=sample_stockmem_record, similar=[])
    assert isinstance(result, PredictResponse)
    assert result.signal == SignalType.BUY
    assert result.confidence == 0.85


async def test_sentiment_http_error_raises(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({}, status_code=500),
    )
    client = AIHubClient()
    with pytest.raises(AIHubClientError):
        await client.sentiment("text")
