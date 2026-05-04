"""Tests for MarketClient."""

import pytest

from main_controller.src.clients.market_client import MarketClient
from main_controller.src.clients.exceptions import MarketClientError
from main_controller.tests.conftest import make_get_client, make_connect_error_client
from shared.models.market import MarketSnapshot


async def test_get_snapshot_happy(monkeypatch, sample_market_snapshot):
    payload = sample_market_snapshot.model_dump(mode="json")
    monkeypatch.setattr("main_controller.src.clients.base.get_client", make_get_client(payload))
    client = MarketClient()
    result = await client.get_snapshot("BTCUSDT")
    assert isinstance(result, MarketSnapshot)
    assert result.symbol == "BTCUSDT"


async def test_get_snapshot_http_500_raises(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({}, status_code=500),
    )
    client = MarketClient()
    with pytest.raises(MarketClientError):
        await client.get_snapshot("BTCUSDT")


async def test_get_snapshot_connect_error_raises(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_connect_error_client(),
    )
    client = MarketClient()
    with pytest.raises(MarketClientError):
        await client.get_snapshot("BTCUSDT")


async def test_health_check_returns_true(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({"status": "ok"}),
    )
    client = MarketClient()
    assert await client.health_check() is True
