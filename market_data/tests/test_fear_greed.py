"""Tests for Alternative.me Fear & Greed Index adapter."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from market_data.src.sources.fear_greed import FearGreedSource, FearGreedSourceError


def _make_get_client(value_str: str | None = "72", status_code: int = 200):
    @asynccontextmanager
    async def _mock(*args, **kwargs):
        mock_resp = MagicMock()
        if value_str is not None:
            mock_resp.json.return_value = {"data": [{"value": value_str}]}
        else:
            mock_resp.json.return_value = {}
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=MagicMock(),
                response=MagicMock(status_code=status_code),
            )
        else:
            mock_resp.raise_for_status.return_value = None
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        yield mock_client

    return _mock


async def test_fetch_returns_int_in_range(monkeypatch) -> None:
    monkeypatch.setattr(
        "market_data.src.sources.fear_greed.get_client",
        _make_get_client("72"),
    )
    result = await FearGreedSource().fetch()
    assert result == 72


async def test_fetch_clamps_high_to_100(monkeypatch) -> None:
    monkeypatch.setattr(
        "market_data.src.sources.fear_greed.get_client",
        _make_get_client("150"),
    )
    result = await FearGreedSource().fetch()
    assert result == 100


async def test_fetch_clamps_low_to_0(monkeypatch) -> None:
    monkeypatch.setattr(
        "market_data.src.sources.fear_greed.get_client",
        _make_get_client("-5"),
    )
    result = await FearGreedSource().fetch()
    assert result == 0


async def test_fetch_http_error_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        "market_data.src.sources.fear_greed.get_client",
        _make_get_client(None, status_code=500),
    )
    with pytest.raises(FearGreedSourceError):
        await FearGreedSource().fetch()


async def test_fetch_missing_data_key_raises(monkeypatch) -> None:
    @asynccontextmanager
    async def _missing_data(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"result": "ok"}  # missing "data" key
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        yield mock_client

    monkeypatch.setattr("market_data.src.sources.fear_greed.get_client", _missing_data)
    with pytest.raises(FearGreedSourceError):
        await FearGreedSource().fetch()


async def test_fetch_malformed_json_raises(monkeypatch) -> None:
    @asynccontextmanager
    async def _bad_json(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        yield mock_client

    monkeypatch.setattr("market_data.src.sources.fear_greed.get_client", _bad_json)
    with pytest.raises(FearGreedSourceError):
        await FearGreedSource().fetch()
