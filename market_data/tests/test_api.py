"""Integration smoke tests for market_data FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from shared.models.market import OHLCV
from market_data.src.api import app
from market_data.src.sources.binance import BinanceSourceError
from market_data.src.sources.fear_greed import FearGreedSourceError


def _ohlcv(close: float = 50000.0) -> OHLCV:
    return OHLCV(
        timestamp=datetime(2026, 4, 25, tzinfo=timezone.utc),
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=1_000_000.0,
        interval="1d",
    )


def _candles(n: int = 25) -> list[OHLCV]:
    return [_ohlcv(50000.0 + i * 100) for i in range(n)]


class _MockBinanceSource:
    def __init__(self, candles: list[OHLCV] | None = None, fail: bool = False) -> None:
        self._candles = candles or _candles()
        self._fail = fail

    async def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> list[OHLCV]:
        if self._fail:
            raise BinanceSourceError("Binance unavailable")
        return self._candles


class _MockFearGreedSource:
    def __init__(self, value: int = 65, fail: bool = False) -> None:
        self._value = value
        self._fail = fail

    async def fetch(self) -> int:
        if self._fail:
            raise FearGreedSourceError("F&G unavailable")
        return self._value


@pytest.fixture
def happy_client() -> TestClient:
    with TestClient(app) as c:
        app.state.binance = _MockBinanceSource()
        app.state.fear_greed = _MockFearGreedSource(value=65)
        yield c


# --- Health ---

def test_health_endpoint() -> None:
    with TestClient(app) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# --- Snapshot happy path ---

def test_snapshot_returns_200_with_all_indicator_keys(happy_client: TestClient) -> None:
    resp = happy_client.get("/snapshot?symbol=BTCUSDT&interval=1d")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert set(body["indicators"].keys()) >= {"rsi", "macd_hist", "sma", "fear_greed_index", "price_change_pct"}


def test_snapshot_recent_candles_non_empty(happy_client: TestClient) -> None:
    resp = happy_client.get("/snapshot?symbol=BTCUSDT&interval=1d")
    assert resp.status_code == 200
    assert len(resp.json()["recent_candles"]) >= 1


def test_snapshot_symbol_normalized_uppercase(happy_client: TestClient) -> None:
    resp = happy_client.get("/snapshot?symbol=btcusdt&interval=1d")
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "BTCUSDT"


def test_snapshot_fear_greed_value_matches_mock(happy_client: TestClient) -> None:
    resp = happy_client.get("/snapshot?symbol=BTCUSDT&interval=1d")
    assert resp.json()["indicators"]["fear_greed_index"] == pytest.approx(65.0)


# --- Fail-fast policy ---

def test_snapshot_binance_fail_returns_500() -> None:
    with TestClient(app) as c:
        app.state.binance = _MockBinanceSource(fail=True)
        app.state.fear_greed = _MockFearGreedSource()
        resp = c.get("/snapshot?symbol=BTCUSDT&interval=1d")
    assert resp.status_code == 500


def test_snapshot_fear_greed_fail_returns_500() -> None:
    with TestClient(app) as c:
        app.state.binance = _MockBinanceSource()
        app.state.fear_greed = _MockFearGreedSource(fail=True)
        resp = c.get("/snapshot?symbol=BTCUSDT&interval=1d")
    assert resp.status_code == 500


# --- History endpoint ---

def test_history_returns_ohlcv_list(happy_client: TestClient) -> None:
    resp = happy_client.get("/history?symbol=BTCUSDT&interval=1d&limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) > 0
    assert "close" in body[0]


def test_history_binance_fail_returns_500() -> None:
    with TestClient(app) as c:
        app.state.binance = _MockBinanceSource(fail=True)
        app.state.fear_greed = _MockFearGreedSource()
        resp = c.get("/history?symbol=BTCUSDT&interval=1d")
    assert resp.status_code == 500


# --- Indicators POST endpoint ---

def test_indicators_post_rsi(happy_client: TestClient) -> None:
    payload = {
        "ohlcv": [
            {
                "timestamp": "2026-04-25T00:00:00Z",
                "open": 49000.0, "high": 51000.0, "low": 48000.0,
                "close": float(50000 + i * 100), "volume": 1000.0, "interval": "1d",
            }
            for i in range(20)
        ],
        "indicator_names": ["rsi"],
    }
    resp = happy_client.post("/indicators", json=payload)
    assert resp.status_code == 200
    assert "rsi" in resp.json()


def test_indicators_post_unknown_name_returns_400(happy_client: TestClient) -> None:
    payload = {
        "ohlcv": [
            {
                "timestamp": "2026-04-25T00:00:00Z",
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.0, "volume": 1000.0, "interval": "1d",
            }
        ],
        "indicator_names": ["nonexistent"],
    }
    resp = happy_client.post("/indicators", json=payload)
    assert resp.status_code == 400
