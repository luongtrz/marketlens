"""Shared test fixtures and helpers for main_controller tests."""

from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from shared.models.article import IngestionRecord
from shared.models.factor import Factor, FactorType, NormalizedFactor
from shared.models.market import MarketSnapshot, OHLCV
from shared.models.memory import SimilarRecord, StockMemRecord
from shared.models.prediction import PredictResponse, SignalType


# ---------------------------------------------------------------------------
# HTTP client mock helpers
# ---------------------------------------------------------------------------

def make_get_client(response_json: object, status_code: int = 200):
    """Return a drop-in mock for shared.http_client.get_client."""

    @asynccontextmanager
    async def _mock(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_json
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=MagicMock(),
                response=MagicMock(status_code=status_code),
            )
        else:
            mock_resp.raise_for_status.return_value = None
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        yield mock_client

    return _mock


def make_connect_error_client():
    """Return a mock get_client that raises ConnectError on request."""

    @asynccontextmanager
    async def _mock(*args, **kwargs):
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        yield mock_client

    return _mock


# ---------------------------------------------------------------------------
# Canonical fixture objects
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_TODAY = date(2026, 1, 1)


@pytest.fixture
def sample_ohlcv() -> OHLCV:
    return OHLCV(
        timestamp=_NOW,
        open=95000.0,
        high=96000.0,
        low=94000.0,
        close=95500.0,
        volume=1000.0,
        interval="1d",
    )


@pytest.fixture
def sample_market_snapshot(sample_ohlcv) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=_NOW,
        ohlcv=sample_ohlcv,
        indicators={"rsi": 55.0, "macd_hist": 20.0, "sma_20": 94000.0},
        source="binance",
        recent_candles=[sample_ohlcv],
    )


@pytest.fixture
def sample_article() -> IngestionRecord:
    return IngestionRecord(
        id="art-001",
        article_name="BTC ETF Approved",
        source="coindesk",
        url="https://coindesk.com/btc-etf",
        date_published=_NOW,
        date_crawled=_NOW,
        sentiment_score=0.7,
        sentiment_label="bullish",
        factors=["institutional adoption", "regulatory approval"],
        raw_text="Bitcoin ETF approved by SEC. Institutional demand surges.",
    )


@pytest.fixture
def sample_normalized_factor() -> NormalizedFactor:
    return NormalizedFactor(
        name="institutional_flows",
        type=FactorType.MACRO,
        weight=0.6,
        polarity=0.4,
        source_article_id="art-001",
        observed_at=_NOW,
    )


@pytest.fixture
def sample_factor() -> Factor:
    return Factor(
        name="institutional adoption",
        type=FactorType.MACRO,
        polarity=0.5,
        confidence=0.8,
    )


@pytest.fixture
def sample_stockmem_record(sample_market_snapshot, sample_normalized_factor) -> StockMemRecord:
    return StockMemRecord(
        id="rec-001",
        date=_TODAY,
        symbol="BTCUSDT",
        sentiment_score=0.7,
        sentiment_label="bullish",
        factors=["institutional_flows"],
        normalized_factors=[sample_normalized_factor],
        market_snapshot=sample_market_snapshot,
        run_id="run-001",
    )


@pytest.fixture
def sample_similar_record(sample_stockmem_record) -> SimilarRecord:
    return SimilarRecord(record=sample_stockmem_record, similarity=0.92)


@pytest.fixture
def sample_predict_response() -> PredictResponse:
    return PredictResponse(
        signal=SignalType.BUY,
        confidence=0.85,
        explanation="Strong institutional inflows and bullish sentiment.",
        reasoning_steps=["Check sentiment", "Check factors", "Signal: BUY"],
    )
