"""Market data models: OHLCV candles, tickers, and snapshots."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class OHLCV(BaseModel):
    """A single OHLCV candle."""

    model_config = ConfigDict(extra="ignore")

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str  # "1m" | "5m" | "1h" | "4h" | "1d"


class Ticker(BaseModel):
    """Real-time ticker information for a symbol."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    price: float
    volume_24h: float
    timestamp: datetime


class MarketSnapshot(BaseModel):
    """Point-in-time market snapshot with computed indicators."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    timestamp: datetime
    ohlcv: OHLCV
    recent_candles: list[OHLCV] = []
    indicators: dict[str, Any]
    source: str  # "binance" | "tradingview"
