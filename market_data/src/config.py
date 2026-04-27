"""MarketData module configuration."""

from shared.config.base_config import BaseAppConfig


class MarketDataConfig(BaseAppConfig):
    """Configuration specific to the MarketData module."""

    market_source: str = "binance"  # "binance" | "tradingview" | "mock"
    binance_api_key: str = ""
    binance_api_secret: str = ""
    default_interval: str = "1h"
    default_indicators: list[str] = ["rsi", "macd", "bb"]
    tracked_symbols: list[str] = ["BTC", "ETH"]  # Symbols available for WebSocket streaming

    model_config = {"env_prefix": "MARKET_", "extra": "ignore"}
