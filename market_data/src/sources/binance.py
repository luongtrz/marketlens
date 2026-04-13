"""Binance REST + WebSocket adapter for market data."""

from shared.models.market import OHLCV, Ticker
from market_data.src.sources.base import MarketSource


class BinanceSource(MarketSource):
    """Binance market data adapter using REST API and optional WebSocket.

    Args:
        api_key: Binance API key.
        api_secret: Binance API secret (optional for public endpoints).
        base_url: Binance API base URL.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://api.binance.com",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url

    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int
    ) -> list[OHLCV]:
        raise NotImplementedError

    async def fetch_ticker(self, symbol: str) -> Ticker:
        raise NotImplementedError
