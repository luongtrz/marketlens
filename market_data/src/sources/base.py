"""Abstract market data source interface."""

from abc import ABC, abstractmethod

from shared.models.market import OHLCV, Ticker


class MarketSource(ABC):
    """Abstract base class for market data sources (Binance, TradingView, etc.)."""

    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int
    ) -> list[OHLCV]:
        """Fetch OHLCV candles for the given symbol.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            interval: Candle interval (e.g. "1h", "4h", "1d").
            limit: Maximum number of candles to return.

        Returns:
            List of OHLCV candles ordered by timestamp ascending.
        """
        ...

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch the current ticker for the given symbol.

        Args:
            symbol: Trading pair.

        Returns:
            Current Ticker information.
        """
        ...
