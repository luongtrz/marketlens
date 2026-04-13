"""MarketData module HTTP client."""

from shared.http_client import get_client
from shared.models.market import MarketSnapshot, OHLCV


class MarketClient:
    """Async HTTP client for the MarketData module.

    Args:
        base_url: MarketData service base URL.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def health_check(self) -> bool:
        """Check if the MarketData service is healthy."""
        raise NotImplementedError

    async def get_snapshot(self, symbol: str, interval: str = "1h") -> MarketSnapshot:
        """Get current market snapshot.

        Args:
            symbol: Trading pair.
            interval: Candle interval.

        Returns:
            MarketSnapshot with OHLCV and indicators.
        """
        raise NotImplementedError

    async def get_history(
        self, symbol: str, interval: str = "1h", limit: int = 200
    ) -> list[OHLCV]:
        """Get historical OHLCV candles.

        Args:
            symbol: Trading pair.
            interval: Candle interval.
            limit: Max candles.

        Returns:
            List of OHLCV candles.
        """
        raise NotImplementedError
