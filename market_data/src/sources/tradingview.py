"""TradingView scraper / API adapter for market data."""

from shared.models.market import OHLCV, Ticker
from market_data.src.sources.base import MarketSource


class TradingViewSource(MarketSource):
    """TradingView market data adapter.

    Args:
        base_url: TradingView API/scraper base URL.
    """

    def __init__(self, base_url: str = "") -> None:
        self._base_url = base_url

    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int
    ) -> list[OHLCV]:
        raise NotImplementedError

    async def fetch_ticker(self, symbol: str) -> Ticker:
        raise NotImplementedError
