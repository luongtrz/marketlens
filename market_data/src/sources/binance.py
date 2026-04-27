"""Binance REST + WebSocket adapter for market data."""

import logging
from datetime import datetime, timezone

import httpx
from shared.models.market import OHLCV, Ticker
from market_data.src.sources.base import MarketSource

logger = logging.getLogger(__name__)

# Binance interval mapping
INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}


class BinanceSource(MarketSource):
    """Binance market data adapter using REST API and WebSocket.

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
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
            headers={"User-Agent": "marketlens/1.0"},
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int = 100, end_time: int | None = None
    ) -> list[OHLCV]:
        """Fetch OHLCV candles from Binance REST API.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            interval: Candle interval (e.g. "1h", "4h", "1d").
            limit: Maximum number of candles to return (max 1000).

        Returns:
            List of OHLCV candles ordered by timestamp ascending.
        """
        binance_interval = INTERVAL_MAP.get(interval, interval)
        limit = min(limit, 1000)  # Binance max

        try:
            params: dict[str, any] = {
                "symbol": symbol.upper(),
                "interval": binance_interval,
                "limit": limit,
            }
            if end_time is not None:
                params["endTime"] = end_time

            response = await self._client.get(
                "/api/v3/klines",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            candles = []
            for candle in data:
                # Binance kline format: [open_time, open, high, low, close, volume, ...]
                timestamp = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                candles.append(
                    OHLCV(
                        timestamp=timestamp,
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5]),
                        interval=interval,
                    )
                )

            logger.info(
                f"Fetched {len(candles)} {interval} candles for {symbol} from Binance"
            )
            return candles

        except httpx.HTTPError as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            return []

    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch current ticker from Binance.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").

        Returns:
            Current Ticker information.
        """
        try:
            response = await self._client.get(
                "/api/v3/ticker/24hr",
                params={"symbol": symbol.upper()},
            )
            response.raise_for_status()
            data = response.json()

            return Ticker(
                symbol=symbol.upper(),
                price=float(data["lastPrice"]),
                volume_24h=float(data.get("quoteVolume", 0)),
                timestamp=datetime.now(timezone.utc),
            )

        except httpx.HTTPError as e:
            logger.error(f"Binance API error for ticker {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching ticker {symbol}: {e}")
            raise
