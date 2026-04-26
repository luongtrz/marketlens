"""Binance REST adapter for market data."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from shared.models.market import OHLCV, Ticker
from market_data.src.sources.base import MarketSource
from shared.http_client import get_client


class BinanceSourceError(Exception):
    """Raised when Binance REST API returns an error or unusable response."""


class BinanceSource(MarketSource):
    """Binance market data adapter using public REST API."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://api.binance.com",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> list[OHLCV]:
        url = f"{self._base_url}/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        try:
            async with get_client() as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                rows = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BinanceSourceError(f"fetch_ohlcv failed: {exc}") from exc
        return [
            OHLCV(
                timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                interval=interval,
            )
            for row in rows
        ]

    async def fetch_ticker(self, symbol: str) -> Ticker:
        url = f"{self._base_url}/api/v3/ticker/24hr"
        try:
            async with get_client() as client:
                resp = await client.get(url, params={"symbol": symbol.upper()})
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BinanceSourceError(f"fetch_ticker failed: {exc}") from exc
        return Ticker(
            symbol=data["symbol"],
            price=float(data["lastPrice"]),
            volume_24h=float(data["volume"]),
            timestamp=datetime.now(timezone.utc),
        )
