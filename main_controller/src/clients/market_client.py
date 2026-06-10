"""MarketData module HTTP client."""

from shared.models.market import OHLCV, MarketSnapshot
from shared.cache import RedisCache

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import MarketClientError


class MarketClient(BaseHTTPClient):
    """Async HTTP client for the MarketData module."""

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        *,
        cache: RedisCache | None = None,
        snapshot_ttl_seconds: int = 45,
        history_ttl_seconds: int = 300,
        historical_history_ttl_seconds: int = 86400,
    ) -> None:
        super().__init__(base_url, MarketClientError)
        self._cache = cache
        self._snapshot_ttl_seconds = snapshot_ttl_seconds
        self._history_ttl_seconds = history_ttl_seconds
        self._historical_history_ttl_seconds = historical_history_ttl_seconds

    async def health_check(self) -> bool:
        body = await self._get("/health")
        return body.get("status") == "ok"  # type: ignore[union-attr]

    async def get_snapshot(self, symbol: str, interval: str = "1d") -> MarketSnapshot:
        key = None
        if self._cache is not None:
            key = self._cache.key("market", "snapshot", symbol.upper(), interval)
            cached = await self._cache.get_json(key)
            if cached is not None:
                return MarketSnapshot.model_validate(cached)

        body = await self._get("/snapshot", symbol=symbol, interval=interval)
        if self._cache is not None and key is not None:
            await self._cache.set_json(key, body, self._snapshot_ttl_seconds)
        return MarketSnapshot.model_validate(body)

    async def get_history(
        self, symbol: str, interval: str = "1d", limit: int = 200, end_time: str | None = None
    ) -> list[OHLCV]:
        params: dict = dict(symbol=symbol, interval=interval, limit=limit)
        if end_time:
            params["end_time"] = end_time
        key = None
        if self._cache is not None:
            key = self._cache.key(
                "market",
                "history",
                symbol.upper(),
                interval,
                limit,
                end_time or "live",
            )
            cached = await self._cache.get_json(key)
            if cached is not None:
                return [OHLCV.model_validate(c) for c in cached]

        body = await self._get("/history", **params)
        if self._cache is not None and key is not None:
            ttl = (
                self._historical_history_ttl_seconds
                if end_time
                else self._history_ttl_seconds
            )
            await self._cache.set_json(key, body, ttl)
        return [OHLCV.model_validate(c) for c in body]  # type: ignore[union-attr]

    async def get_indicators(
        self, candles: list[OHLCV], indicator_names: list[str] | None = None
    ) -> dict:
        names = indicator_names or ["rsi", "macd", "bb"]
        body = await self._post(
            "/indicators",
            json_body={
                "ohlcv": [c.model_dump(mode="json") for c in candles],
                "indicator_names": names,
            },
        )
        return body if isinstance(body, dict) else {}
