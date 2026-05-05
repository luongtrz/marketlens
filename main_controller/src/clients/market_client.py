"""MarketData module HTTP client."""

from shared.models.market import OHLCV, MarketSnapshot

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import MarketClientError


class MarketClient(BaseHTTPClient):
    """Async HTTP client for the MarketData module."""

    def __init__(self, base_url: str = "http://localhost:8002") -> None:
        super().__init__(base_url, MarketClientError)

    async def health_check(self) -> bool:
        body = await self._get("/health")
        return body.get("status") == "ok"  # type: ignore[union-attr]

    async def get_snapshot(self, symbol: str, interval: str = "1d") -> MarketSnapshot:
        body = await self._get("/snapshot", symbol=symbol, interval=interval)
        return MarketSnapshot.model_validate(body)

    async def get_history(
        self, symbol: str, interval: str = "1d", limit: int = 200, end_time: str | None = None
    ) -> list[OHLCV]:
        params: dict = dict(symbol=symbol, interval=interval, limit=limit)
        if end_time:
            params["end_time"] = end_time
        body = await self._get("/history", **params)
        return [OHLCV.model_validate(c) for c in body]  # type: ignore[union-attr]
