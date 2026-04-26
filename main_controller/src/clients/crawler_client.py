"""Crawler module HTTP client."""

from shared.models.article import IngestionRecord

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import CrawlerClientError


class CrawlerClient(BaseHTTPClient):
    """Async HTTP client for the Crawler module."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        super().__init__(base_url, CrawlerClientError)

    async def health_check(self) -> bool:
        body = await self._get("/health")
        return body.get("status") == "ok"  # type: ignore[union-attr]

    async def get_latest(self, symbol: str) -> list[IngestionRecord]:
        body = await self._get("/articles/latest", symbol=symbol)
        return [IngestionRecord.model_validate(a) for a in body]  # type: ignore[union-attr]
