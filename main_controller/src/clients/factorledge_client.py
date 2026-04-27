"""FactorLedge module HTTP client."""

from shared.models.factor import NormalizedFactor

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import FactorLedgeClientError


class FactorLedgeClient(BaseHTTPClient):
    """Async HTTP client for the FactorLedge module."""

    def __init__(self, base_url: str = "http://localhost:8004") -> None:
        super().__init__(base_url, FactorLedgeClientError)

    async def health_check(self) -> bool:
        body = await self._get("/health")
        return body.get("status") == "ok"  # type: ignore[union-attr]

    async def ingest(
        self, article_id: str, factors: list[str], source: str
    ) -> list[NormalizedFactor]:
        body = await self._post(
            "/ingest",
            {"article_id": article_id, "factors": factors, "source": source},
        )
        items = body.get("factors", body) if isinstance(body, dict) else body  # type: ignore[union-attr]
        return [NormalizedFactor.model_validate(f) for f in items]
