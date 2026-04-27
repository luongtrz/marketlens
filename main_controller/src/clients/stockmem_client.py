"""StockMem module HTTP client."""

from shared.models.memory import SimilarRecord, StockMemRecord

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import StockMemClientError


class StockMemClient(BaseHTTPClient):
    """Async HTTP client for the StockMem module."""

    def __init__(self, base_url: str = "http://localhost:8003") -> None:
        super().__init__(base_url, StockMemClientError)

    async def health_check(self) -> bool:
        body = await self._get("/health")
        return body.get("status") == "ok"  # type: ignore[union-attr]

    async def save(self, record: StockMemRecord) -> str:
        body = await self._post("/record", {"record": record.model_dump(mode="json")})
        return body["id"]  # type: ignore[index]

    async def search(self, query: StockMemRecord, k: int = 5) -> list[SimilarRecord]:
        body = await self._post(
            "/search",
            {"query": query.model_dump(mode="json"), "k": k},
        )
        return [SimilarRecord.model_validate(r) for r in body["results"]]  # type: ignore[index]
