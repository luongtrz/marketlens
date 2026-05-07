"""StockMem module HTTP client."""

from datetime import date

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

    async def search(
        self,
        query: StockMemRecord,
        k: int = 5,
        before_date: date | None = None,
    ) -> list[SimilarRecord]:
        payload: dict = {"query": query.model_dump(mode="json"), "k": k}
        if before_date is not None:
            payload["before_date"] = before_date.isoformat()
        body = await self._post("/search", payload)
        return [SimilarRecord.model_validate(r) for r in body["results"]]  # type: ignore[index]

    async def list_missing_returns(self, symbol: str | None = None) -> list[StockMemRecord]:
        params = {"symbol": symbol} if symbol else {}
        body = await self._get("/records/missing-returns", **params)
        return [StockMemRecord.model_validate(r) for r in body]  # type: ignore[union-attr]

    async def update_future_returns(
        self,
        record_id: str,
        future_return_1d: float | None = None,
        future_return_7d: float | None = None,
        future_return_30d: float | None = None,
    ) -> None:
        await self._patch(
            f"/record/{record_id}/returns",
            {k: v for k, v in {
                "future_return_1d": future_return_1d,
                "future_return_7d": future_return_7d,
                "future_return_30d": future_return_30d,
            }.items() if v is not None},
        )
