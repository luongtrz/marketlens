"""AIHub module HTTP client."""

from shared.models.factor import Factor
from shared.models.memory import SimilarRecord, StockMemRecord
from shared.models.prediction import PredictResponse

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import AIHubClientError


class AIHubClient(BaseHTTPClient):
    """Async HTTP client for the AIHub module."""

    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        super().__init__(base_url, AIHubClientError)

    async def health_check(self) -> bool:
        body = await self._get("/health")
        return body.get("status") == "ok"  # type: ignore[union-attr]

    async def sentiment(self, text: str) -> dict:
        body = await self._post("/sentiment", {"text": text})
        return {"score": body["score"], "label": body["label"]}  # type: ignore[index]

    async def factors(self, text: str, ticker: str) -> list[Factor]:
        body = await self._post("/factors", {"ticker": ticker, "text": text})
        return [Factor.model_validate(f) for f in body["factors"]]  # type: ignore[index]

    async def predict(
        self, current: StockMemRecord, similar: list[SimilarRecord]
    ) -> PredictResponse:
        body = await self._post(
            "/predict",
            {
                "current": current.model_dump(mode="json"),
                "similar": [s.model_dump(mode="json") for s in similar],
            },
        )
        return PredictResponse.model_validate(body)
