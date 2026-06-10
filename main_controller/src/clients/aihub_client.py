"""AIHub module HTTP client."""

from shared.cache import RedisCache
from shared.models.factor import Factor
from shared.models.memory import SimilarRecord, StockMemRecord
from shared.models.prediction import PredictResponse

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import AIHubClientError


def _prediction_cache_payload(payload: dict) -> dict:
    """Drop per-run fields that should not invalidate equivalent predictions."""
    current = dict(payload["current"])
    current.pop("id", None)
    current.pop("run_id", None)

    similar = []
    for item in payload["similar"]:
        cloned = dict(item)
        record = cloned.get("record")
        if isinstance(record, dict):
            cloned["record"] = dict(record)
            cloned["record"].pop("run_id", None)
        similar.append(cloned)

    return {"current": current, "similar": similar}


class AIHubClient(BaseHTTPClient):
    """Async HTTP client for the AIHub module."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        *,
        cache: RedisCache | None = None,
        predict_ttl_seconds: int = 90000,
    ) -> None:
        super().__init__(base_url, AIHubClientError)
        self._cache = cache
        self._predict_ttl_seconds = predict_ttl_seconds

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
        payload = {
            "current": current.model_dump(mode="json"),
            "similar": [s.model_dump(mode="json") for s in similar],
        }
        key_payload = _prediction_cache_payload(payload)
        key = self._cache.hashed_key("aihub:predict", key_payload) if self._cache else None
        if self._cache is not None and key is not None:
            cached = await self._cache.get_json(key)
            if cached is not None:
                return PredictResponse.model_validate(cached)

        body = await self._post(
            "/predict",
            payload,
        )
        if self._cache is not None and key is not None:
            await self._cache.set_json(key, body, self._predict_ttl_seconds)
        return PredictResponse.model_validate(body)
