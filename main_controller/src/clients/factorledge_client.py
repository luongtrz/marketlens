"""FactorLedge module HTTP client."""

from shared.models.factor import NormalizedFactor

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import FactorLedgeClientError


class FactorLedgeClient(BaseHTTPClient):
    """Async HTTP client for the FactorLedge module (Python gateway :8004)."""

    def __init__(self, base_url: str = "http://localhost:8004") -> None:
        super().__init__(base_url, FactorLedgeClientError)

    async def health_check(self) -> bool:
        body = await self._get("/health")
        return body.get("status") == "ok"  # type: ignore[union-attr]

    async def update_ledger(
        self,
        records: list[dict],   # [{"date": "YYYY-MM-DD", "factors": [...]}]
        window_days: int = 7,
    ) -> list[NormalizedFactor]:
        """Push DailyRecord list to ledger-service and rebuild the rolling ledger.

        Returns an empty list — the ledger is a side-effect store; callers that
        need the resulting vector should call get_factor_vector() afterwards.
        """
        await self._post(
            "/ledger/update",
            {"records": records, "windowDays": window_days},
        )
        return []

    async def get_vector(self) -> dict:
        """Return the 13-dim group weight vector from query-service."""
        return await self._get("/query/vector")  # type: ignore[return-value]

    async def get_factor_vector(self) -> dict:
        """Return the 75-dim binary factor vector from query-service."""
        return await self._get("/query/factor-vector")  # type: ignore[return-value]

    async def classify_vector(
        self,
        factors: list[str],
    ) -> list[float]:
        """Classify factor names via classify-service and return the 75d binary vector.

        Uses the full 5-tier classification pipeline (exact → cache → keyword →
        fuzzy → LLM fallback) to produce per-record factor vectors for StockMem.
        """
        body = await self._post("/classify/vector", {"factors": factors})
        return body.get("factorVector", [])  # type: ignore[return-value]
