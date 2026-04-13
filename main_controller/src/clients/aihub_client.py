"""AIHub module HTTP client."""

from shared.http_client import get_client
from shared.models.factor import Factor
from shared.models.memory import StockMemRecord, SimilarRecord
from shared.models.prediction import PredictResponse


class AIHubClient:
    """Async HTTP client for the AIHub module.

    Args:
        base_url: AIHub service base URL.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def health_check(self) -> bool:
        """Check if the AIHub service is healthy."""
        raise NotImplementedError

    async def sentiment(self, text: str) -> dict:
        """Call the /sentiment endpoint.

        Args:
            text: Text to analyze.

        Returns:
            Dict with score and label.
        """
        raise NotImplementedError

    async def factors(self, text: str) -> list[Factor]:
        """Call the /factors endpoint.

        Args:
            text: Text to extract factors from.

        Returns:
            List of Factor objects.
        """
        raise NotImplementedError

    async def predict(
        self, current: StockMemRecord, similar: list[SimilarRecord]
    ) -> PredictResponse:
        """Call the /predict endpoint with RAG context.

        Args:
            current: Current pipeline record.
            similar: Similar historical records.

        Returns:
            PredictResponse with signal, confidence, and explanation.
        """
        raise NotImplementedError
