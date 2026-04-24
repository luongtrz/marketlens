"""SentimentModel — crypto sentiment analysis via CryptoBert HTTP API."""

import httpx
from pydantic import BaseModel

from aihub.src.llm.models.base import BaseAIModel


class SentimentResult(BaseModel):
    """Result of a sentiment analysis inference."""

    score: float  # -1.0 (bearish) to 1.0 (bullish)
    label: str  # "bullish" | "bearish" | "neutral"


class SentimentModel(BaseAIModel):
    """CryptoBert sentiment analysis model.

    Unlike ExplainModel and PredictModel, this model does NOT use a generic
    LLMClient. It calls the CryptoBert HTTP API directly. It still inherits
    BaseAIModel so it participates in the factory pattern uniformly.

    Args:
        hf_model_url: URL of the CryptoBert HuggingFace Spaces API.
        model_path: Local path to model weights (for future local inference).
    """

    def __init__(
        self,
        hf_model_url: str,
        model_path: str = "",
    ) -> None:
        super().__init__(llm=None)
        self._hf_model_url = hf_model_url
        self._model_path = model_path

    async def run(self, text: str) -> SentimentResult:
        """Run sentiment inference on the given text.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentResult with score and label.

        Raises:
            httpx.HTTPStatusError: If the remote model service returns an error status.
            httpx.RequestError: If the request cannot be sent (network error, timeout, etc.).
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._hf_model_url, json={"texts": [text]}, timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json().get("results")[0]
            return SentimentResult(
                score=float(data.get("numeric_score", 0.0)),
                label=data.get("label", "neutral"),
            )
