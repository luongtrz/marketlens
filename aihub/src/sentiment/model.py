"""CryptoBert model loader and inference for sentiment analysis."""

from pydantic import BaseModel


class SentimentResult(BaseModel):
    """Result of a sentiment analysis inference."""

    score: float  # -1.0 (bearish) to 1.0 (bullish)
    label: str  # "bullish" | "bearish" | "neutral"


class CryptoBertModel:
    """CryptoBert sentiment analysis model.

    Loaded once at startup and kept in memory. Thread-safe inference
    via asyncio.to_thread.

    Args:
        model_path: Path to the pretrained CryptoBert model.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None

    def load(self) -> None:
        """Load the CryptoBert model into memory.

        Should be called once at application startup.
        """
        raise NotImplementedError

    def predict(self, text: str) -> SentimentResult:
        """Run sentiment inference on the given text.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentResult with score and label.
        """
        raise NotImplementedError
