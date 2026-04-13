"""Request/Response schemas for the /sentiment endpoint."""

from pydantic import BaseModel, ConfigDict


class SentimentRequest(BaseModel):
    """Request payload for sentiment analysis."""

    model_config = ConfigDict(extra="ignore")

    text: str


class SentimentResponse(BaseModel):
    """Response from sentiment analysis."""

    model_config = ConfigDict(extra="ignore")

    score: float  # -1.0 to 1.0
    label: str  # "bullish" | "bearish" | "neutral"
