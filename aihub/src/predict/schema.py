"""Request/Response schemas for the /predict endpoint."""

from pydantic import BaseModel, ConfigDict

from shared.models.memory import StockMemRecord, SimilarRecord
from shared.models.prediction import SignalType


class PredictRequest(BaseModel):
    """Request payload for prediction."""

    model_config = ConfigDict(extra="ignore")

    current: StockMemRecord
    similar: list[SimilarRecord] = []


class PredictResponse(BaseModel):
    """Response from prediction endpoint."""

    model_config = ConfigDict(extra="ignore")

    signal: SignalType
    confidence: float
    explanation: str
    reasoning_steps: list[str] = []
