"""Prediction-related data models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict

from shared.models.factor import NormalizedFactor
from shared.models.market import MarketSnapshot
from shared.models.memory import SimilarRecord, StockMemRecord


class SignalType(str, Enum):
    """Trading signal direction."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ExplainRequest(BaseModel):
    """Request payload for the /predict RAG endpoint."""

    model_config = ConfigDict(extra="ignore")

    current: StockMemRecord
    similar: list[SimilarRecord] = []


class PredictResponse(BaseModel):
    """Response from the AIHub /predict endpoint."""

    model_config = ConfigDict(extra="ignore")

    signal: SignalType
    confidence: float  # 0 to 1
    explanation: str
    reasoning_steps: list[str] = []


class PredictionResult(BaseModel):
    """Final assembled result from a complete pipeline run."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    symbol: str
    timestamp: datetime
    signal: SignalType
    confidence: float  # 0 to 1
    explanation: str  # Human-readable narrative
    reasoning_steps: list[str]  # Step-by-step reasoning
    similar_cases: list[SimilarRecord]  # The k retrieved cases
    sentiment_score: float
    key_factors: list[NormalizedFactor]
    market_snapshot: MarketSnapshot | None = None
    errors: list[str] = []  # Non-fatal errors encountered during run


class CEMRAGPrediction(BaseModel):
    """Probability-calibrated prediction from the CEM-RAG policy layer."""

    model_config = ConfigDict(extra="ignore")

    horizon: str = "7d"
    p_up: float
    p_down: float
    p_hold: float
    signal: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    tau: float
    explanation: str = ""
    retrieval_count: int = 0
