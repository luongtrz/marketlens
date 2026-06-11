"""StockMem record and similarity search models."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shared.models.event import DailyEventState
from shared.models.market import MarketSnapshot


class StockMemRecord(BaseModel):
    """A daily record stored in StockMem for similarity search."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None  # Assigned on write
    date: date
    symbol: str
    sentiment_score: float
    sentiment_label: str = "neutral"
    factors: list[str]
    normalized_factors: list[Any] = Field(
        default_factory=list
    )  # Factor | NormalizedFactor | dict
    market_snapshot: MarketSnapshot
    factor_vector: list[float] = Field(
        default_factory=list
    )  # Pre-computed 75d factor vector (from FactorLedge)
    summary: str | None = None
    article_ids: list[str] = Field(default_factory=list)
    article_sources: list[str] = Field(default_factory=list)
    article_published_at: list[datetime] = Field(default_factory=list)
    event_state: DailyEventState | None = None
    event_vector: list[float] = Field(default_factory=list)
    future_return_1d: float | None = None
    future_return_3d: float | None = None
    future_return_7d: float | None = None
    future_return_15d: float | None = None
    future_return_30d: float | None = None
    run_id: str | None = None


class SimilarRecord(BaseModel):
    """A past record retrieved by similarity search, with similarity score."""

    model_config = ConfigDict(extra="ignore")

    record: StockMemRecord
    similarity: float  # Cosine similarity [0, 1]
    outcome: str | None = None  # What happened after this date, if known
    event_match: dict[str, float] = Field(default_factory=dict)
    retriever_version: str = "fixed_knn_v1"
