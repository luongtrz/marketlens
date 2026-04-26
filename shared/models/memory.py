"""StockMem record and similarity search models."""

from datetime import date

from pydantic import BaseModel, ConfigDict

from shared.models.factor import NormalizedFactor
from shared.models.market import MarketSnapshot


class StockMemRecord(BaseModel):
    """A daily record stored in StockMem for similarity search."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None  # Assigned on write
    date: date
    symbol: str
    sentiment_score: float
    sentiment_label: str
    factors: list[str]
    normalized_factors: list[NormalizedFactor] = []
    market_snapshot: MarketSnapshot
    indicator_vec: list[float] = []  # Pre-computed market indicator vector
    summary: str | None = None
    article_ids: list[str] = []
    future_return_1d: float | None = None
    future_return_7d: float | None = None
    future_return_30d: float | None = None
    run_id: str | None = None


class SimilarRecord(BaseModel):
    """A past record retrieved by similarity search, with similarity score."""

    model_config = ConfigDict(extra="ignore")

    record: StockMemRecord
    similarity: float  # Cosine similarity [0, 1]
    outcome: str | None = None  # What happened after this date, if known
