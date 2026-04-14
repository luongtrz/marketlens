from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class MarketSnapshot(BaseModel):
    rsi: float
    macd_hist: float
    msi: Optional[float] = None
    fear_greed_index: Optional[float] = None
    price_change_pct: Optional[float] = None


class StockMemRecord(BaseModel):
    id: Optional[str] = None
    date: date
    symbol: str
    sentiment_score: float
    factors: list[str]
    market_snapshot: MarketSnapshot
    summary: Optional[str] = None
    article_ids: list[str] = Field(default_factory=list)


class SimilarRecord(BaseModel):
    record: StockMemRecord
    similarity: float
    outcome: Optional[str] = None


class RecordCreateRequest(BaseModel):
    record: StockMemRecord


class RecordCreateResponse(BaseModel):
    id: str


class SearchRequest(BaseModel):
    query: StockMemRecord
    k: int = 5


class SearchResponse(BaseModel):
    results: list[SimilarRecord]


class HealthResponse(BaseModel):
    status: str
    vector_backend: str
    db_url: str
