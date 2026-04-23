from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class CandleData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketSnapshot(BaseModel):
    rsi: float
    macd_hist: float = 0.0
    msi: float = 0.0
    fear_greed_index: float = 0.0
    price_change_pct: float = 0.0
    candles: list[CandleData] = Field(default_factory=list)


class StockMemRecord(BaseModel):
    id: Optional[str] = None
    date: date
    symbol: str
    sentiment_score: float
    factors: list[str]
    market_snapshot: MarketSnapshot
    summary: Optional[str] = None
    article_ids: list[str] = Field(default_factory=list)
    future_return_1d: Optional[float] = None
    future_return_7d: Optional[float] = None
    future_return_30d: Optional[float] = None


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
