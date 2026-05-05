from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandleData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketSnapshot(BaseModel):
    """Accepts the old flat format AND the shared nested-indicators format.

    Old (tests/legacy):  MarketSnapshot(rsi=50.0, msi=50.0, candles=[...])
    New (main_controller): MarketSnapshot(indicators={"rsi": 50.0, ...}, recent_candles=[...], ...)

    extra="allow" preserves shared-format fields (symbol, timestamp, etc.) so they
    round-trip correctly when the client parses the returned record as shared.MarketSnapshot.
    """

    model_config = ConfigDict(extra="allow")

    # Flat fields used by the embedder
    rsi: float = 50.0
    macd_hist: float = 0.0
    msi: float = 0.0
    fear_greed_index: float = 0.0
    price_change_pct: float = 0.0
    candles: list[CandleData] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _bridge_shared_format(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        indic = data.get("indicators") or {}
        if indic:
            data.setdefault("rsi", float(indic.get("rsi", 50.0)))
            data.setdefault("macd_hist", float(indic.get("macd_hist", 0.0)))
            data.setdefault("msi", float(indic.get("msi", 0.0)))
            data.setdefault("fear_greed_index", float(indic.get("fear_greed_index", 0.0)))
            data.setdefault("price_change_pct", float(indic.get("price_change_pct", 0.0)))
        if not data.get("candles"):
            recent = data.get("recent_candles") or []
            data["candles"] = [
                {
                    "open": c.get("open", 0.0) if isinstance(c, dict) else getattr(c, "open", 0.0),
                    "high": c.get("high", 0.0) if isinstance(c, dict) else getattr(c, "high", 0.0),
                    "low": c.get("low", 0.0) if isinstance(c, dict) else getattr(c, "low", 0.0),
                    "close": c.get("close", 0.0) if isinstance(c, dict) else getattr(c, "close", 0.0),
                    "volume": c.get("volume", 0.0) if isinstance(c, dict) else getattr(c, "volume", 0.0),
                }
                for c in recent
            ]
        return data


class StockMemRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    date: date
    symbol: str
    sentiment_score: float
    sentiment_label: str = "neutral"
    factors: list[str]
    normalized_factors: list[Any] = Field(default_factory=list)
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
