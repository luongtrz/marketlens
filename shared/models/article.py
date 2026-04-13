"""Article and ingestion data models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class RawArticle(BaseModel):
    """An article as initially parsed from an RSS feed entry, before enrichment."""

    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    source: str
    category: str
    published: datetime | None = None
    text: str | None = None


class EnrichedFields(BaseModel):
    """Fields produced by the LLM enrichment step."""

    model_config = ConfigDict(extra="ignore")

    sentiment_score: float  # -1.0 (bearish) to 1.0 (bullish)
    summary: str | None = None
    factors: list[str] = []


class IngestionRecord(BaseModel):
    """A fully enriched article record persisted in the Ingestion Database."""

    model_config = ConfigDict(extra="ignore")

    id: str
    article_name: str
    source: str
    url: str
    date_published: datetime
    date_crawled: datetime
    summary: str | None = None  # Nullable — generated only if enabled
    sentiment_score: float  # -1.0 (bearish) to 1.0 (bullish)
    sentiment_label: str  # "bullish" | "bearish" | "neutral"
    factors: list[str]  # Raw factor strings from LLM
    raw_text: str | None = None  # Original article body (may be truncated)
    metadata: dict[str, Any] = {}  # Source-specific extra fields
