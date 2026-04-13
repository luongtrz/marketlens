"""Factor-related data models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class FactorType(str, Enum):
    """Classification of a market factor."""

    MACRO = "macro"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    ON_CHAIN = "on_chain"
    EXCHANGE = "exchange"


class Factor(BaseModel):
    """A single factor extracted from article text."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: FactorType
    polarity: float  # -1 to 1: negative = bearish signal
    confidence: float  # 0 to 1


class NormalizedFactor(BaseModel):
    """A cleaned, weighted, and enriched factor ready for downstream use."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: FactorType
    weight: float  # 0–1
    polarity: float  # -1 to 1
    sector: str | None = None
    related_symbols: list[str] = []
    source_article_id: str
    observed_at: datetime
