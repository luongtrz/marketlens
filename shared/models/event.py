"""Structured daily event-memory models shared across pipeline services."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_group: str
    event_type: str
    entities: list[str] = Field(default_factory=list)
    polarity: float = 0.0
    confidence: float = 0.0
    observed_at: datetime | None = None
    description: str | None = None


class DailyEventState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: date
    symbol: str
    events: list[EventRecord] = Field(default_factory=list)
    article_count: int = 0
    source_count: int = 0
    source_diversity: float = 0.0
    temporal_span_hours: float = 0.0
    novelty_7d: float = 0.0
    novelty_30d: float = 0.0
    incremental_information: float = 0.0
    dominant_event_groups: list[str] = Field(default_factory=list)
