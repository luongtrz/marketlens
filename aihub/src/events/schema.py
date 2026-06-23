"""Pydantic schemas for the event extraction endpoint."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from shared.models.event import EventRecord


class EventExtractionRequest(BaseModel):
    symbol: str
    title: str
    summary: str | None = None
    factors: list[str] = Field(default_factory=list)
    article_id: str | None = None
    published_at: datetime | None = None


class EventExtractionResponse(BaseModel):
    events: list[EventRecord] = Field(default_factory=list)
    method: Literal["rule_based", "llm"] = "rule_based"
