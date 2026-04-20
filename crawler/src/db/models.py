"""SQLAlchemy / ORM models for the Crawler ingestion database."""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for ORM models."""


class IngestionRecordORM(Base):
    """Database table for persisted ingestion records."""

    __tablename__ = "ingestion_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    article_name: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    url: Mapped[str] = mapped_column(String(2000), unique=True, nullable=False, index=True)
    date_published: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    date_crawled: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text(), nullable=True)
    sentiment_score: Mapped[float] = mapped_column(Float(), nullable=False)
    sentiment_label: Mapped[str] = mapped_column(String(32), nullable=False)
    factors: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    raw_text: Mapped[str | None] = mapped_column(Text(), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
