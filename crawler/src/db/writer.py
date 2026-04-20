"""Ingestion database writer — persists enriched article records."""

from shared.models.article import IngestionRecord
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from crawler.src.db.models import Base, IngestionRecordORM


class IngestionDBWriter:
    """Writes enriched IngestionRecord objects to the database.

    Args:
        db_url: SQLAlchemy-compatible async database URL.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._engine = create_async_engine(db_url, future=True)
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._initialized = False

    async def write(self, record: IngestionRecord) -> str:
        """Persist an IngestionRecord to the database.

        Args:
            record: The enriched article record to store.

        Returns:
            The record ID (UUID string).
        """
        if not self._initialized:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True

        async with self._session_factory() as session:
            existing = await session.scalar(
                select(IngestionRecordORM).where(IngestionRecordORM.url == record.url)
            )
            if existing is None:
                orm_obj = IngestionRecordORM(
                    id=record.id,
                    article_name=record.article_name,
                    source=record.source,
                    url=record.url,
                    date_published=record.date_published,
                    date_crawled=record.date_crawled,
                    summary=record.summary,
                    sentiment_score=record.sentiment_score,
                    sentiment_label=record.sentiment_label,
                    factors=record.factors,
                    raw_text=record.raw_text,
                    metadata_json=record.metadata,
                )
                session.add(orm_obj)
            else:
                existing.article_name = record.article_name
                existing.source = record.source
                existing.date_published = record.date_published
                existing.date_crawled = record.date_crawled
                existing.summary = record.summary
                existing.sentiment_score = record.sentiment_score
                existing.sentiment_label = record.sentiment_label
                existing.factors = record.factors
                existing.raw_text = record.raw_text
                existing.metadata_json = record.metadata

            await session.commit()
            return record.id
