"""SQLAlchemy / ORM models for the Crawler ingestion database."""

# TODO: Define SQLAlchemy table models for IngestionRecord persistence
# These should mirror the shared.models.article.IngestionRecord schema.

# Example:
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
# from sqlalchemy import String, Float, DateTime, JSON
#
# class Base(DeclarativeBase):
#     pass
#
# class IngestionRecordORM(Base):
#     __tablename__ = "ingestion_records"
#     id: Mapped[str] = mapped_column(String, primary_key=True)
#     article_name: Mapped[str] = mapped_column(String)
#     source: Mapped[str] = mapped_column(String)
#     url: Mapped[str] = mapped_column(String, unique=True)
#     ...
