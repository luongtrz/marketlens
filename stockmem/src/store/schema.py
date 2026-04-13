"""Database table models for StockMem storage."""

# TODO: Define SQLAlchemy table models for StockMemRecord persistence
# These should mirror the shared.models.memory.StockMemRecord schema.

# Example:
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
# from sqlalchemy import String, Float, Date, JSON
#
# class Base(DeclarativeBase):
#     pass
#
# class StockMemRecordORM(Base):
#     __tablename__ = "stockmem_records"
#     id: Mapped[str] = mapped_column(String, primary_key=True)
#     date: Mapped[date] = mapped_column(Date)
#     symbol: Mapped[str] = mapped_column(String)
#     sentiment_score: Mapped[float] = mapped_column(Float)
#     ...
