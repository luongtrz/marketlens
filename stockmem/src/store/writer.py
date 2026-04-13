"""Persist daily StockMemRecord to relational database."""

from shared.models.memory import StockMemRecord


class RecordWriter:
    """Writes StockMemRecord objects to the database and triggers embedding.

    Args:
        db_url: SQLAlchemy-compatible async database URL.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    async def save(self, record: StockMemRecord) -> str:
        """Persist a record to the relational DB and compute its vector embedding.

        Also triggers the embedder to compute and store the vector in the
        vector index for similarity search.

        Args:
            record: The StockMemRecord to persist.

        Returns:
            Record ID (UUID string).
        """
        raise NotImplementedError
