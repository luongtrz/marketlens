"""Ingestion database writer — persists enriched article records."""

from shared.models.article import IngestionRecord


class IngestionDBWriter:
    """Writes enriched IngestionRecord objects to the database.

    Args:
        db_url: SQLAlchemy-compatible async database URL.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    async def write(self, record: IngestionRecord) -> str:
        """Persist an IngestionRecord to the database.

        Args:
            record: The enriched article record to store.

        Returns:
            The record ID (UUID string).
        """
        raise NotImplementedError
