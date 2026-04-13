"""Read StockMemRecord by date or ID."""

from datetime import date

from shared.models.memory import StockMemRecord


class RecordReader:
    """Reads StockMemRecord objects from the database.

    Args:
        db_url: SQLAlchemy-compatible async database URL.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    async def get_by_id(self, record_id: str) -> StockMemRecord | None:
        """Retrieve a record by its ID.

        Args:
            record_id: UUID string.

        Returns:
            StockMemRecord or None if not found.
        """
        raise NotImplementedError

    async def get_by_date(self, target_date: date, symbol: str) -> StockMemRecord | None:
        """Retrieve a record by date and symbol.

        Args:
            target_date: Target date.
            symbol: Trading pair.

        Returns:
            StockMemRecord or None if not found.
        """
        raise NotImplementedError
