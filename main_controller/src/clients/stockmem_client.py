"""StockMem module HTTP client."""

from shared.http_client import get_client
from shared.models.memory import StockMemRecord, SimilarRecord


class StockMemClient:
    """Async HTTP client for the StockMem module.

    Args:
        base_url: StockMem service base URL.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def health_check(self) -> bool:
        """Check if the StockMem service is healthy."""
        raise NotImplementedError

    async def save(self, record: StockMemRecord) -> str:
        """Save a record to StockMem.

        Args:
            record: The daily record to persist.

        Returns:
            Record ID.
        """
        raise NotImplementedError

    async def search(self, query: StockMemRecord, k: int = 5) -> list[SimilarRecord]:
        """Search for similar historical records.

        Args:
            query: Current record.
            k: Number of similar records.

        Returns:
            List of SimilarRecord objects.
        """
        raise NotImplementedError
