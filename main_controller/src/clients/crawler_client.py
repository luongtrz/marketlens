"""Crawler module HTTP client."""

from shared.http_client import get_client
from shared.models.article import IngestionRecord


class CrawlerClient:
    """Async HTTP client for the Crawler module.

    Args:
        base_url: Crawler service base URL.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def health_check(self) -> bool:
        """Check if the Crawler service is healthy."""
        raise NotImplementedError

    async def get_latest(self, symbol: str) -> list[IngestionRecord]:
        """Get the latest enriched articles for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            List of IngestionRecord objects.
        """
        raise NotImplementedError
