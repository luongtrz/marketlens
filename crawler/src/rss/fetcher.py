"""RSS feed polling loop — fetches entries from configured feed sources."""

from pydantic import BaseModel

from shared.models.article import RawArticle


class FeedSource(BaseModel):
    """Configuration for a single RSS feed source."""

    name: str
    url: str
    category: str


class RSSFetcher:
    """Continuously polls RSS feeds and yields new articles.

    Args:
        sources: List of feed sources to poll.
        poll_interval_seconds: Seconds between polling cycles.
    """

    def __init__(self, sources: list[FeedSource], poll_interval_seconds: int) -> None:
        self._sources = sources
        self._poll_interval_seconds = poll_interval_seconds

    async def poll_forever(self) -> None:
        """Start the infinite polling loop. Polls all sources and processes new entries."""
        raise NotImplementedError

    async def fetch_one(self, source: FeedSource) -> list[RawArticle]:
        """Fetch and parse all entries from a single feed source.

        Args:
            source: The feed source to fetch.

        Returns:
            List of parsed RawArticle objects.
        """
        raise NotImplementedError
