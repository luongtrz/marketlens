"""RSS feed entry parser — converts raw feed entries into RawArticle objects."""

from typing import Any

from shared.models.article import RawArticle


class FeedParser:
    """Parses raw RSS/Atom feed entries into structured RawArticle objects."""

    def parse(self, entry: dict[str, Any], source_name: str, category: str) -> RawArticle:
        """Parse a single feed entry dictionary into a RawArticle.

        Args:
            entry: Raw feed entry from feedparser.
            source_name: Name of the feed source.
            category: Category classification.

        Returns:
            A RawArticle with extracted fields.
        """
        raise NotImplementedError
