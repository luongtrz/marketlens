"""RSS feed entry parser — converts raw feed entries into RawArticle objects."""

import datetime
from typing import Any

from shared.models.article import RawArticle

from crawler.src.rss.title_hints import normalize_article_title


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
        title = str(entry.get("title") or "").strip()
        url = (
            entry.get("link")
            or entry.get("id")
            or entry.get("guid")
            or entry.get("url")
            or ""
        )
        url = str(url).strip()

        published = None
        if entry.get("published_parsed"):
            try:
                published = datetime.datetime(
                    *entry["published_parsed"][:6], tzinfo=datetime.timezone.utc
                )
            except Exception:
                published = None
        elif entry.get("updated_parsed"):
            try:
                published = datetime.datetime(
                    *entry["updated_parsed"][:6], tzinfo=datetime.timezone.utc
                )
            except Exception:
                published = None

        summary = str(entry.get("summary") or entry.get("description") or "").strip() or None

        resolved = normalize_article_title(title if title else None, url)

        return RawArticle(
            title=resolved,
            url=url,
            source=source_name,
            category=category,
            published=published,
            text=summary,
        )
