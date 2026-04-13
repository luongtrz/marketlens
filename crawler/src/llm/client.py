"""LLM API wrapper — calls AIHub or direct LLM for article enrichment."""

from shared.models.article import RawArticle, EnrichedFields


class LLMClient:
    """Enriches articles with sentiment, summary, and factors via LLM.

    Calls AIHub /sentiment and /factors endpoints, or falls back to direct
    LLM API if AIHub is unavailable.

    Args:
        aihub_url: Base URL for the AIHub service.
        enable_summary: Whether to generate article summaries.
    """

    def __init__(self, aihub_url: str, enable_summary: bool = False) -> None:
        self._aihub_url = aihub_url
        self._enable_summary = enable_summary

    async def enrich(self, article: RawArticle) -> EnrichedFields:
        """Enrich a raw article with LLM-derived fields.

        Args:
            article: The raw article to enrich.

        Returns:
            EnrichedFields containing sentiment_score, summary, and factors.
        """
        raise NotImplementedError

    async def summarize(self, text: str) -> str | None:
        """Generate a summary of the given text via LLM.

        Args:
            text: Article body text.

        Returns:
            Summary string or None if summarization is disabled.
        """
        raise NotImplementedError
