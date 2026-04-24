"""Crawler module HTTP client — reads latest news from Supabase (PostgREST)."""

import os

from shared.models.article import IngestionRecord
from shared.supabase_news import check_supabase_rest_reachable, fetch_news_articles_from_supabase


class CrawlerClient:
    """Supplies recent news articles for the pipeline (backed by Supabase).

    The crawler process writes into the same ``news_articles`` table; this client
    reads via PostgREST using ``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY`` or
    ``SUPABASE_ANON_KEY``.

    Args:
        base_url: Reserved for a future HTTP crawler API; unused for Supabase reads.
    """

    def __init__(self, base_url: str = "") -> None:
        self._base_url = base_url

    async def health_check(self) -> bool:
        """True when Supabase env is set and PostgREST returns 2xx."""
        if not (os.getenv("SUPABASE_URL") and (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        )):
            return False
        return await check_supabase_rest_reachable()

    async def get_latest(self, symbol: str) -> list[IngestionRecord]:
        """Get the latest news rows, optionally filtered by ``symbol`` (e.g. BTCUSDT)."""
        return await fetch_news_articles_from_supabase(limit=50, symbol=symbol)
