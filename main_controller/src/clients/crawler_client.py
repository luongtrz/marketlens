"""Crawler module HTTP client — reads latest news from Supabase (PostgREST)."""

import os
from datetime import datetime

from shared.cache import RedisCache
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

    def __init__(
        self,
        base_url: str = "",
        *,
        cache: RedisCache | None = None,
        news_ttl_seconds: int = 120,
    ) -> None:
        self._base_url = base_url
        self._cache = cache
        self._news_ttl_seconds = news_ttl_seconds

    async def health_check(self) -> bool:
        """True when Supabase env is set and PostgREST returns 2xx."""
        if not (os.getenv("SUPABASE_URL") and (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        )):
            return False
        return await check_supabase_rest_reachable()

    async def get_latest(
        self,
        symbol: str,
        *,
        limit: int = 50,
        offset: int = 0,
        lite: bool = False,
        publish_gte: datetime | None = None,
        publish_lte: datetime | None = None,
    ) -> list[IngestionRecord]:
        """Get recent news rows, optionally filtered by ``symbol`` (e.g. BTCUSDT).

        ``lite=True`` skips full article body in PostgREST — faster for UI lists; pipeline
        keeps ``lite=False`` for richer text when needed.

        ``publish_gte`` / ``publish_lte`` narrow by ``publish_at`` (inclusive) in Supabase.

        ``offset`` is paired with ``limit`` when reading without symbol (PostgREST OFFSET).
        With symbol filters, paging is bounded by scanning in ``shared.supabase_news``.
        """
        offset = max(0, offset)
        should_cache = self._cache is not None and lite and limit <= 50
        key = None
        if should_cache:
            key = self._cache.key(
                "news",
                "latest",
                symbol.upper() or "all",
                limit,
                offset,
                "lite",
                publish_gte.isoformat() if publish_gte else "none",
                publish_lte.isoformat() if publish_lte else "none",
            )
            cached = await self._cache.get_json(key)
            if cached is not None:
                return [IngestionRecord.model_validate(item) for item in cached]

        records = await fetch_news_articles_from_supabase(
            limit=limit,
            offset=offset,
            symbol=symbol,
            lite=lite,
            publish_gte=publish_gte,
            publish_lte=publish_lte,
        )
        if should_cache and key is not None:
            await self._cache.set_json(
                key,
                [record.model_dump(mode="json") for record in records],
                self._news_ttl_seconds,
            )
        return records
