"""URL-based article deduplication backed by Redis SET or in-memory set."""

from typing import Any


class Deduplicator:
    """Tracks seen article URLs to prevent duplicate processing.

    Backed by Redis SET in production or an in-memory set for testing.

    Args:
        backend: "redis" or "memory".
        redis_url: Redis connection URL (required if backend is "redis").
    """

    def __init__(self, backend: str = "memory", redis_url: str | None = None) -> None:
        self._backend = backend
        self._redis_url = redis_url
        self._seen: set[str] = set()  # In-memory fallback
        self._redis: Any = None

    async def is_seen(self, url: str) -> bool:
        """Check if a URL has already been processed.

        Args:
            url: The article URL to check.

        Returns:
            True if already seen; False otherwise.
        """
        if not url:
            return True
        if self._backend == "redis":
            r = await self._get_redis()
            if r is None:
                return url in self._seen
            return bool(await r.sismember("crawler:seen_urls", url))
        return url in self._seen

    async def mark_seen(self, url: str) -> None:
        """Mark a URL as seen/processed.

        Args:
            url: The article URL to record.
        """
        if not url:
            return
        if self._backend == "redis":
            r = await self._get_redis()
            if r is None:
                self._seen.add(url)
                return
            await r.sadd("crawler:seen_urls", url)
            return
        self._seen.add(url)

    async def _get_redis(self) -> Any:
        if self._redis is not None:
            return self._redis
        if not self._redis_url:
            return None
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            return self._redis
        except Exception:
            self._redis = None
            return None
