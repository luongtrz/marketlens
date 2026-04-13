"""URL-based article deduplication backed by Redis SET or in-memory set."""


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

    async def is_seen(self, url: str) -> bool:
        """Check if a URL has already been processed.

        Args:
            url: The article URL to check.

        Returns:
            True if already seen; False otherwise.
        """
        raise NotImplementedError

    async def mark_seen(self, url: str) -> None:
        """Mark a URL as seen/processed.

        Args:
            url: The article URL to record.
        """
        raise NotImplementedError
