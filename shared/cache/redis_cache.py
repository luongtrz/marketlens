"""Small Redis JSON cache wrapper with graceful degradation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class RedisCache:
    """Async JSON cache backed by Redis.

    Cache failures are logged and treated as misses so Redis never becomes a
    hard dependency for the application path using it.
    """

    def __init__(self, redis_url: str, *, namespace: str = "marketlens") -> None:
        self._redis_url = redis_url
        self._namespace = namespace.strip(":")
        self._client: Any | None = None

    async def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            from redis.asyncio import Redis

            self._client = Redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            return self._client
        except Exception as exc:
            logger.warning("Redis cache disabled: %s", exc)
            return None

    def key(self, *parts: object) -> str:
        clean = ":".join(str(part).strip(":") for part in parts if part is not None)
        return f"{self._namespace}:{clean}"

    def hashed_key(self, prefix: str, payload: Any) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.key(prefix, digest)

    async def get_json(self, key: str) -> Any | None:
        client = await self._get_client()
        if client is None:
            return None
        try:
            raw = await client.get(key)
            if not raw:
                logger.debug("Redis cache miss: %s", key)
                return None
            logger.info("Redis cache hit: %s", key)
            return json.loads(raw)
        except Exception as exc:
            logger.debug("Redis get failed for %s: %s", key, exc)
            return None

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        client = await self._get_client()
        if client is None:
            return
        try:
            raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            await client.set(key, raw, ex=ttl_seconds)
            logger.debug("Redis cache set: %s ttl=%ss", key, ttl_seconds)
        except Exception as exc:
            logger.debug("Redis set failed for %s: %s", key, exc)

    async def get_or_set(
        self,
        key: str,
        ttl_seconds: int,
        loader: Callable[[], Awaitable[Any]],
    ) -> Any:
        cached = await self.get_json(key)
        if cached is not None:
            return cached
        value = await loader()
        await self.set_json(key, value, ttl_seconds)
        return value

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.aclose()
        except Exception:
            logger.debug("Redis close failed", exc_info=True)
        finally:
            self._client = None
