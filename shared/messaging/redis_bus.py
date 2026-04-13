"""Redis Streams implementation of the MessageBus interface."""

from typing import Any, Callable, Awaitable

from shared.messaging.base import MessageBus


class RedisBus(MessageBus):
    """Message bus backed by Redis Streams."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        # TODO: Initialize Redis connection

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def subscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError
