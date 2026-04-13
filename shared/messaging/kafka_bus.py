"""Kafka implementation of the MessageBus interface."""

from typing import Any, Callable, Awaitable

from shared.messaging.base import MessageBus


class KafkaBus(MessageBus):
    """Message bus backed by Apache Kafka."""

    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        # TODO: Initialize Kafka producer/consumer

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def subscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError
