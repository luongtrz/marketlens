"""Abstract message bus interface for async inter-module communication."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable


class MessageBus(ABC):
    """Abstract message bus for publish/subscribe communication between modules."""

    @abstractmethod
    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a message to the specified topic.

        Args:
            topic: The topic/channel name (e.g. "factors.raw").
            payload: JSON-serializable message payload.
        """
        ...

    @abstractmethod
    async def subscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Subscribe to a topic with an async handler.

        Args:
            topic: The topic/channel name to subscribe to.
            handler: Async callable invoked for each received message.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Gracefully close the message bus connection."""
        ...
