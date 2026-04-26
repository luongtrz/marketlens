"""Abstract base class for all task-specific AI models."""

from abc import ABC, abstractmethod
from typing import Any

from aihub.src.llm.base import LLMClient


class BaseAIModel(ABC):
    """Base class for task-specific AI models (explain, predict, sentiment).

    Each subclass wraps a particular AI task with its own prompt template,
    input/output schema, and optional post-processing. The underlying LLM
    client is injected via constructor and can be ``None`` for models that
    do not use a generic LLM backend.

    Args:
        llm: Configured LLMClient backend, or ``None`` for non-LLM models.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the task-specific inference.

        Subclasses define their own argument signature and return type.
        """
        ...
