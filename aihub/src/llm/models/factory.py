"""AIModelFactory — central factory for creating task-specific AI models."""

from __future__ import annotations

from aihub.src.config import AIHubConfig
from aihub.src.llm.base import LLMClient
from aihub.src.llm.models.sentiment import SentimentModel


def _build_client(backend: str, config: AIHubConfig) -> LLMClient:
    """Create an LLMClient for the given backend name.

    Args:
        backend: One of ``"gemini"``, ``"openai"``, or ``"groq"``.
        config: Application configuration with API keys and model names.

    Returns:
        Configured LLMClient instance.

    Raises:
        ValueError: If the backend is not supported.
    """
    if backend == "gemini":
        from aihub.src.llm.gemini import GeminiClient

        return GeminiClient(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
        )
    if backend == "openai":
        from aihub.src.llm.openai import OpenAIClient

        return OpenAIClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.openai_base_url or None,
        )
    if backend == "groq":
        from aihub.src.llm.groq import GroqClient

        return GroqClient(
            api_key=config.groq_api_key,
            model=config.groq_model,
        )
    raise ValueError(
        f"Unsupported LLM backend: {backend!r}. "
        f"Expected 'gemini', 'openai', or 'groq'."
    )


class AIModelFactory:
    """Factory for creating task-specific AI models.

    Builds SentimentModel from configuration and provides LLM clients for
    prediction and other tasks. Each task can be assigned its own LLM backend
    (or fall back to the global ``llm_backend`` setting). LLM clients are
    cached so the same backend is not instantiated twice.

    Args:
        config: AIHubConfig with API keys, model names, and per-task backends.
    """

    def __init__(self, config: AIHubConfig) -> None:
        self._config = config
        self._clients: dict[str, LLMClient] = {}

    def _resolve_backend(self, task_backend: str) -> str:
        """Return the effective backend name for a task.

        If ``task_backend`` is empty, falls back to the global ``llm_backend``.
        """
        return task_backend or self._config.llm_backend

    def _get_or_create_client(self, backend: str) -> LLMClient:
        """Get a cached client or create and cache a new one."""
        if backend not in self._clients:
            self._clients[backend] = _build_client(backend, self._config)
        return self._clients[backend]

    def get_default_client(self) -> LLMClient:
        """Return the LLM client for the global default backend.

        Useful for components that don't fit the task-model pattern
        (e.g. FactorExtractor / SKGP).
        """
        return self._get_or_create_client(self._config.llm_backend)

    def get_client(self, backend: str) -> LLMClient:
        """Return a cached LLM client for the specified backend.

        Args:
            backend: One of ``"gemini"``, ``"openai"``, or ``"groq"``.
        """
        return self._get_or_create_client(backend)

    def create_sentiment_model(self) -> SentimentModel:
        """Create a SentimentModel (CryptoBert HTTP API).

        Sentiment uses its own HTTP endpoint, not a generic LLM backend.
        """
        return SentimentModel(
            hf_model_url=self._config.hf_model_path,
            model_path=self._config.model_path,
        )
