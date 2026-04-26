# aihub.src.llm

from aihub.src.llm.base import LLMClient
from aihub.src.llm.models.factory import AIModelFactory

__all__ = ["LLMClient", "AIModelFactory", "build_llm_client"]


def build_llm_client(config) -> LLMClient:  # type: ignore[type-arg]
    """Factory: create the configured LLM backend from AIHubConfig.

    .. deprecated::
        Prefer ``AIModelFactory`` for task-specific model creation.
        This function is kept for backward compatibility and for components
        that need a raw LLMClient (e.g. FactorExtractor / SKGP).
    """
    if config.llm_backend == "gemini":
        from aihub.src.llm.gemini import GeminiClient
        return GeminiClient(api_key=config.gemini_api_key, model=config.gemini_model)
    elif config.llm_backend == "openai":
        from aihub.src.llm.openai import OpenAIClient
        return OpenAIClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.openai_base_url or None,
        )
    elif config.llm_backend == "groq":
        from aihub.src.llm.groq import GroqClient
        return GroqClient(api_key=config.groq_api_key, model=config.groq_model)
    else:
        raise ValueError(
            f"Unsupported llm_backend: {config.llm_backend!r}. Expected 'gemini' or 'openai' or 'groq'."
        )   
