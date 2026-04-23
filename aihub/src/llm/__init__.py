# aihub.src.llm

from aihub.src.llm.base import LLMClient


def build_llm_client(config) -> LLMClient:  # type: ignore[type-arg]
    """Factory: create the configured LLM backend from AIHubConfig."""
    if config.llm_backend == "gemini":
        from aihub.src.llm.gemini import GeminiClient
        return GeminiClient(api_key=config.gemini_api_key, model=config.gemini_model)
    if config.llm_backend == "openai":
        from aihub.src.llm.openai import OpenAIClient
        return OpenAIClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.openai_base_url or None,
        )
    raise ValueError(
        f"Unsupported llm_backend: {config.llm_backend!r}. Expected 'gemini' or 'openai'."
    )
