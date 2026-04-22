"""OpenAI-compatible LLM client (works with any OpenAI-spec API)."""

from openai import AsyncOpenAI

from aihub.src.llm.base import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI / OpenAI-compatible backend.

    Args:
        api_key: API key.
        model: Model identifier (default: gpt-4o-mini).
        base_url: Override base URL for self-hosted or proxy endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict] = []  # type: ignore[type-arg]
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""
