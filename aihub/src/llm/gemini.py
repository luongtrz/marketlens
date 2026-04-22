"""Gemini LLM client using google-generativeai."""

import google.generativeai as genai

from aihub.src.llm.base import LLMClient


class GeminiClient(LLMClient):
    """Google Gemini backend.

    Args:
        api_key: Google AI Studio / Vertex API key.
        model: Gemini model identifier (default: gemini-2.0-flash).
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        genai.configure(api_key=api_key)
        self._model_name = model

    async def generate(self, prompt: str, system: str | None = None) -> str:
        kwargs: dict = {}
        if system:
            kwargs["system_instruction"] = system
        model = genai.GenerativeModel(self._model_name, **kwargs)
        resp = await model.generate_content_async(prompt)
        return str(resp.text)
