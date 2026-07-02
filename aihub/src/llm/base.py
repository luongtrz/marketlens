"""Abstract LLM client interface."""

import json
import re
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Common interface for all LLM backends (Gemini, OpenAI, etc.)."""

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Return raw text response from the model."""
        ...

    async def generate_json(self, prompt: str, system: str | None = None) -> dict:  # type: ignore[type-arg]
        """Generate a response and parse it as JSON.

        Strips markdown code fences (```json ... ```) that models sometimes wrap
        around their output before parsing.
        """
        raw = await self.generate(prompt, system)
        # Remove leading/trailing code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        try:
            return json.loads(raw)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            start = raw.find("{")
            if start != -1:
                decoder = json.JSONDecoder()
                parsed, _ = decoder.raw_decode(raw[start:])
                return parsed  # type: ignore[no-any-return]
            raise
