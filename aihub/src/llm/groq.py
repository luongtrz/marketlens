"""Groq LLM client — native Groq SDK backend with response cleaning for OSS models."""

import re
from dataclasses import dataclass, field

from groq import AsyncGroq

from aihub.src.llm.base import LLMClient


@dataclass
class GroqResponse:
    """Structured response from GroqClient with reasoning extracted."""

    content: str
    reasoning_steps: list[str] = field(default_factory=list)


class GroqClient(LLMClient):
    """Groq backend for open-source models (e.g. openai/gpt-oss-120b).

    Uses the native Groq Python SDK (``groq.AsyncGroq``) to access
    Groq-specific parameters such as ``reasoning_effort`` and
    ``tools=[{"type": "browser_search"}]`` that are not part of the
    standard OpenAI SDK interface.

    Adds response cleaning to strip instruction tags that OSS models
    sometimes leak (e.g. ``<think>``, ``[INST]``, role prefixes).

    Provides ``extract_reasoning_steps`` to capture the model's
    chain-of-thought from ``<think>`` blocks before cleaning.

    Args:
        api_key: Groq API key.
        model: Model identifier (default: openai/gpt-oss-120b).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    async def _generate_raw(self, prompt: str, system: str | None = None) -> str:
        """Internal method to generate a raw response using the native Groq SDK."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.5,
            max_completion_tokens=20461,
            top_p=1,
            reasoning_effort="high",
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
            tools=[{"type": "browser_search"}],  # type: ignore[list-item]
        )
        return resp.choices[0].message.content or ""

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response and clean OSS model instruction artifacts."""
        raw = await self._generate_raw(prompt, system)
        return self._clean_instruction_tags(raw)

    async def generate_with_reasoning(
        self, prompt: str, system: str | None = None
    ) -> GroqResponse:
        """Generate a response, extracting reasoning steps from ``<think>`` blocks.

        The model's chain-of-thought (inside ``<think>...</think>``) is parsed
        into a list of reasoning steps matching ``PredictResponse.reasoning_steps``.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Returns:
            GroqResponse with cleaned content and extracted reasoning_steps.
        """
        raw = await self._generate_raw(prompt, system)
        reasoning_steps = self.extract_reasoning_steps(raw)
        clean = self._clean_instruction_tags(raw)
        return GroqResponse(content=clean, reasoning_steps=reasoning_steps)

    @staticmethod
    def extract_reasoning_steps(text: str) -> list[str]:
        """Extract reasoning steps from ``<think>...</think>`` blocks.

        Parses the model's chain-of-thought into individual steps.
        Each non-empty line or numbered item inside the think block
        becomes a separate step in the returned list.

        Args:
            text: Raw LLM response that may contain ``<think>`` blocks.

        Returns:
            List of reasoning step strings. Empty list if no think blocks found.
        """
        if not text:
            return []

        # Extract all <think> block contents
        think_blocks = re.findall(
            r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL
        )
        if not think_blocks:
            return []

        steps: list[str] = []
        for block in think_blocks:
            for line in block.strip().splitlines():
                line = line.strip()
                # Strip numbered prefixes like "1.", "2)", "- ", "* "
                line = re.sub(r"^\d+[.)]\s*", "", line)
                line = re.sub(r"^[-*•]\s*", "", line)
                line = line.strip()
                if line:
                    steps.append(line)

        return steps

    @staticmethod
    def _clean_instruction_tags(text: str) -> str:
        """Strip leaked instruction tags from open-source model output.

        Removes ``<think>...</think>`` blocks, ``<|...|>`` tokens,
        ``[INST]``/``[/INST]`` markers, and bare role prefixes.
        """
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<\|.*?\|>", "", text)
        text = re.sub(r"\[/?(INST|SYS|USER|ASSISTANT).*?\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(assistant|user|system):\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def parse_json(text: str) -> dict:  # type: ignore[type-arg]
        """Parse a cleaned LLM response as JSON.

        Strips markdown code fences (triple-backtick json blocks) that models
        sometimes wrap around their output before parsing.

        Args:
            text: Cleaned text response (after instruction tag removal).

        Returns:
            Parsed JSON dict.
        """
        import json
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())
        return json.loads(text)  # type: ignore[no-any-return]
