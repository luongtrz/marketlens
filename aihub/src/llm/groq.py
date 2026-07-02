"""Groq LLM client — native Groq SDK backend with response cleaning for OSS models."""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field

from groq import AsyncGroq, BadRequestError, RateLimitError

from aihub.src.llm.base import LLMClient

logger = logging.getLogger(__name__)
_RETRY_AFTER_RE = re.compile(
    r"please try again in (?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?P<seconds>[0-9.]+)s",
    flags=re.IGNORECASE,
)


@dataclass
class GroqResponse:
    """Structured response from GroqClient with reasoning extracted."""

    content: str
    reasoning_steps: list[str] = field(default_factory=list)


class GroqClient(LLMClient):
    """Groq backend for chat-completions (e.g. Llama, Mixtral, GPT-OSS on Groq).

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
        model: Model identifier (see Groq docs; default suits free-tier input limits).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._max_completion_tokens = int(os.environ.get("AIHUB_GROQ_MAX_COMPLETION_TOKENS", "64"))

    @staticmethod
    def _parse_retry_after_seconds(message: str) -> float | None:
        match = _RETRY_AFTER_RE.search(message)
        if not match:
            return None
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = float(match.group("seconds") or 0.0)
        return hours * 3600 + minutes * 60 + seconds

    async def _create_with_backoff(self, create_kwargs: dict) -> object:
        while True:
            try:
                return await self._client.chat.completions.create(**create_kwargs)
            except RateLimitError as exc:
                delay_seconds = self._parse_retry_after_seconds(str(exc))
                if delay_seconds is None:
                    raise
                wait_seconds = min(delay_seconds + 1.0, 3600.0)
                logger.warning(
                    "Groq rate limit for model %s; sleeping %.1fs before retry",
                    self._model,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

    async def _generate_raw(self, prompt: str, system: str | None = None) -> str:
        """Internal method to generate a raw response using the native Groq SDK."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        create_kwargs: dict = dict(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.2,
            max_completion_tokens=self._max_completion_tokens,
            top_p=1,
            stream=False,
            stop=None,
            response_format={"type": "json_object"},
        )
        # Only GPT-OSS supports ``reasoning_effort``; Llama/Mixtral reject it.
        if "gpt-oss" in self._model.lower():
            create_kwargs["reasoning_effort"] = "low"

        try:
            resp = await self._create_with_backoff(create_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "json_validate_failed" not in message:
                raise
            # Some Groq-hosted models reject strict JSON mode even when they can
            # still follow a JSON-only prompt. Retry once without response_format.
            create_kwargs.pop("response_format", None)
            resp = await self._create_with_backoff(create_kwargs)
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
