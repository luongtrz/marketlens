"""OpenCode Go client used by the LLM gateway."""

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from llm_gateway.src.config import LLMGatewayConfig
from llm_gateway.src.schema import LLMDecisionResponse
from shared.models.prediction import SignalType


SIGNAL_ONLY_SYSTEM_PROMPT = """You are a deterministic crypto trading action classifier.
Output exactly one compact JSON object and no analysis:
{"signal":"BUY"} or {"signal":"HOLD"} or {"signal":"SELL"}.
Choose HOLD when evidence is mixed, weak, stale, or not actionable."""

WITH_REASON_SYSTEM_PROMPT = """You are a deterministic crypto trading action classifier.
Output exactly one compact JSON object and no analysis:
{"signal":"BUY|HOLD|SELL","reason":"short reason under 50 words"}.
Choose HOLD when evidence is mixed, weak, stale, or not actionable."""


class LLMGatewayError(RuntimeError):
    """Raised when the upstream LLM request cannot produce a valid response."""


@dataclass(frozen=True)
class ParsedDecision:
    """Internal normalized decision."""

    signal: SignalType
    reason: str | None


class OpenCodeGoClient:
    """Minimal OpenAI-compatible client for OpenCode Go chat completions."""

    def __init__(self, config: LLMGatewayConfig) -> None:
        self._config = config

    async def decide(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        include_reason: bool = False,
    ) -> LLMDecisionResponse:
        """Call OpenCode Go and normalize the result to BUY/HOLD/SELL."""
        selected_model = (model or self._config.default_model).strip()
        last_error: Exception | None = None

        for attempt in range(1, self._config.bounded_max_attempts + 1):
            try:
                content = await self._chat(
                    prompt=prompt,
                    model=selected_model,
                    system=system,
                    include_reason=include_reason,
                )
                parsed = self.parse_decision(content)
                reason = parsed.reason if include_reason else None
                return LLMDecisionResponse(
                    signal=parsed.signal,
                    reason=reason,
                )
            except Exception as exc:  # noqa: BLE001 - retry wraps upstream and parse failures.
                last_error = exc

        detail = str(last_error) if last_error else "unknown error"
        raise LLMGatewayError(f"OpenCode Go call failed after retries: {detail}") from last_error

    async def list_models(self) -> dict[str, Any]:
        """Fetch OpenCode Go model metadata for diagnostics."""
        headers = self._headers()
        timeout = httpx.Timeout(self._config.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(self._config.models_endpoint, headers=headers)
            response.raise_for_status()
            payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}

    async def _chat(
        self,
        prompt: str,
        model: str,
        system: str | None = None,
        include_reason: bool = False,
    ) -> str:
        headers = self._headers()
        system_prompt = system or (
            WITH_REASON_SYSTEM_PROMPT if include_reason else SIGNAL_ONLY_SYSTEM_PROMPT
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
        }
        timeout = httpx.Timeout(self._config.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self._config.opencode_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMGatewayError("OpenCode Go response did not include chat content") from exc

        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) for part in content)
        text = str(content).strip()
        if not text:
            raise LLMGatewayError("OpenCode Go returned empty content")
        return text

    def _headers(self) -> dict[str, str]:
        api_key = self._config.opencode_go_api_key.strip()
        if not api_key:
            raise LLMGatewayError("LLM gateway API key is not configured")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def parse_decision(raw: str) -> ParsedDecision:
        """Parse model output, accepting fenced JSON or a plain signal fallback."""
        text = raw.strip()
        parsed: Any | None = None

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None

        if isinstance(parsed, dict):
            signal_raw = str(parsed.get("signal", "")).strip().upper()
            reason_raw = parsed.get("reason")
            reason = str(reason_raw).strip() if reason_raw is not None else None
            if not reason:
                reason = None
            try:
                return ParsedDecision(signal=SignalType(signal_raw), reason=reason)
            except ValueError:
                pass

        signal_match = re.search(r"\b(BUY|HOLD|SELL)\b", text, flags=re.IGNORECASE)
        if signal_match:
            return ParsedDecision(signal=SignalType(signal_match.group(1).upper()), reason=None)

        raise LLMGatewayError("Could not parse BUY/HOLD/SELL from LLM response")
