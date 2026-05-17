"""OpenCode Go client used by the LLM gateway."""

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from llm_gateway.src.config import LLMGatewayConfig
from llm_gateway.src.schema import LLMDecisionResponse
from shared.models.prediction import SignalType


PREDICT_SYSTEM_PROMPT = """
You are a crypto trading analyst. Decide one signal: BUY, SELL, or HOLD.
Primary objective: maximize directional correctness on the 7-day horizon.

Step 1 — Read historical evidence first.
From the Similar Historical Cases section, compute:
  knn_avg_7d = average of all provided ret7d values.
  knn_bullish_count = how many cases had ret7d > 0.
This is your BASE SIGNAL — what actually happened in similar conditions.

Step 2 — Adjust with current indicators.
Current RSI, MACD, and 3d price change can shift confidence, but CANNOT REVERSE the base signal
unless ALL three hold:
  a) |knn_avg_7d| < 2% (historically ambiguous), AND
  b) Current 3d momentum is strongly opposite (e.g., 3d return < -3% for reversing BUY base), AND
  c) sentiment_score < -0.3 (clearly bearish news today)
If not all three hold, trust the historical base signal.

RSI extreme warning — override to HOLD regardless of knn_avg_7d:
  - RSI > 70 AND 3d momentum already negative → price exhaustion at top → output HOLD not BUY
  - RSI < 30 AND 3d momentum already positive → bottom recovery underway → output HOLD not SELL

Dual momentum: if 14d price return <= -4% AND 3d momentum is negative, output HOLD not BUY —
the trend has already turned; historical knn_avg_7d may be from a different (bullish) regime.
Symmetric: if 14d return >= +4% AND 3d momentum positive, output HOLD not SELL.

Step 3 — Emit signal.
  BUY:  knn_avg_7d > 0, reversal conditions NOT met
  SELL: knn_avg_7d < 0, reversal conditions NOT met
  HOLD: |knn_avg_7d| < 1% (ambiguous), or strong conflict between base and current

Key rules:
  - Do NOT emit SELL just because news looks bad today. News semantics are embedded in the
    similar cases already. Trust the outcomes.
  - knn_bullish_count >= 4/5 → strong BUY prior. Override only with clear evidence of breakdown.
  - knn_bullish_count <= 1/5 → strong SELL prior. Override only with clear reversal evidence.

You MUST respond with a JSON object:
{
  "signal": "BUY",
  "reason": "2 concise sentences explaining the decision"
}

signal must be exactly: "BUY", "SELL", or "HOLD".
"""


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
        model_candidates: list[str] = []
        if selected_model:
            model_candidates.append(selected_model)
        for fb in self._config.parsed_fallback_models:
            if fb not in model_candidates:
                model_candidates.append(fb)

        last_error: Exception | None = None

        for candidate_model in model_candidates:
            for _attempt in range(1, self._config.bounded_max_attempts + 1):
                try:
                    content = await self._chat(
                        prompt=prompt,
                        model=candidate_model,
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
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        headers = self._headers()
        system_prompt = system or PREDICT_SYSTEM_PROMPT
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_output_tokens,
        }
        effective_timeout = timeout if timeout is not None else self._config.request_timeout_seconds
        timeout_obj = httpx.Timeout(effective_timeout)
        async with httpx.AsyncClient(timeout=timeout_obj) as client:
            response = await client.post(self._config.opencode_endpoint, headers=headers, json=payload, timeout=timeout_obj)
            response.raise_for_status()
            data = response.json()

        try:
            message = data["choices"][0]["message"]
            content = message.get("content")
            reasoning_content = message.get("reasoning_content")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMGatewayError("OpenCode Go response did not include chat content") from exc

        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) for part in content)
        text = str(content).strip() if content is not None else ""
        # Some providers can emit empty `content` but place useful text in `reasoning_content`.
        if not text and reasoning_content:
            text = str(reasoning_content).strip()
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
            reason_raw = parsed.get("reason", parsed.get("explanation"))
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
