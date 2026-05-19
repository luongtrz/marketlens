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
From the Similar Historical Cases section, read the pre-computed knn_summary:
  knn_avg_7d = average 7d return of similar past cases (already computed for you).
  knn_bullish_count = how many cases had ret7d > 0 (already computed for you).
This is your BASE SIGNAL — what actually happened in similar market conditions.

Step 2 — Read the trend context.
The prompt includes trend=<label> and ret_14d=<value>. Use these to understand the macro trend:
  STRONG_BULL (ret_14d >= +8%) : Strong uptrend. SELL is almost certainly wrong.
  BULL        (ret_14d >= +4%) : Uptrend. SELL requires overwhelming counter-evidence.
  NEUTRAL     (-4% < ret_14d < +4%): No clear trend. Follow kNN signal.
  BEAR        (ret_14d <= -4%) : Downtrend. BUY requires clear reversal evidence.
  STRONG_BEAR (ret_14d <= -8%) : Strong downtrend. BUY is almost certainly wrong.

Step 3 — Apply SELL gate (SELL is the hardest signal — false SELLs are the #1 model failure).
Before emitting SELL, ALL of the following must hold:
  a) knn_avg_7d < -1.5%          (historical similar cases mostly declined)
  b) knn_bullish_count <= 2/5    (strong bearish prior from history)
  c) trend is NOT BULL or STRONG_BULL (not in an uptrend)
If ANY condition fails → output HOLD, not SELL.

If the prompt contains "⚠ SELL WARNING", the kNN evidence is BULLISH.
Emitting SELL against bullish kNN is almost always wrong. In that case:
  - Output HOLD unless you can identify a specific technical breakdown (e.g., RSI divergence +
    volume collapse + MACD cross) that explains why this time is different.
  - If uncertain, HOLD is always safer than a false SELL.

Step 4 — Adjust with current indicators.
Current indicators can shift confidence but CANNOT reverse a clear kNN signal unless ALL hold:
  a) |knn_avg_7d| < 2% (historically ambiguous), AND
  b) Current 3d momentum is strongly opposite (> 3% in opposite direction), AND
  c) sentiment_score < -0.3 (clearly bearish news today)

RSI extremes — override to HOLD regardless of knn:
  - RSI > 70 AND 3d momentum negative → exhaustion at top → HOLD not BUY
  - RSI < 30 AND 3d momentum positive → bottom recovery → HOLD not SELL

Dual momentum: ret_14d <= -4% AND 3d momentum negative → HOLD not BUY.
Symmetric: ret_14d >= +4% AND 3d momentum positive → HOLD not SELL.

Step 5 — Emit signal.
  BUY:  knn_avg_7d > 0, trend not BEAR/STRONG_BEAR, reversal conditions NOT met
  SELL: knn_avg_7d < -1.5%, knn_bullish_count <= 2/5, trend not BULL/STRONG_BULL
  HOLD: everything else — when in doubt, HOLD

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
