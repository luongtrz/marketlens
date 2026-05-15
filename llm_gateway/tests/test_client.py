import pytest

from llm_gateway.src.client import LLMGatewayError, OpenCodeGoClient
from llm_gateway.src.schema import LLMDecisionResponse
from shared.models.prediction import SignalType


def test_parse_decision_from_json() -> None:
    parsed = OpenCodeGoClient.parse_decision(
        '{"signal":"BUY","reason":"breakout with strong volume"}'
    )

    assert parsed.signal == SignalType.BUY
    assert parsed.reason == "breakout with strong volume"


def test_parse_decision_from_fenced_json() -> None:
    parsed = OpenCodeGoClient.parse_decision(
        '```json\n{"signal":"SELL","reason":"support failed"}\n```'
    )

    assert parsed.signal == SignalType.SELL
    assert parsed.reason == "support failed"


def test_parse_decision_plain_signal_fallback() -> None:
    parsed = OpenCodeGoClient.parse_decision("Final answer: HOLD")

    assert parsed.signal == SignalType.HOLD
    assert parsed.reason is None


def test_decision_response_has_public_shape_only() -> None:
    response = LLMDecisionResponse(signal=SignalType.BUY)

    assert response.model_dump(mode="json", exclude_none=True) == {"signal": "BUY"}


def test_parse_decision_rejects_invalid_signal() -> None:
    with pytest.raises(LLMGatewayError):
        OpenCodeGoClient.parse_decision('{"signal":"WAIT","reason":"unclear"}')
