from __future__ import annotations

from aihub.src.llm.groq import GroqClient


def test_parse_retry_after_seconds_with_minutes_and_fractional_seconds() -> None:
    message = "Please try again in 40m57.216s."
    delay = GroqClient._parse_retry_after_seconds(message)
    assert delay is not None
    assert 2457.0 < delay < 2458.0


def test_parse_retry_after_seconds_returns_none_when_absent() -> None:
    assert GroqClient._parse_retry_after_seconds("no retry hint here") is None
