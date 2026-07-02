"""Tests for current-context AI evaluation helpers."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from aihub.src.llm.base import LLMClient
from aihub.src.predict.ablation import predict_current_context, render_current_context
from stockmem.src.models import MarketSnapshot, StockMemRecord


class FakeLLM(LLMClient):
    async def generate(self, prompt: str, system: str | None = None) -> str:
        assert "Sentiment 1d:" in prompt
        return json.dumps(
            {
                "signal": "BUY",
                "confidence": 0.62,
                "explanation": "stub",
                "reasoning_steps": ["stub"],
            }
        )


def _record(summary: str = "ETF inflow title. Exchange reserves fall.") -> StockMemRecord:
    return StockMemRecord(
        date=date(2026, 1, 10),
        symbol="BTC",
        sentiment_score=0.41,
        sentiment_label="bullish",
        factors=["ETF inflow"],
        normalized_factors=[],
        market_snapshot=MarketSnapshot(
            symbol="BTC",
            timestamp=datetime(2026, 1, 10, tzinfo=timezone.utc),
            ohlcv={
                "timestamp": datetime(2026, 1, 10, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 110.0,
                "low": 95.0,
                "close": 108.0,
                "volume": 1000.0,
                "interval": "1d",
            },
            recent_candles=[
                {
                    "timestamp": datetime(2026, 1, 7, tzinfo=timezone.utc),
                    "open": 95.0,
                    "high": 96.0,
                    "low": 90.0,
                    "close": 92.0,
                    "volume": 1000.0,
                    "interval": "1d",
                },
                {
                    "timestamp": datetime(2026, 1, 8, tzinfo=timezone.utc),
                    "open": 92.0,
                    "high": 98.0,
                    "low": 91.0,
                    "close": 95.0,
                    "volume": 1000.0,
                    "interval": "1d",
                },
                {
                    "timestamp": datetime(2026, 1, 9, tzinfo=timezone.utc),
                    "open": 95.0,
                    "high": 102.0,
                    "low": 94.0,
                    "close": 101.0,
                    "volume": 1000.0,
                    "interval": "1d",
                },
                {
                    "timestamp": datetime(2026, 1, 10, tzinfo=timezone.utc),
                    "open": 101.0,
                    "high": 110.0,
                    "low": 100.0,
                    "close": 108.0,
                    "volume": 1000.0,
                    "interval": "1d",
                },
            ],
            indicators={"rsi": 62.5, "macd_hist": 1.1, "price_change_pct": 4.0, "msi": 0.7},
            source="binance",
        ),
        summary=summary,
        future_return_7d=3.0,
    )


def test_render_current_context_contains_market_and_news_fields() -> None:
    text = render_current_context(_record())
    assert "Close:" in text
    assert "Change 1d:" in text
    assert "Sentiment 1d:" in text
    assert "Titles:" in text


def test_render_current_context_truncates_long_summary() -> None:
    text = render_current_context(_record(summary="x" * 1000))
    assert "Titles:" in text
    assert len(text) < 600


def test_render_current_context_omits_nonscalar_indicators() -> None:
    record = _record()
    record.market_snapshot.indicators = {
        "rsi": 62.5,
        "macd_hist": 1.1,
        "series": [1, 2, 3],
        "bands": {"upper": 1.0},
    }
    text = render_current_context(record)
    assert "rsi=62.5000" in text
    assert "macd_hist=1.1000" in text
    assert "series" not in text
    assert "bands" not in text


@pytest.mark.asyncio
async def test_predict_current_context_parses_model_output() -> None:
    out = await predict_current_context(FakeLLM(), _record())
    assert out["predicted_signal"] == "BUY"
    assert out["confidence"] == pytest.approx(0.62)
    assert out["prompt_chars"] > 0
