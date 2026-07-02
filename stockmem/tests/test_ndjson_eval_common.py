from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stockmem.scripts.ndjson_eval_common import (
    HistoricalRow,
    actual_signal,
    fixed_knn_signal,
    knn_returns_signal,
    retrieve_fixed_knn,
    summarize_predictions,
)
from stockmem.src.models import MarketSnapshot, SimilarRecord, StockMemRecord


def _record(day: date, ret7: float, *, factor=(1.0, 0.0), indicator=(0.0, 1.0), price=(1.0, 1.0)) -> HistoricalRow:
    record = StockMemRecord(
        date=day,
        symbol="BTC",
        sentiment_score=0.0,
        sentiment_label="neutral",
        factors=[],
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
            recent_candles=[],
            indicators={},
            source="binance",
        ),
        future_return_1d=ret7 / 4,
        future_return_3d=ret7 / 2,
        future_return_7d=ret7,
        future_return_15d=ret7,
        future_return_30d=ret7,
    )
    return HistoricalRow(
        record=record,
        factor_vec=__import__("numpy").asarray(factor, dtype=float),
        indicator_vec=__import__("numpy").asarray(indicator, dtype=float),
        price_vec=__import__("numpy").asarray(price, dtype=float),
        split="test",
    )


def test_retrieve_fixed_knn_prefers_highest_weighted_similarity() -> None:
    query = _record(date(2026, 1, 10), 3.0)
    a = _record(date(2026, 1, 1), 4.0, factor=(1.0, 0.0), indicator=(0.0, 1.0), price=(1.0, 1.0))
    b = _record(date(2026, 1, 2), -4.0, factor=(0.0, 1.0), indicator=(1.0, 0.0), price=(0.0, 0.0))
    similar = retrieve_fixed_knn(query, [a, b], weights=(1.0, 0.0, 0.0), k=1)
    assert similar[0].record.date == a.record.date


def test_fixed_knn_signal_and_knn_returns_signal_agree_on_positive_cases() -> None:
    similar = [
        SimilarRecord(record=_record(date(2026, 1, 1), 6.0).record, similarity=0.9),
        SimilarRecord(record=_record(date(2026, 1, 2), 4.0).record, similarity=0.8),
    ]
    fixed_signal, _ = fixed_knn_signal(similar, threshold=2.0)
    returns_signal, _ = knn_returns_signal(similar, threshold=2.0)
    assert fixed_signal == "BUY"
    assert returns_signal == "BUY"


def test_summarize_predictions_reports_coverage_and_hit_rate() -> None:
    rows = [
        {
            "predicted_signal": "BUY",
            "actual_return_7d": 3.0,
            "confidence": 0.7,
            "top5_same_sign": True,
        },
        {
            "predicted_signal": "HOLD",
            "actual_return_7d": 0.5,
            "confidence": 0.5,
            "top5_same_sign": False,
        },
        {
            "predicted_signal": "SELL",
            "actual_return_7d": -4.0,
            "confidence": 0.8,
            "top5_same_sign": True,
        },
    ]
    metrics = summarize_predictions("demo", rows, label_threshold=2.0)
    assert metrics.overall_acc == pytest.approx(1.0)
    assert metrics.coverage == pytest.approx(2 / 3)
    assert metrics.hit_at_5_same_sign == pytest.approx(2 / 3)
    assert metrics.actual_counts["BUY"] == 1


def test_actual_signal_uses_hold_band() -> None:
    assert actual_signal(3.0, 2.0) == "BUY"
    assert actual_signal(-3.0, 2.0) == "SELL"
    assert actual_signal(1.0, 2.0) == "HOLD"
