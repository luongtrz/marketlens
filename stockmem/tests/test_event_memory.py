from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest

from stockmem.src.models import MarketSnapshot, StockMemRecord
from stockmem.src.search.event_memory import (
    EVENT_DIM,
    build_daily_event_state,
    build_event_vector,
)
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


def _record(
    day: str,
    factors: list[str],
    sources: list[str],
) -> StockMemRecord:
    parsed = date.fromisoformat(day)
    return StockMemRecord(
        date=parsed,
        symbol="BTC",
        sentiment_score=0.4,
        factors=factors,
        normalized_factors=[
            {
                "name": factor,
                "polarity": 0.7,
                "confidence": 0.8,
                "observed_at": datetime.combine(
                    parsed,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ),
            }
            for factor in factors
        ],
        market_snapshot=MarketSnapshot(),
        article_ids=[f"a{index}" for index in range(len(sources))],
        article_sources=sources,
        article_published_at=[
            datetime(parsed.year, parsed.month, parsed.day, index, tzinfo=timezone.utc)
            for index in range(len(sources))
        ],
    )


def test_event_state_quantifies_dissemination_and_temporal_span() -> None:
    record = _record(
        "2025-01-10",
        ["Record ETF inflows"],
        ["source-a", "source-b", "source-c"],
    )

    state = build_daily_event_state(record)

    assert state.article_count == 3
    assert state.source_count == 3
    assert state.source_diversity == pytest.approx(1.0)
    assert state.temporal_span_hours == 2.0
    assert state.novelty_7d == 1.0
    assert state.incremental_information > 0.0


def test_repeated_event_has_lower_point_in_time_novelty() -> None:
    previous = _record(
        "2025-01-05",
        ["Record ETF inflows"],
        ["source-a"],
    )
    previous_state = build_daily_event_state(previous)
    previous = previous.model_copy(update={"event_state": previous_state})
    repeated = _record(
        "2025-01-10",
        ["Record ETF inflows"],
        ["source-b"],
    )
    novel = _record(
        "2025-01-10",
        ["Major exchange hack"],
        ["source-b"],
    )

    repeated_state = build_daily_event_state(repeated, [previous])
    novel_state = build_daily_event_state(novel, [previous])

    assert repeated_state.novelty_7d == 0.0
    assert novel_state.novelty_7d == 1.0


def test_event_vector_contains_taxonomy_and_dissemination_features() -> None:
    state = build_daily_event_state(
        _record(
            "2025-01-10",
            ["Record ETF inflows"],
            ["source-a", "source-b"],
        )
    )
    vector = build_event_vector(state)

    assert vector.shape == (EVENT_DIM,)
    assert np.count_nonzero(vector) >= 8
    assert np.isfinite(vector).all()


def test_four_block_metric_uses_event_block() -> None:
    metric = LearnedDiagonalMetric(
        block_dims=(2, 1, 1, 1),
        diagonal=np.ones(5, dtype=np.float64),
        block_scales=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    query = (
        np.array([1.0, 0.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
    )
    same_event = (
        np.array([1.0, 0.0]),
        np.array([-1.0]),
        np.array([-1.0]),
        np.array([-1.0]),
    )
    different_event = (
        np.array([0.0, 1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
    )

    assert metric.score(query, same_event) > metric.score(query, different_event)
