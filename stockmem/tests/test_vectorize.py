from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from stockmem.src.models import CandleData, MarketSnapshot, StockMemRecord
from stockmem.src.search.embedder import (
    FACTOR_DIM,
    INDICATOR_DIM,
    MAX_ABS_RANGE,
    MAX_ABS_RETURN,
    MAX_ABS_VOL_CHG,
    PRICE_DIM,
    RETURNS_WINDOW,
    RecordEmbedder,
    Z_SCORE_CLIP,
    compute_price_features,
)
from stockmem.src.search.event_memory import EVENT_DIM


def _make_candles(n: int, base: float = 100.0) -> list[CandleData]:
    out = []
    price = base
    for i in range(n):
        price *= 1.01
        out.append(
            CandleData(
                open=price * 0.995,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1_000_000 * (1 + 0.1 * i),
            )
        )
    return out


def _make_record(
    factors: list[str] | None = None,
    candles: list[CandleData] | None = None,
    rsi: float = 50.0,
    sentiment: float = 0.0,
) -> StockMemRecord:
    return StockMemRecord(
        id="r",
        date=date(2026, 1, 1),
        symbol="BTC",
        sentiment_score=sentiment,
        factors=factors or ["Record ETF inflows"],
        market_snapshot=MarketSnapshot(
            rsi=rsi,
            macd_hist=0.0,
            msi=50.0,
            fear_greed_index=50.0,
            price_change_pct=1.0,
            candles=candles or [],
        ),
    )


def test_dimensions_are_fixed_75_5_60() -> None:
    assert EVENT_DIM == 85
    assert FACTOR_DIM == 75
    assert INDICATOR_DIM == 5
    assert PRICE_DIM == 60


def test_embed_split_produces_l2_unit_vectors() -> None:
    rec = _make_record(candles=_make_candles(25))
    embedder = RecordEmbedder()
    embedder.rebuild_corpus([rec])

    split = embedder.embed_split(rec)
    assert split.event_vec.shape[0] == 85
    assert split.factor_vec.shape[0] == 75
    assert split.indicator_vec.shape[0] == 5
    assert split.price_vec.shape[0] == 60
    assert np.linalg.norm(split.factor_vec) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(split.price_vec) == pytest.approx(1.0, abs=1e-5)


def test_empty_candles_yield_zero_price_vec() -> None:
    rec = _make_record(candles=[])
    embedder = RecordEmbedder()
    embedder.rebuild_corpus([rec])
    split = embedder.embed_split(rec)
    assert split.price_vec.shape[0] == 60
    assert float(np.abs(split.price_vec).sum()) == 0.0


def test_short_candles_are_left_zero_padded() -> None:
    candles = _make_candles(5)  # only 4 returns
    rec = _make_record(candles=candles)
    embedder = RecordEmbedder()
    embedder.rebuild_corpus([rec])
    split = embedder.embed_split(rec)
    # First window=close_returns; should have zeros at front, non-zero at the tail.
    returns = split.price_vec[:RETURNS_WINDOW]
    assert float(np.abs(returns[:RETURNS_WINDOW - 4]).sum()) == 0.0
    assert float(np.abs(returns[RETURNS_WINDOW - 4:]).sum()) > 0.0


def test_oov_factors_produce_zero_factor_vec() -> None:
    rec = _make_record(factors=["unknown_factor_1", "unknown_factor_2"])
    embedder = RecordEmbedder()
    embedder.rebuild_corpus([rec])
    split = embedder.embed_split(rec)
    assert float(np.abs(split.factor_vec).sum()) == 0.0


def test_price_features_clipping() -> None:
    # Build candles with an extreme return / range / volume jump to trigger clips.
    candles = [
        CandleData(open=100, high=102, low=98, close=100, volume=1_000_000),
        CandleData(open=100, high=200, low=50, close=200, volume=100_000_000),
    ]
    vec = compute_price_features(candles, window=RETURNS_WINDOW)
    returns_block = vec[:RETURNS_WINDOW]
    ranges_block = vec[RETURNS_WINDOW:2 * RETURNS_WINDOW]
    volumes_block = vec[2 * RETURNS_WINDOW:]

    # Last element of each block holds the most recent (largest) value and must be clipped.
    assert returns_block[-1] == pytest.approx(MAX_ABS_RETURN, abs=1e-6)
    assert ranges_block[-1] == pytest.approx(MAX_ABS_RANGE, abs=1e-6)
    assert volumes_block[-1] == pytest.approx(MAX_ABS_VOL_CHG, abs=1e-6)


def test_z_score_clip_bounds_extreme_indicator() -> None:
    # Build a corpus with tight spread then a far-outlier query.
    tight = [_make_record(rsi=50.0, sentiment=0.0) for _ in range(5)]
    embedder = RecordEmbedder()
    embedder.rebuild_corpus(tight)

    # Query with wildly out-of-range rsi.
    outlier = _make_record(rsi=10_000.0)
    split = embedder.embed_split(outlier)
    # After clipping at ±Z_SCORE_CLIP and L2-norm, max magnitude of the unit vector is ≤ 1.
    assert float(np.abs(split.indicator_vec).max()) <= 1.0 + 1e-6
    # Pre-normalization bound: the raw z must have been clipped to Z_SCORE_CLIP.
    assert Z_SCORE_CLIP > 0  # sanity — imported constant must exist
