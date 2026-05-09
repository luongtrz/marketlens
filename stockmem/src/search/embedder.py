"""
StockMem + History Rhymes split-vector embedder.

Produces three L2-normalized vectors per record:
  - factor_vec    (75d)  = typeVec(62) + groupVec(13)
  - indicator_vec (5d)   = z-scored [msi, rsi, sentiment_score, fgi, price_change_pct]
  - price_vec     (60d)  = OHLCV features [close_returns(20) | ranges(20) | volumes(20)]

Search combines them via weighted cosine:
  score = w1 * sim(factor) + w2 * sim(indicator) + w3 * sim(price)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..models import CandleData, StockMemRecord
from .taxonomy import (
    NUM_GROUPS,
    NUM_TYPES,
    build_group_vector,
    build_group_vector_from_types,
    build_type_vector,
)


RETURNS_WINDOW = 20
FACTOR_DIM = NUM_TYPES + NUM_GROUPS  # 75
INDICATOR_DIM = 5
PRICE_DIM = 3 * RETURNS_WINDOW  # 60
JOINT_DIM = FACTOR_DIM + INDICATOR_DIM + PRICE_DIM  # 140

MAX_ABS_RETURN = 0.30
MAX_ABS_RANGE = 0.50
MAX_ABS_VOL_CHG = 5.0
Z_SCORE_CLIP = 6.0
ALPHA_NUMERIC = 0.5  # scales indicator block when packed into joint vector


@dataclass(frozen=True)
class SplitEmbedding:
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray


@dataclass
class NormStats:
    mean: np.ndarray = field(default_factory=lambda: np.zeros(INDICATOR_DIM, dtype=np.float64))
    std: np.ndarray = field(default_factory=lambda: np.ones(INDICATOR_DIM, dtype=np.float64))
    count: int = 0


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _extract_raw_numerical(record: StockMemRecord) -> np.ndarray:
    snap = record.market_snapshot
    return np.array(
        [
            float(snap.msi),
            float(snap.rsi),
            float(record.sentiment_score),
            float(snap.fear_greed_index),
            float(snap.price_change_pct),
        ],
        dtype=np.float64,
    )


def _pad_or_slice(arr: list[float], window: int) -> list[float]:
    sliced = arr[-window:] if len(arr) >= window else list(arr)
    while len(sliced) < window:
        sliced.insert(0, 0.0)
    return sliced


def _compute_close_returns(candles: list[CandleData], window: int = RETURNS_WINDOW) -> list[float]:
    if len(candles) < 2:
        return [0.0] * window
    out: list[float] = []
    for i in range(1, len(candles)):
        prev = candles[i - 1].close
        curr = candles[i].close
        if prev == 0:
            out.append(0.0)
            continue
        raw = (curr - prev) / prev
        clipped = max(-MAX_ABS_RETURN, min(MAX_ABS_RETURN, raw))
        out.append(clipped if np.isfinite(clipped) else 0.0)
    return _pad_or_slice(out, window)


def _compute_ranges(candles: list[CandleData], window: int = RETURNS_WINDOW) -> list[float]:
    if len(candles) < 2:
        return [0.0] * window
    out: list[float] = []
    for c in candles[1:]:
        if c.close == 0:
            out.append(0.0)
            continue
        raw = (c.high - c.low) / c.close
        clipped = max(0.0, min(MAX_ABS_RANGE, raw))
        out.append(clipped if np.isfinite(clipped) else 0.0)
    return _pad_or_slice(out, window)


def _compute_volume_changes(candles: list[CandleData], window: int = RETURNS_WINDOW) -> list[float]:
    if len(candles) < 2:
        return [0.0] * window
    out: list[float] = []
    for i in range(1, len(candles)):
        prev_vol = candles[i - 1].volume
        curr_vol = candles[i].volume
        if prev_vol == 0 or not np.isfinite(prev_vol):
            out.append(0.0)
            continue
        raw = (curr_vol - prev_vol) / prev_vol
        clipped = max(-MAX_ABS_VOL_CHG, min(MAX_ABS_VOL_CHG, raw))
        out.append(clipped if np.isfinite(clipped) else 0.0)
    return _pad_or_slice(out, window)


def compute_price_features(candles: list[CandleData], window: int = RETURNS_WINDOW) -> np.ndarray:
    """60d vector: [close_returns(20) | intraday_ranges(20) | volume_changes(20)]."""
    returns = _compute_close_returns(candles, window)
    ranges = _compute_ranges(candles, window)
    volumes = _compute_volume_changes(candles, window)
    return np.array(returns + ranges + volumes, dtype=np.float32)


class RecordEmbedder:
    """
    Produces SplitEmbedding(factor=75d, indicator=5d, price=60d).

    Indicator z-score stats come from the corpus; call rebuild_corpus() before
    embedding queries so query indicators use the same normalization.
    """

    def __init__(self) -> None:
        self._stats = NormStats()

    @property
    def stats(self) -> NormStats:
        return self._stats

    def rebuild_corpus(self, records: Iterable[StockMemRecord]) -> None:
        rows = [_extract_raw_numerical(r) for r in records]
        if not rows:
            self._stats = NormStats()
            return

        mat = np.vstack(rows).astype(np.float64)
        mean = mat.mean(axis=0)
        variance = np.maximum(mat.var(axis=0), 1e-8)
        std = np.sqrt(variance)
        self._stats = NormStats(mean=mean, std=std, count=mat.shape[0])

    def _z_score(self, raw: np.ndarray) -> np.ndarray:
        stats = self._stats
        denom = np.where(stats.std > 1e-8, stats.std, 1e-8)
        z = (raw - stats.mean) / denom
        z = np.where(np.isfinite(z), z, 0.0)
        return np.clip(z, -Z_SCORE_CLIP, Z_SCORE_CLIP)

    def embed_split(self, record: StockMemRecord) -> SplitEmbedding:
        # ── Factor vector ───────────────────────────────────────────────────
        if record.factor_vector and len(record.factor_vector) == 75:
            factor_vec = _l2_normalize(
                np.array(record.factor_vector, dtype=np.float32)
            )
        else:
            type_vec = np.array(build_type_vector(record.factors), dtype=np.float32)
            group_vec = np.array(build_group_vector(record.factors), dtype=np.float32)

            # When factor names don't match the taxonomy (free-form AIHub output),
            # fall back to using FactorType from normalized_factors to populate group bits.
            if not any(group_vec) and record.normalized_factors:
                factor_types = [
                    nf.get("type") if isinstance(nf, dict) else getattr(nf, "type", None)
                    for nf in record.normalized_factors
                ]
                fallback = np.array(build_group_vector_from_types(factor_types), dtype=np.float32)
                group_vec = np.maximum(group_vec, fallback)

            factor_vec = _l2_normalize(np.concatenate([type_vec, group_vec]))

        raw = _extract_raw_numerical(record)
        z = self._z_score(raw)
        indicator_vec = _l2_normalize(z.astype(np.float32))

        price_features = compute_price_features(record.market_snapshot.candles)
        price_vec = _l2_normalize(price_features)

        return SplitEmbedding(
            factor_vec=factor_vec,
            indicator_vec=indicator_vec,
            price_vec=price_vec,
        )

    def embed(self, record: StockMemRecord) -> np.ndarray:
        """Legacy concat embedding for FAISS index fallback. Not used for weighted search."""
        split = self.embed_split(record)
        indicator_scaled = split.indicator_vec.astype(np.float32) * np.float32(ALPHA_NUMERIC)
        joint = np.concatenate([split.factor_vec, indicator_scaled, split.price_vec])
        return _l2_normalize(joint)
