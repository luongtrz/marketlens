"""
StockMem + History Rhymes split-vector embedder.

Produces four L2-normalized vectors per record:
  - event_vec     (85d)  = type/group occurrences + dissemination/novelty
  - factor_vec    (75d)  = typeVec(62) + groupVec(13)
  - indicator_vec (5d)   = z-scored [msi, rsi, chosen_sentiment, fgi, price_change_pct]
  - price_vec     (60d)  = OHLCV features [close_returns(20) | ranges(20) | volumes(20)]

Search combines them via weighted cosine:
  score = w1 * sim(factor) + w2 * sim(indicator) + w3 * sim(price)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

from ..models import CandleData, StockMemRecord
from .event_memory import EVENT_DIM, build_event_vector
from .taxonomy import (
    NUM_GROUPS,
    NUM_TYPES,
    build_group_vector_dense,
    build_group_vector_from_types,
    build_type_vector_dense,
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
SentimentSource = Literal["sentiment_score", "finbert", "auto"]


@dataclass(frozen=True)
class SplitEmbedding:
    event_vec: np.ndarray
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


def _resolve_sentiment_value(
    record: StockMemRecord,
    sentiment_source: SentimentSource,
) -> float:
    if sentiment_source == "sentiment_score":
        return float(record.sentiment_score)
    finbert_score = record.finbert_sentiment_score
    if sentiment_source == "finbert":
        if finbert_score is not None:
            return float(finbert_score)
        return float(record.sentiment_score)
    if finbert_score is not None:
        return float(finbert_score)
    return float(record.sentiment_score)


def _extract_raw_numerical(
    record: StockMemRecord,
    *,
    sentiment_source: SentimentSource,
) -> np.ndarray:
    snap = record.market_snapshot
    return np.array(
        [
            float(snap.msi),
            float(snap.rsi),
            _resolve_sentiment_value(record, sentiment_source),
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
    Produces SplitEmbedding(event=85d, factor=75d, indicator=5d, price=60d).

    Indicator z-score stats come from the corpus; call rebuild_corpus() before
    embedding queries so query indicators use the same normalization.
    """

    def __init__(self, sentiment_source: SentimentSource = "sentiment_score") -> None:
        self._sentiment_source = sentiment_source
        self._stats = NormStats()

    @property
    def stats(self) -> NormStats:
        return self._stats

    def rebuild_corpus(self, records: Iterable[StockMemRecord]) -> None:
        rows = [
            _extract_raw_numerical(r, sentiment_source=self._sentiment_source)
            for r in records
        ]
        if not rows:
            self._stats = NormStats()
            return

        mat = np.vstack(rows).astype(np.float64)
        mean = mat.mean(axis=0)
        variance = np.maximum(mat.var(axis=0), 1e-8)
        std = np.sqrt(variance)
        self._stats = NormStats(mean=mean, std=std, count=mat.shape[0])

    def update_corpus_with_record(self, record: StockMemRecord) -> None:
        """Incrementally update indicator normalization stats (Welford)."""
        raw = _extract_raw_numerical(
            record,
            sentiment_source=self._sentiment_source,
        ).astype(np.float64)
        stats = self._stats
        if stats.count <= 0:
            self._stats = NormStats(
                mean=raw,
                std=np.ones(INDICATOR_DIM, dtype=np.float64),
                count=1,
            )
            return

        n = stats.count
        mean_old = stats.mean
        var_old = np.square(stats.std)
        m2_old = var_old * n

        n_new = n + 1
        delta = raw - mean_old
        mean_new = mean_old + (delta / n_new)
        delta2 = raw - mean_new
        m2_new = m2_old + delta * delta2
        var_new = np.maximum(m2_new / n_new, 1e-8)
        std_new = np.sqrt(var_new)

        self._stats = NormStats(mean=mean_new, std=std_new, count=n_new)

    def _z_score(self, raw: np.ndarray) -> np.ndarray:
        stats = self._stats
        denom = np.where(stats.std > 1e-8, stats.std, 1e-8)
        z = (raw - stats.mean) / denom
        z = np.where(np.isfinite(z), z, 0.0)
        return np.clip(z, -Z_SCORE_CLIP, Z_SCORE_CLIP)

    def embed_split(self, record: StockMemRecord) -> SplitEmbedding:
        raw_event_vec = np.asarray(record.event_vector, dtype=np.float32)
        if raw_event_vec.size == EVENT_DIM:
            event_vec = _l2_normalize(raw_event_vec)
        else:
            event_vec = _l2_normalize(build_event_vector(record.event_state))

        # ── Factor vector (dense) ────────────────────────────────────────────
        # Always recompute from factor phrases using group-propagated dense encoding.
        # Dense: active type=1.0, same-group inactive types=0.3, group intensity ∈ [0.6, 1.0].
        # This gives meaningful cosine similarity between days sharing the same thematic
        # group even when exact phrases differ — fixing the sparse binary problem.
        type_vec = np.array(build_type_vector_dense(record.factors), dtype=np.float32)
        group_vec = np.array(build_group_vector_dense(record.factors), dtype=np.float32)

        # Fallback: if no factors matched taxonomy, try FactorType from normalized_factors
        if not any(type_vec) and record.normalized_factors:
            factor_types = [
                nf.get("type") if isinstance(nf, dict) else getattr(nf, "type", None)
                for nf in record.normalized_factors
            ]
            fallback_groups = np.array(
                build_group_vector_from_types(factor_types), dtype=np.float32
            )
            group_vec = np.maximum(group_vec, fallback_groups * 0.6)

        factor_vec = _l2_normalize(np.concatenate([type_vec, group_vec]))

        raw = _extract_raw_numerical(
            record,
            sentiment_source=self._sentiment_source,
        )
        z = self._z_score(raw)
        indicator_vec = _l2_normalize(z.astype(np.float32))

        price_features = compute_price_features(record.market_snapshot.candles)
        price_vec = _l2_normalize(price_features)

        return SplitEmbedding(
            event_vec=event_vec,
            factor_vec=factor_vec,
            indicator_vec=indicator_vec,
            price_vec=price_vec,
        )

    def embed(self, record: StockMemRecord) -> np.ndarray:
        """Legacy 140d embedding; event features remain learned-retriever only."""
        split = self.embed_split(record)
        indicator_scaled = split.indicator_vec.astype(np.float32) * np.float32(ALPHA_NUMERIC)
        joint = np.concatenate([split.factor_vec, indicator_scaled, split.price_vec])
        return _l2_normalize(joint)
