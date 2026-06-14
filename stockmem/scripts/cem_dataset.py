from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from collections.abc import Callable
from typing import Literal, Sequence

import numpy as np

from stockmem.scripts.optimize_weights import Row, weighted_similarity


Direction = Literal[-1, 0, 1]
TRAIN_END = date(2024, 12, 24)
VAL_START = date(2025, 1, 1)
VAL_END = date(2025, 6, 23)
TEST_START = date(2025, 7, 1)
TEST_END = date(2026, 5, 1)


@dataclass(frozen=True)
class LabeledRow:
    row: Row
    parsed_date: date
    split: str
    direction: Direction
    band_value: float
    causal_volatility: float

    @property
    def blocks(self) -> tuple[np.ndarray, ...]:
        legacy = (self.row.factor_vec, self.row.indicator_vec, self.row.price_vec)
        if self.row.event_vec.size:
            return (self.row.event_vec, *legacy)
        return legacy


@dataclass(frozen=True)
class MinedCandidates:
    positives: tuple[LabeledRow, ...]
    positive_weights: tuple[float, ...]
    negatives: tuple[LabeledRow, ...]


def is_mature(candidate_date: date, query_date: date, horizon_days: int = 7) -> bool:
    return candidate_date + timedelta(days=horizon_days) <= query_date


def assign_split(day: date) -> str:
    if day <= TRAIN_END:
        return "train"
    if VAL_START <= day <= VAL_END:
        return "val"
    if TEST_START <= day <= TEST_END:
        return "test"
    return "embargo"


def _causal_sigma(rows: Sequence[Row], index: int, window: int = 30) -> float | None:
    query_date = date.fromisoformat(rows[index].date)
    available = [
        row.future_return_7d
        for row in rows[:index]
        if is_mature(date.fromisoformat(row.date), query_date)
    ]
    sample = np.asarray(available[-window:], dtype=np.float64)
    if sample.size < 2:
        return None
    return float(sample.std(ddof=0))


def label_rows(
    rows: Sequence[Row],
    *,
    band: str = "0.5sigma",
    sigma_multiplier: float = 0.5,
    fixed_band: float = 1.0,  # percent: ±1% FLAT band (future_return_7d is in %)
) -> list[LabeledRow]:
    labeled: list[LabeledRow] = []
    for index, row in enumerate(rows):
        parsed = date.fromisoformat(row.date)
        if band == "fixed":
            threshold = fixed_band
            volatility = fixed_band / max(sigma_multiplier, 1e-12)
        elif band == "0.5sigma":
            sigma = _causal_sigma(rows, index)
            threshold = fixed_band if sigma is None else sigma_multiplier * sigma
            volatility = (
                fixed_band / max(sigma_multiplier, 1e-12) if sigma is None else sigma
            )
        else:
            raise ValueError(f"Unsupported band: {band}")

        value = row.future_return_7d
        direction: Direction = 0
        if value > threshold:
            direction = 1
        elif value < -threshold:
            direction = -1
        labeled.append(
            LabeledRow(
                row=row,
                parsed_date=parsed,
                split=assign_split(parsed),
                direction=direction,
                band_value=threshold,
                causal_volatility=volatility,
            )
        )
    return labeled


def matured_pool(
    rows: Sequence[LabeledRow],
    query: LabeledRow,
    *,
    guard: bool = True,
) -> list[LabeledRow]:
    return [
        candidate
        for candidate in rows
        if candidate.parsed_date < query.parsed_date
        and (not guard or is_mature(candidate.parsed_date, query.parsed_date))
    ]


def _regime_similarity(anchor: LabeledRow, candidate: LabeledRow) -> float:
    epsilon = 1e-8
    log_ratio = abs(
        np.log(
            (anchor.causal_volatility + epsilon)
            / (candidate.causal_volatility + epsilon)
        )
    )
    return float(np.exp(-log_ratio))


def _outcome_similarity(anchor: LabeledRow, candidate: LabeledRow) -> float:
    scale = max(
        anchor.causal_volatility,
        candidate.causal_volatility,
        anchor.band_value,
        candidate.band_value,
        1e-8,
    )
    difference = abs(
        abs(anchor.row.future_return_7d) - abs(candidate.row.future_return_7d)
    )
    return float(np.exp(-difference / scale))


def teacher_relevance(
    anchor: LabeledRow,
    candidate: LabeledRow,
    *,
    baseline_similarity: float,
) -> float:
    """Outcome-derived proxy for FinSeer's unavailable LLM relevance reward."""
    if anchor.direction == 0 or candidate.direction != anchor.direction:
        return 0.0
    surface_similarity = float(np.clip((baseline_similarity + 1.0) / 2.0, 0.0, 1.0))
    return (
        0.45 * _outcome_similarity(anchor, candidate)
        + 0.35 * _regime_similarity(anchor, candidate)
        + 0.20 * surface_similarity
    )


def mine_candidates(
    anchor: LabeledRow,
    pool: Sequence[LabeledRow],
    *,
    weights: tuple[float, float, float],
    hard_negs: int,
    positive_count: int = 3,
    flat_negs: int = 2,
    learned_score: Callable[[LabeledRow, LabeledRow], float] | None = None,
) -> MinedCandidates | None:
    if anchor.direction == 0:
        return None
    positives = [row for row in pool if row.direction == anchor.direction]
    opposite = [row for row in pool if row.direction == -anchor.direction]
    flat = [row for row in pool if row.direction == 0]
    if not positives or not opposite:
        return None

    w1, w2, w3 = weights
    baseline_scores = {
        id(row): weighted_similarity(anchor.row, row.row, w1, w2, w3)
        for row in pool
    }
    positives_scored = sorted(
        positives,
        key=lambda row: teacher_relevance(
            anchor,
            row,
            baseline_similarity=baseline_scores[id(row)],
        ),
        reverse=True,
    )
    selected_positives = positives_scored[: max(1, positive_count)]
    positive_weights = tuple(
        teacher_relevance(
            anchor,
            row,
            baseline_similarity=baseline_scores[id(row)],
        )
        for row in selected_positives
    )

    def select_hard(rows: list[LabeledRow], count: int) -> list[LabeledRow]:
        if count <= 0:
            return []
        shortlist_size = max(count, count * 4)
        shortlist = sorted(
            rows,
            key=lambda row: baseline_scores[id(row)],
            reverse=True,
        )[:shortlist_size]
        if learned_score is not None:
            shortlist.sort(
                key=lambda row: learned_score(anchor, row),
                reverse=True,
            )
        return shortlist[:count]

    negatives = [
        *select_hard(opposite, max(1, hard_negs)),
        *select_hard(flat, flat_negs),
    ]
    return MinedCandidates(
        positives=tuple(selected_positives),
        positive_weights=positive_weights,
        negatives=tuple(negatives),
    )
