from __future__ import annotations

from collections import Counter
from datetime import timedelta
import math
from typing import Sequence

import numpy as np

from ..models import StockMemRecord
from shared.models.event import DailyEventState, EventRecord
from .taxonomy import (
    GROUP_INDEX,
    NUM_GROUPS,
    NUM_TYPES,
    TYPE_INDEX,
    get_factor_group,
    get_factor_type,
)


EVENT_SCALAR_DIM = 10
EVENT_DIM = NUM_TYPES + NUM_GROUPS + EVENT_SCALAR_DIM


def _normalized_entropy(values: Sequence[str]) -> float:
    cleaned = [value.strip().lower() for value in values if value.strip()]
    counts = Counter(cleaned)
    if len(counts) <= 1:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=np.float64)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(counts))


def _event_type_set(state: DailyEventState | None) -> set[str]:
    if state is None:
        return set()
    return {event.event_type for event in state.events if event.event_type}


def _record_event_types(record: StockMemRecord) -> set[str]:
    from_state = _event_type_set(record.event_state)
    if from_state:
        return from_state
    return {
        event_type
        for factor in record.factors
        if (event_type := get_factor_type(factor)) is not None
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _novelty(
    current_types: set[str],
    history: Sequence[StockMemRecord],
    *,
    current_date,
    window_days: int,
) -> float:
    if not current_types:
        return 0.0
    start = current_date - timedelta(days=window_days)
    similarities = [
        _jaccard(current_types, _record_event_types(record))
        for record in history
        if start <= record.date < current_date
    ]
    return 1.0 - max(similarities, default=0.0)


def build_daily_event_state(
    record: StockMemRecord,
    history: Sequence[StockMemRecord] = (),
) -> DailyEventState:
    normalized = {
        str(
            item.get("name", "") if isinstance(item, dict) else getattr(item, "name", "")
        ): item
        for item in record.normalized_factors
    }
    events: list[EventRecord] = []
    seen: set[tuple[str, str]] = set()
    for factor in record.factors:
        event_type = get_factor_type(factor)
        event_group = get_factor_group(factor)
        if event_type is None or event_group is None:
            continue
        key = (event_group, event_type)
        if key in seen:
            continue
        seen.add(key)
        metadata = normalized.get(factor)
        polarity = (
            metadata.get("polarity", 0.0)
            if isinstance(metadata, dict)
            else getattr(metadata, "polarity", 0.0)
        )
        confidence = (
            metadata.get("weight", metadata.get("confidence", 0.0))
            if isinstance(metadata, dict)
            else getattr(metadata, "weight", getattr(metadata, "confidence", 0.0))
        )
        observed_at = (
            metadata.get("observed_at")
            if isinstance(metadata, dict)
            else getattr(metadata, "observed_at", None)
        )
        events.append(
            EventRecord(
                event_group=event_group,
                event_type=event_type,
                polarity=float(polarity or 0.0),
                confidence=float(confidence or 0.0),
                observed_at=observed_at,
                description=factor,
            )
        )

    sources = [source for source in record.article_sources if source]
    published = sorted(record.article_published_at)
    temporal_span = (
        (published[-1] - published[0]).total_seconds() / 3600.0
        if len(published) >= 2
        else 0.0
    )
    current_types = {event.event_type for event in events}
    novelty_7d = _novelty(
        current_types,
        history,
        current_date=record.date,
        window_days=7,
    )
    novelty_30d = _novelty(
        current_types,
        history,
        current_date=record.date,
        window_days=30,
    )
    source_count = len(set(source.strip().lower() for source in sources))
    source_diversity = _normalized_entropy(sources)
    breadth = min(1.0, math.log1p(source_count) / math.log(21.0))
    incremental_information = novelty_30d * breadth
    group_counts = Counter(event.event_group for event in events)

    return DailyEventState(
        date=record.date,
        symbol=record.symbol.upper(),
        events=events,
        article_count=max(len(record.article_ids), len(sources)),
        source_count=source_count,
        source_diversity=source_diversity,
        temporal_span_hours=max(0.0, temporal_span),
        novelty_7d=novelty_7d,
        novelty_30d=novelty_30d,
        incremental_information=incremental_information,
        dominant_event_groups=[
            group
            for group, _ in sorted(
                group_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
        ],
    )


def build_event_vector(state: DailyEventState | None) -> np.ndarray:
    vector = np.zeros(EVENT_DIM, dtype=np.float32)
    if state is None:
        return vector

    polarities: list[float] = []
    confidences: list[float] = []
    for event in state.events:
        type_index = TYPE_INDEX.get(event.event_type)
        if type_index is not None:
            vector[type_index] = 1.0
        group_index = GROUP_INDEX.get(event.event_group)
        if group_index is not None:
            vector[NUM_TYPES + group_index] = 1.0
        polarities.append(float(event.polarity))
        confidences.append(float(event.confidence))

    scalar_offset = NUM_TYPES + NUM_GROUPS
    vector[scalar_offset:] = np.asarray(
        [
            float(np.mean(polarities)) if polarities else 0.0,
            max((abs(value) for value in polarities), default=0.0),
            min(1.0, math.log1p(state.article_count) / math.log(51.0)),
            min(1.0, math.log1p(state.source_count) / math.log(21.0)),
            float(np.clip(state.source_diversity, 0.0, 1.0)),
            float(np.clip(state.novelty_7d, 0.0, 1.0)),
            float(np.clip(state.novelty_30d, 0.0, 1.0)),
            float(np.mean(confidences)) if confidences else 0.0,
            min(1.0, state.temporal_span_hours / 168.0),
            float(np.clip(state.incremental_information, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    return vector
