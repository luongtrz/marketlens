from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from stockmem.src.models import SimilarRecord, StockMemRecord

TRAIN_END = date(2024, 12, 24)
VAL_START = date(2025, 1, 1)
VAL_END = date(2025, 6, 23)
TEST_START = date(2025, 7, 1)
TEST_END = date(2026, 5, 1)
DEFAULT_KNN_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)
DEFAULT_KNN_RETURNS_HEAD = {
    "1d": 0.15,
    "3d": 0.25,
    "7d": 0.35,
    "15d": 0.15,
    "30d": 0.10,
}
CLASSES = ("BUY", "HOLD", "SELL")


@dataclass(frozen=True)
class HistoricalRow:
    record: StockMemRecord
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray
    split: str

    @property
    def parsed_date(self) -> date:
        return self.record.date


@dataclass(frozen=True)
class PredictionMetrics:
    name: str
    n: int
    overall_acc: float
    active_acc: float
    coverage: float
    buy_rate: float
    hold_rate: float
    sell_rate: float
    avg_confidence: float
    hit_at_5_same_sign: float
    actual_counts: dict[str, int]
    predicted_counts: dict[str, int]
    confusion: dict[str, dict[str, int]]


@dataclass(frozen=True)
class HeadConfig:
    name: str
    retriever: str
    k: int
    buy_threshold: float
    sell_threshold: float
    return_weights: dict[str, float]


def assign_split(day: date) -> str:
    if day <= TRAIN_END:
        return "train"
    if VAL_START <= day <= VAL_END:
        return "val"
    if TEST_START <= day <= TEST_END:
        return "test"
    return "embargo"


def load_historical_rows(path: Path) -> list[HistoricalRow]:
    rows: list[HistoricalRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            payload = raw.get("payload", raw)
            record = StockMemRecord.model_validate(payload)
            if record.future_return_7d is None:
                continue
            rows.append(
                HistoricalRow(
                    record=record,
                    factor_vec=np.asarray(payload.get("factor_vec", []), dtype=np.float64),
                    indicator_vec=np.asarray(payload.get("indicator_vec", []), dtype=np.float64),
                    price_vec=np.asarray(payload.get("price_vec", []), dtype=np.float64),
                    split=assign_split(record.date),
                )
            )
    rows.sort(key=lambda row: row.record.date)
    return rows


def matured_pool(
    rows: list[HistoricalRow],
    query: HistoricalRow,
    *,
    horizon_days: int = 7,
) -> list[HistoricalRow]:
    cutoff = query.parsed_date
    return [
        candidate
        for candidate in rows
        if candidate.parsed_date < cutoff
        and candidate.parsed_date + timedelta(days=horizon_days) <= cutoff
    ]


def load_knn_weights(path: Path) -> tuple[float, float, float]:
    if not path.exists():
        return DEFAULT_KNN_WEIGHTS
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights", payload)
    try:
        return (
            float(weights["w1_factor"]),
            float(weights["w2_indicator"]),
            float(weights["w3_price"]),
        )
    except (KeyError, TypeError, ValueError):
        return DEFAULT_KNN_WEIGHTS


def load_head_config(path: Path) -> HeadConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    head = payload.get("head", {})
    return HeadConfig(
        name=str(payload.get("name", path.stem)),
        retriever=str(payload.get("retriever", "fixed_knn")),
        k=int(head["k"]),
        buy_threshold=float(head["buy_threshold"]),
        sell_threshold=float(head["sell_threshold"]),
        return_weights={str(k): float(v) for k, v in head["return_weights"].items()},
    )


def actual_signal(value: float | None, threshold: float) -> str:
    actual = float(value or 0.0)
    if actual > threshold:
        return "BUY"
    if actual < -threshold:
        return "SELL"
    return "HOLD"


def retrieve_fixed_knn(
    query: HistoricalRow,
    pool: list[HistoricalRow],
    *,
    weights: tuple[float, float, float],
    k: int,
) -> list[SimilarRecord]:
    w1, w2, w3 = weights
    scored: list[tuple[float, HistoricalRow]] = []
    for candidate in pool:
        score = (
            w1 * float(np.dot(query.factor_vec, candidate.factor_vec))
            + w2 * float(np.dot(query.indicator_vec, candidate.indicator_vec))
            + w3 * float(np.dot(query.price_vec, candidate.price_vec))
        )
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        SimilarRecord(
            record=candidate.record,
            similarity=score,
            retriever_version="fixed_knn_current_pipeline",
        )
        for score, candidate in scored[:k]
    ]


def fixed_knn_signal(
    similar_records: list[SimilarRecord],
    *,
    threshold: float,
) -> tuple[str, float]:
    values = [
        item.record.future_return_7d
        for item in similar_records
        if item.record.future_return_7d is not None
    ]
    if not values:
        return "HOLD", 0.5
    avg = float(np.mean(values))
    if avg > threshold:
        signal = "BUY"
        confidence = min(0.55 + min((avg - threshold) / 15.0, 0.35), 0.95)
    elif avg < -threshold:
        signal = "SELL"
        confidence = min(0.55 + min((abs(avg) - threshold) / 15.0, 0.35), 0.95)
    else:
        signal = "HOLD"
        confidence = 0.5
    return signal, round(confidence, 3)


def knn_returns_signal(
    similar_records: list[SimilarRecord],
    *,
    threshold: float,
    horizon_weights: dict[str, float] | None = None,
) -> tuple[str, float]:
    horizon_weights = horizon_weights or DEFAULT_KNN_RETURNS_HEAD
    per_record_avgs: list[float] = []
    for sr in similar_records:
        rec = sr.record
        total_w = total_v = 0.0
        for horizon, weight in horizon_weights.items():
            value = getattr(rec, f"future_return_{horizon}", None)
            if value is not None:
                total_v += float(value) * weight
                total_w += weight
        if total_w > 0:
            per_record_avgs.append(total_v / total_w)
    if not per_record_avgs:
        return "HOLD", 0.5
    avg = float(np.mean(per_record_avgs))
    if avg > threshold:
        signal = "BUY"
        confidence = min(0.55 + min((avg - threshold) / 15.0, 0.35), 0.95)
    elif avg < -threshold:
        signal = "SELL"
        confidence = min(0.55 + min((abs(avg) - threshold) / 15.0, 0.35), 0.95)
    else:
        signal = "HOLD"
        confidence = 0.5
    return signal, round(confidence, 3)


def configured_head_signal(
    similar_records: list[SimilarRecord],
    *,
    head: HeadConfig,
) -> tuple[str, float]:
    per_record_avgs: list[float] = []
    for sr in similar_records[: head.k]:
        rec = sr.record
        total_w = total_v = 0.0
        for horizon, weight in head.return_weights.items():
            value = getattr(rec, f"future_return_{horizon}", None)
            if value is not None:
                total_v += float(value) * weight
                total_w += weight
        if total_w > 0:
            per_record_avgs.append(total_v / total_w)
    if not per_record_avgs:
        return "HOLD", 0.5
    avg = float(np.mean(per_record_avgs))
    if avg > head.buy_threshold:
        signal = "BUY"
        confidence = min(0.55 + min((avg - head.buy_threshold) / 15.0, 0.35), 0.95)
    elif avg < -head.sell_threshold:
        signal = "SELL"
        confidence = min(0.55 + min((abs(avg) - head.sell_threshold) / 15.0, 0.35), 0.95)
    else:
        signal = "HOLD"
        confidence = 0.5
    return signal, round(confidence, 3)


def summarize_predictions(
    name: str,
    rows: list[dict[str, Any]],
    *,
    label_threshold: float,
) -> PredictionMetrics:
    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    actual_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    total = correct = active_total = active_correct = hit_total = hit_correct = 0
    confidence_sum = 0.0

    for row in rows:
        predicted = str(row.get("predicted_signal", "HOLD"))
        actual = actual_signal(row.get("actual_return_7d"), label_threshold)
        actual_counts[actual] += 1
        predicted_counts[predicted] += 1
        confusion[actual][predicted] += 1
        total += 1
        confidence_sum += float(row.get("confidence") or 0.0)
        correct += int(predicted == actual)
        if predicted in ("BUY", "SELL"):
            active_total += 1
            if predicted == "BUY":
                active_correct += int(float(row.get("actual_return_7d") or 0.0) > 0.0)
            else:
                active_correct += int(float(row.get("actual_return_7d") or 0.0) < 0.0)
        hit = row.get("top5_same_sign")
        if hit is not None:
            hit_total += 1
            hit_correct += int(bool(hit))

    buy_rate = predicted_counts["BUY"] / total if total else 0.0
    hold_rate = predicted_counts["HOLD"] / total if total else 0.0
    sell_rate = predicted_counts["SELL"] / total if total else 0.0
    return PredictionMetrics(
        name=name,
        n=total,
        overall_acc=(correct / total) if total else 0.0,
        active_acc=(active_correct / active_total) if active_total else 0.0,
        coverage=(active_total / total) if total else 0.0,
        buy_rate=buy_rate,
        hold_rate=hold_rate,
        sell_rate=sell_rate,
        avg_confidence=(confidence_sum / total) if total else 0.0,
        hit_at_5_same_sign=(hit_correct / hit_total) if hit_total else 0.0,
        actual_counts=dict(actual_counts),
        predicted_counts=dict(predicted_counts),
        confusion=confusion,
    )
