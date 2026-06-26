from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from stockmem.src.search.learned_metric import LearnedDiagonalMetric


SEARCH_WEIGHTS_DEFAULT = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)
RETURN_WEIGHTS_DEFAULT = {"1d": 0.40, "3d": 0.30, "7d": 0.15, "15d": 0.10, "30d": 0.05}
CLASSES = ("UP", "HOLD", "DOWN")


@dataclass(frozen=True)
class Record:
    date: str
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray
    event_vec: np.ndarray
    future: dict[str, float | None]

    @property
    def blocks(self) -> tuple[np.ndarray, ...]:
        return (self.event_vec, self.factor_vec, self.indicator_vec, self.price_vec)


def _l2(arr: np.ndarray) -> np.ndarray:
    arr64 = np.asarray(arr, dtype=np.float64)
    norm = float(np.linalg.norm(arr64))
    if norm <= 1e-12:
        return np.zeros_like(arr64, dtype=np.float32)
    return (arr64 / norm).astype(np.float32)


def _load_ndjson(path: Path, *, start_date: str) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            payload = raw["payload"]
            date = str(raw.get("record_date") or payload.get("date"))
            if date < start_date:
                continue
            factor_raw = payload.get("factor_vec") or payload.get("factor_vector") or []
            indicator_raw = payload.get("indicator_vec") or []
            price_raw = payload.get("price_vec") or []
            event_raw = payload.get("event_vec") or []
            if len(factor_raw) != 75 or len(indicator_raw) != 5 or len(price_raw) != 60:
                continue
            if len(event_raw) != 85:
                event_raw = [0.0] * 85
            records.append(
                Record(
                    date=date,
                    factor_vec=_l2(np.array(factor_raw, dtype=np.float32)),
                    indicator_vec=_l2(np.array(indicator_raw, dtype=np.float32)),
                    price_vec=_l2(np.array(price_raw, dtype=np.float32)),
                    event_vec=_l2(np.array(event_raw, dtype=np.float32)),
                    future={
                        "1d": payload.get("future_return_1d"),
                        "3d": payload.get("future_return_3d"),
                        "7d": payload.get("future_return_7d"),
                        "15d": payload.get("future_return_15d"),
                        "30d": payload.get("future_return_30d"),
                    },
                )
            )
    return records


def _fixed_score(q: Record, c: Record, weights: tuple[float, float, float]) -> float:
    return (
        weights[0] * float(np.dot(q.factor_vec, c.factor_vec))
        + weights[1] * float(np.dot(q.indicator_vec, c.indicator_vec))
        + weights[2] * float(np.dot(q.price_vec, c.price_vec))
    )


def _weighted_avg(record: Record, weights: dict[str, float]) -> float | None:
    total_w = 0.0
    total_v = 0.0
    for horizon, weight in weights.items():
        value = record.future.get(horizon)
        if value is None:
            continue
        total_v += float(value) * weight
        total_w += weight
    if total_w <= 1e-12:
        return None
    return total_v / total_w


def _signal(avg: float, buy_thr: float, sell_thr: float) -> str:
    if avg > buy_thr:
        return "UP"
    if avg < -sell_thr:
        return "DOWN"
    return "HOLD"


def _evaluate(
    name: str,
    records: list[Record],
    score_fn: Callable[[Record, Record], float],
    *,
    k: int,
    buy_thr: float,
    sell_thr: float,
    return_weights: dict[str, float],
) -> dict:
    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    buy = sell = hold = 0
    buy_correct = sell_correct = hold_correct = 0
    predictions = 0

    for idx, query in enumerate(records):
        if query.future.get("7d") is None:
            continue
        pool = records[:idx]
        if len(pool) < k:
            continue
        scored = sorted(
            ((score_fn(query, candidate), candidate) for candidate in pool),
            key=lambda item: item[0],
            reverse=True,
        )[:k]
        neighbor_avgs = [avg for _, row in scored if (avg := _weighted_avg(row, return_weights)) is not None]
        if not neighbor_avgs:
            continue
        overall_avg = float(np.mean(neighbor_avgs))
        pred = _signal(overall_avg, buy_thr, sell_thr)
        actual_ret = float(query.future["7d"])
        actual = _signal(actual_ret, buy_thr, sell_thr)
        confusion[actual][pred] += 1
        predictions += 1
        if pred == "UP":
            buy += 1
            buy_correct += int(actual_ret > 0)
        elif pred == "DOWN":
            sell += 1
            sell_correct += int(actual_ret < 0)
        else:
            hold += 1
            hold_correct += int(-buy_thr <= actual_ret <= buy_thr)

    total_correct = buy_correct + sell_correct + hold_correct
    active = buy + sell
    result = {
        "name": name,
        "n": predictions,
        "buy": buy,
        "sell": sell,
        "hold": hold,
        "coverage_pct": (100.0 * active / predictions) if predictions else 0.0,
        "da_buy": (100.0 * buy_correct / buy) if buy else 0.0,
        "da_sell": (100.0 * sell_correct / sell) if sell else 0.0,
        "da_hold": (100.0 * hold_correct / hold) if hold else 0.0,
        "da_all": (100.0 * total_correct / predictions) if predictions else 0.0,
        "active_accuracy": (100.0 * (buy_correct + sell_correct) / active) if active else 0.0,
        "confusion": confusion,
        "predicted_counts": {"UP": buy, "HOLD": hold, "DOWN": sell},
        "actual_counts": {
            cls: int(sum(confusion[cls].values()))
            for cls in CLASSES
        },
    }
    return result


def _print_result(result: dict) -> None:
    print(f"\n[{result['name']}]")
    print(
        f"n={result['n']} coverage={result['coverage_pct']:.1f}% "
        f"BUY_DA={result['da_buy']:.1f}% SELL_DA={result['da_sell']:.1f}% "
        f"HOLD_DA={result['da_hold']:.1f}% overall_DA={result['da_all']:.1f}% "
        f"active_acc={result['active_accuracy']:.1f}%"
    )
    print("actual_counts:", json.dumps(result["actual_counts"], sort_keys=True))
    print("predicted_counts:", json.dumps(result["predicted_counts"], sort_keys=True))
    print("actual\\pred\tUP\tHOLD\tDOWN")
    for actual in CLASSES:
        row = result["confusion"][actual]
        print(f"{actual}\t{row['UP']}\t{row['HOLD']}\t{row['DOWN']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-kNN and learned retriever under the docs knn_returns decision head."
    )
    parser.add_argument("--input-ndjson", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--buy-thr", type=float, default=2.0)
    parser.add_argument("--sell-thr", type=float, default=2.0)
    args = parser.parse_args()

    search_weights = SEARCH_WEIGHTS_DEFAULT
    weights_path = Path(args.weights)
    if weights_path.exists():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        source = payload.get("weights", payload)
        search_weights = (
            float(source["w1_factor"]),
            float(source["w2_indicator"]),
            float(source["w3_price"]),
        )

    records = _load_ndjson(Path(args.input_ndjson), start_date=args.start_date)
    metric = LearnedDiagonalMetric.load(args.artifact)

    baseline = _evaluate(
        "baseline_knn_returns_head",
        records,
        lambda q, c: _fixed_score(q, c, search_weights),
        k=args.k,
        buy_thr=args.buy_thr,
        sell_thr=args.sell_thr,
        return_weights=RETURN_WEIGHTS_DEFAULT,
    )
    learned = _evaluate(
        "learned_retriever_knn_returns_head",
        records,
        lambda q, c: metric.score(q.blocks, c.blocks),
        k=args.k,
        buy_thr=args.buy_thr,
        sell_thr=args.sell_thr,
        return_weights=RETURN_WEIGHTS_DEFAULT,
    )

    print(
        json.dumps(
            {
                "start_date": args.start_date,
                "input_records": len(records),
                "k": args.k,
                "buy_threshold_pct": args.buy_thr,
                "sell_threshold_pct": args.sell_thr,
                "search_weights": {
                    "w_factor": search_weights[0],
                    "w_indicator": search_weights[1],
                    "w_price": search_weights[2],
                },
                "return_weights": RETURN_WEIGHTS_DEFAULT,
            },
            indent=2,
        )
    )
    _print_result(baseline)
    _print_result(learned)


if __name__ == "__main__":
    main()
