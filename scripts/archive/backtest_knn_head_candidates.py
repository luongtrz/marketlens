from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stockmem.src.search.learned_metric import LearnedDiagonalMetric


SEARCH_WEIGHTS_DEFAULT = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)
PRODUCTION_RETURN_WEIGHTS = {"1d": 0.15, "3d": 0.25, "7d": 0.35, "15d": 0.15, "30d": 0.10}
CLASSES = ("BUY", "HOLD", "SELL")
HORIZONS = ("1d", "3d", "7d", "15d", "30d")


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


@dataclass(frozen=True)
class HeadConfig:
    name: str
    retriever: str
    k: int
    buy_thr: float
    sell_thr: float
    return_weights: dict[str, float]


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
            day = str(raw.get("record_date") or payload.get("date"))
            if day < start_date:
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
                    date=day,
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


def _load_search_weights(path: Path) -> tuple[float, float, float]:
    if not path.exists():
        return SEARCH_WEIGHTS_DEFAULT
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("weights", payload)
    return (
        float(source["w1_factor"]),
        float(source["w2_indicator"]),
        float(source["w3_price"]),
    )


def _load_head_config(path: Path) -> HeadConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    head = payload.get("head", {})
    return HeadConfig(
        name=str(payload.get("name", path.stem)),
        retriever=str(payload.get("retriever", "fixed_knn")),
        k=int(head["k"]),
        buy_thr=float(head["buy_threshold"]),
        sell_thr=float(head["sell_threshold"]),
        return_weights={k: float(v) for k, v in head["return_weights"].items()},
    )


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
        return "BUY"
    if avg < -sell_thr:
        return "SELL"
    return "HOLD"


def _simple_sharpe(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std()
    return float(arr.mean() / std) if std > 1e-12 else 0.0


def _simple_sortino(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    downside = arr[arr < 0]
    if downside.size == 0:
        return float("inf")
    dd = downside.std()
    return float(arr.mean() / dd) if dd > 1e-12 else 0.0


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(values, dtype=np.float64) / 100.0)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(running_max, 1e-12) - 1.0
    return float(drawdowns.min()) * 100.0


def _evaluate_candidate(
    name: str,
    records: list[Record],
    score_fn,
    config: HeadConfig,
) -> dict[str, object]:
    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    rows: list[dict[str, object]] = []
    buy = sell = hold = buy_correct = sell_correct = hold_correct = 0
    realized_returns: list[float] = []

    for idx, query in enumerate(records):
        if query.future.get("7d") is None:
            continue
        pool = records[:idx]
        if len(pool) < config.k:
            continue
        scored = sorted(
            ((score_fn(query, candidate), candidate) for candidate in pool),
            key=lambda item: item[0],
            reverse=True,
        )[: config.k]
        avgs = [avg for _, row in scored if (avg := _weighted_avg(row, config.return_weights)) is not None]
        if not avgs:
            continue
        weighted_avg = float(np.mean(avgs))
        pred = _signal(weighted_avg, config.buy_thr, config.sell_thr)
        actual_ret = float(query.future["7d"])
        actual = _signal(actual_ret, config.buy_thr, config.sell_thr)
        confusion[actual][pred] += 1

        if pred == "BUY":
            buy += 1
            buy_correct += int(actual_ret > 0.0)
            realized = actual_ret
        elif pred == "SELL":
            sell += 1
            sell_correct += int(actual_ret < 0.0)
            realized = -actual_ret
        else:
            hold += 1
            hold_correct += int(-config.sell_thr <= actual_ret <= config.buy_thr)
            realized = 0.0
        realized_returns.append(realized)

        rows.append(
            {
                "date": query.date,
                "signal": pred,
                "weighted_avg_return": round(weighted_avg, 4),
                "actual_return_7d": round(actual_ret, 4),
                "realized_strategy_return_7d": round(realized, 4),
                **{f"actual_return_{h}": query.future.get(h) for h in HORIZONS},
            }
        )

    n = len(rows)
    active = buy + sell
    total_correct = buy_correct + sell_correct + hold_correct
    result = {
        "name": name,
        "config": {
            "retriever": config.retriever,
            "k": config.k,
            "buy_threshold": config.buy_thr,
            "sell_threshold": config.sell_thr,
            "return_weights": config.return_weights,
        },
        "summary": {
            "n": n,
            "buy": buy,
            "sell": sell,
            "hold": hold,
            "coverage_pct": (100.0 * active / n) if n else 0.0,
            "buy_da_pct": (100.0 * buy_correct / buy) if buy else 0.0,
            "sell_da_pct": (100.0 * sell_correct / sell) if sell else 0.0,
            "hold_da_pct": (100.0 * hold_correct / hold) if hold else 0.0,
            "overall_da_pct": (100.0 * total_correct / n) if n else 0.0,
            "active_acc_pct": (100.0 * (buy_correct + sell_correct) / active) if active else 0.0,
            "avg_strategy_return_7d_pct": float(np.mean(realized_returns)) if realized_returns else 0.0,
            "sharpe_like": _simple_sharpe(realized_returns),
            "sortino_like": _simple_sortino(realized_returns),
            "max_drawdown_pct": _max_drawdown(realized_returns),
        },
        "confusion": confusion,
        "rows": rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline backtest of learned stable vs fixed stable vs production knn_returns baseline."
    )
    parser.add_argument("--input-ndjson", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--learned-head", default="stockmem/config/knn_head.learned_finbert_rolling_stable.json")
    parser.add_argument("--fixed-head", default="stockmem/config/knn_head.fixed_knn_rolling_stable.json")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--output", default="artifacts/backtests/knn_head_candidates.json")
    args = parser.parse_args()

    records = _load_ndjson(Path(args.input_ndjson), start_date=args.start_date)
    learned_metric = LearnedDiagonalMetric.load(args.artifact)
    search_weights = _load_search_weights(Path(args.weights))

    learned_candidate = _load_head_config(Path(args.learned_head))
    fixed_candidate = _load_head_config(Path(args.fixed_head))
    production = HeadConfig(
        name="production_knn_returns_baseline",
        retriever="fixed_knn",
        k=5,
        buy_thr=2.0,
        sell_thr=2.0,
        return_weights=PRODUCTION_RETURN_WEIGHTS,
    )

    fixed_score = lambda q, c: _fixed_score(q, c, search_weights)
    learned_score = lambda q, c: learned_metric.score(q.blocks, c.blocks)

    payload = {
        "start_date": args.start_date,
        "record_count": len(records),
        "search_weights": {
            "w_factor": search_weights[0],
            "w_indicator": search_weights[1],
            "w_price": search_weights[2],
        },
        "results": [
            _evaluate_candidate("learned_finbert_rolling_stable", records, learned_score, learned_candidate),
            _evaluate_candidate("fixed_knn_rolling_stable", records, fixed_score, fixed_candidate),
            _evaluate_candidate("production_knn_returns_baseline", records, fixed_score, production),
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "record_count": len(records)}, indent=2))
    for result in payload["results"]:
        summary = result["summary"]
        print(
            f"{result['name']}: "
            f"n={summary['n']} coverage={summary['coverage_pct']:.1f}% "
            f"BUY_DA={summary['buy_da_pct']:.1f}% SELL_DA={summary['sell_da_pct']:.1f}% "
            f"HOLD_DA={summary['hold_da_pct']:.1f}% overall_DA={summary['overall_da_pct']:.1f}% "
            f"active_acc={summary['active_acc_pct']:.1f}% avg_ret7d={summary['avg_strategy_return_7d_pct']:.2f}% "
            f"sharpe={summary['sharpe_like']:.3f} mdd={summary['max_drawdown_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
