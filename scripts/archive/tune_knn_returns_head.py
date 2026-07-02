from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
import math
from pathlib import Path
import random
import time

TRAIN_END = date(2023, 12, 31)
VAL_START = date(2024, 1, 1)
VAL_END = date(2024, 12, 31)
TEST_START = date(2025, 1, 1)
CLASSES = ("BUY", "HOLD", "SELL")
DEFAULT_SEARCH_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)
DEFAULT_RETURN_WEIGHTS = {"1d": 0.40, "3d": 0.30, "7d": 0.15, "15d": 0.10, "30d": 0.05}


@dataclass(frozen=True)
class Record:
    day: date
    factor_vec: list[float]
    indicator_vec: list[float]
    price_vec: list[float]
    event_vec: list[float]
    future: dict[str, float | None]

    @property
    def blocks(self) -> tuple[list[float], ...]:
        return (self.event_vec, self.factor_vec, self.indicator_vec, self.price_vec)


@dataclass(frozen=True)
class HeadConfig:
    k: int
    buy_thr: float
    sell_thr: float
    return_weights: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {
            "k": self.k,
            "buy_threshold": self.buy_thr,
            "sell_threshold": self.sell_thr,
            "return_weights": self.return_weights,
        }


@dataclass(frozen=True)
class LearnedMetric:
    block_dims: tuple[int, ...]
    diagonal: list[float]
    block_scales: list[float]

    @classmethod
    def load(cls, path: str | Path) -> "LearnedMetric":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            block_dims=tuple(int(v) for v in payload["block_dims"]),
            diagonal=[float(v) for v in payload["d"]],
            block_scales=[float(v) for v in payload["block_scales"]],
        )

    def score(self, query_blocks: tuple[list[float], ...], candidate_blocks: tuple[list[float], ...]) -> float:
        total = 0.0
        offset = 0
        for dim, scale, query, candidate in zip(
            self.block_dims, self.block_scales, query_blocks, candidate_blocks
        ):
            diag = self.diagonal[offset : offset + dim]
            q_weighted = _normalize([diag[i] * query[i] for i in range(dim)])
            c_weighted = _normalize([diag[i] * candidate[i] for i in range(dim)])
            total += scale * _dot(q_weighted, c_weighted)
            offset += dim
        return total


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return [0.0] * len(values)
    return [v / norm for v in values]


def _load_dataset(path: Path) -> list[Record]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[Record] = []
    for item in payload:
        factor_raw = item.get("factor_vec") or []
        indicator_raw = item.get("indicator_vec") or []
        price_raw = item.get("price_vec") or []
        event_raw = item.get("event_vec") or []
        if len(factor_raw) != 75 or len(indicator_raw) != 5 or len(price_raw) != 60:
            continue
        if len(event_raw) != 85:
            event_raw = [0.0] * 85
        records.append(
            Record(
                day=date.fromisoformat(str(item["date"])),
                factor_vec=_normalize([float(v) for v in factor_raw]),
                indicator_vec=_normalize([float(v) for v in indicator_raw]),
                price_vec=_normalize([float(v) for v in price_raw]),
                event_vec=_normalize([float(v) for v in event_raw]),
                future={
                    "1d": item.get("future_return_1d"),
                    "3d": item.get("future_return_3d"),
                    "7d": item.get("future_return_7d"),
                    "15d": item.get("future_return_15d"),
                    "30d": item.get("future_return_30d"),
                },
            )
        )
    records.sort(key=lambda row: row.day)
    return records


def _split_name(day: date) -> str:
    if day <= TRAIN_END:
        return "train"
    if VAL_START <= day <= VAL_END:
        return "val"
    if day >= TEST_START:
        return "test"
    return "embargo"


def _fixed_score(q: Record, c: Record, weights: tuple[float, float, float]) -> float:
    return (
        weights[0] * _dot(q.factor_vec, c.factor_vec)
        + weights[1] * _dot(q.indicator_vec, c.indicator_vec)
        + weights[2] * _dot(q.price_vec, c.price_vec)
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


def _matured_pool(records: list[Record], index: int, query_day: date) -> list[Record]:
    min_query_day = query_day
    return [
        candidate
        for candidate in records[:index]
        if candidate.day + timedelta(days=7) <= min_query_day
    ]


def _evaluate(
    records: list[Record],
    score_fn,
    split: str,
    config: HeadConfig,
) -> dict[str, object]:
    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    buy = sell = hold = 0
    buy_correct = sell_correct = hold_correct = 0
    n = 0

    for idx, query in enumerate(records):
        if _split_name(query.day) != split:
            continue
        if query.future.get("7d") is None:
            continue
        pool = _matured_pool(records, idx, query.day)
        if len(pool) < config.k:
            continue

        scored = sorted(
            ((score_fn(query, candidate), candidate) for candidate in pool),
            key=lambda item: item[0],
            reverse=True,
        )[: config.k]
        neighbor_avgs = [
            avg for _, row in scored if (avg := _weighted_avg(row, config.return_weights)) is not None
        ]
        if not neighbor_avgs:
            continue

        predicted_avg = sum(neighbor_avgs) / len(neighbor_avgs)
        pred = _signal(predicted_avg, config.buy_thr, config.sell_thr)
        actual_ret = float(query.future["7d"])
        actual = _signal(actual_ret, config.buy_thr, config.sell_thr)
        confusion[actual][pred] += 1
        n += 1

        if pred == "BUY":
            buy += 1
            buy_correct += int(actual_ret > 0.0)
        elif pred == "SELL":
            sell += 1
            sell_correct += int(actual_ret < 0.0)
        else:
            hold += 1
            hold_correct += int(-config.sell_thr <= actual_ret <= config.buy_thr)

    total_correct = buy_correct + sell_correct + hold_correct
    active = buy + sell
    return {
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
        "confusion": confusion,
    }


def _selection_score(metrics: dict[str, object]) -> float:
    buy = float(metrics["buy_da_pct"])
    sell = float(metrics["sell_da_pct"])
    overall = float(metrics["overall_da_pct"])
    active = float(metrics["active_acc_pct"])
    coverage = float(metrics["coverage_pct"])
    # Favor balanced BUY/SELL action quality first, then overall/action stability, then coverage.
    return 0.35 * buy + 0.35 * sell + 0.15 * active + 0.10 * overall + 0.05 * coverage


def _sample_head_config(rng: random.Random, k_values: list[int]) -> HeadConfig:
    weights = _dirichlet5(rng)
    return HeadConfig(
        k=int(rng.choice(k_values)),
        buy_thr=round(1.0 + 3.0 * rng.random(), 2),
        sell_thr=round(1.0 + 3.0 * rng.random(), 2),
        return_weights={
            "1d": round(weights[0], 4),
            "3d": round(weights[1], 4),
            "7d": round(weights[2], 4),
            "15d": round(weights[3], 4),
            "30d": round(weights[4], 4),
        },
    )


def _dirichlet5(rng: random.Random) -> list[float]:
    draws = [-math.log(max(rng.random(), 1e-12)) for _ in range(5)]
    total = sum(draws)
    return [value / total for value in draws]


def _dedupe_configs(configs: list[HeadConfig]) -> list[HeadConfig]:
    deduped: list[HeadConfig] = []
    seen: set[str] = set()
    for config in configs:
        key = json.dumps(config.to_json(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(config)
    return deduped


def _print_metrics(label: str, metrics: dict[str, object]) -> None:
    print(
        f"{label}: "
        f"n={metrics['n']} coverage={float(metrics['coverage_pct']):.1f}% "
        f"BUY_DA={float(metrics['buy_da_pct']):.1f}% "
        f"SELL_DA={float(metrics['sell_da_pct']):.1f}% "
        f"HOLD_DA={float(metrics['hold_da_pct']):.1f}% "
        f"overall_DA={float(metrics['overall_da_pct']):.1f}% "
        f"active_acc={float(metrics['active_acc_pct']):.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the knn_returns decision head on the full 2018+ dataset using a fixed retriever."
    )
    parser.add_argument("--data", default="stockmem/data/real_optimizer_finbert.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", default="3,5,7,10,15,20,25,30")
    parser.add_argument(
        "--output",
        default="artifacts/train_logs/knn_head_tuning_finbert_full.json",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    records = _load_dataset(Path(args.data))
    metric = LearnedMetric.load(args.artifact)
    rng = random.Random(args.seed)
    k_values = [int(part.strip()) for part in args.k_values.split(",") if part.strip()]

    baseline_search_weights = DEFAULT_SEARCH_WEIGHTS
    weights_path = Path(args.weights)
    if weights_path.exists():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        source = payload.get("weights", payload)
        baseline_search_weights = (
            float(source["w1_factor"]),
            float(source["w2_indicator"]),
            float(source["w3_price"]),
        )

    fixed_score = lambda q, c: _fixed_score(q, c, baseline_search_weights)
    learned_score = lambda q, c: metric.score(q.blocks, c.blocks)

    default_config = HeadConfig(
        k=5,
        buy_thr=2.0,
        sell_thr=2.0,
        return_weights=DEFAULT_RETURN_WEIGHTS,
    )

    candidate_configs = [default_config]
    # Add a few structured candidates before random search.
    for k in k_values:
        candidate_configs.append(
            HeadConfig(
                k=k,
                buy_thr=2.0,
                sell_thr=2.0,
                return_weights=DEFAULT_RETURN_WEIGHTS,
            )
        )
        candidate_configs.append(
            HeadConfig(
                k=k,
                buy_thr=2.5,
                sell_thr=2.5,
                return_weights=DEFAULT_RETURN_WEIGHTS,
            )
        )
        candidate_configs.append(
            HeadConfig(
                k=k,
                buy_thr=3.0,
                sell_thr=3.0,
                return_weights=DEFAULT_RETURN_WEIGHTS,
            )
        )

    for _ in range(args.trials):
        candidate_configs.append(_sample_head_config(rng, k_values))
    candidate_configs = _dedupe_configs(candidate_configs)

    best_config = default_config
    best_val = _evaluate(records, learned_score, "val", best_config)
    best_score = _selection_score(best_val)
    trials_summary: list[dict[str, object]] = []

    started_at = time.time()
    for index, config in enumerate(candidate_configs, start=1):
        val_metrics = _evaluate(records, learned_score, "val", config)
        score = _selection_score(val_metrics)
        trials_summary.append(
            {
                "rank_candidate": index,
                "selection_score": score,
                "config": config.to_json(),
                "val_metrics": val_metrics,
            }
        )
        if score > best_score:
            best_config = config
            best_val = val_metrics
            best_score = score
        if args.progress_every > 0 and (
            index == 1
            or index % args.progress_every == 0
            or index == len(candidate_configs)
        ):
            elapsed = time.time() - started_at
            print(
                f"[progress] {index}/{len(candidate_configs)} "
                f"elapsed={elapsed:.1f}s best_score={best_score:.3f}"
            , flush=True)

    default_learned_val = _evaluate(records, learned_score, "val", default_config)
    default_learned_test = _evaluate(records, learned_score, "test", default_config)
    tuned_learned_test = _evaluate(records, learned_score, "test", best_config)
    default_fixed_val = _evaluate(records, fixed_score, "val", default_config)
    default_fixed_test = _evaluate(records, fixed_score, "test", default_config)
    tuned_fixed_test = _evaluate(records, fixed_score, "test", best_config)

    report = {
        "data": args.data,
        "artifact": args.artifact,
        "weights": args.weights,
        "split": {
            "train_end": TRAIN_END.isoformat(),
            "val_start": VAL_START.isoformat(),
            "val_end": VAL_END.isoformat(),
            "test_start": TEST_START.isoformat(),
        },
        "search_space": {
            "k_values": k_values,
            "trials": args.trials,
            "seed": args.seed,
        },
        "baseline_search_weights": {
            "w_factor": baseline_search_weights[0],
            "w_indicator": baseline_search_weights[1],
            "w_price": baseline_search_weights[2],
        },
        "default_config": default_config.to_json(),
        "best_config": best_config.to_json(),
        "best_val_selection_score": best_score,
        "learned_default": {
            "val": default_learned_val,
            "test": default_learned_test,
        },
        "learned_tuned": {
            "val": best_val,
            "test": tuned_learned_test,
        },
        "fixed_default": {
            "val": default_fixed_val,
            "test": default_fixed_test,
        },
        "fixed_with_learned_tuned_head": {
            "test": tuned_fixed_test,
        },
        "trial_count": len(candidate_configs),
        "top_val_trials": sorted(
            trials_summary,
            key=lambda item: float(item["selection_score"]),
            reverse=True,
        )[:20],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "trial_count": len(candidate_configs)}, indent=2), flush=True)
    print("\n[learned default]", flush=True)
    _print_metrics("val", default_learned_val)
    _print_metrics("test", default_learned_test)
    print("\n[learned tuned]", flush=True)
    print(json.dumps(best_config.to_json(), indent=2), flush=True)
    _print_metrics("val", best_val)
    _print_metrics("test", tuned_learned_test)
    print("\n[fixed baseline]", flush=True)
    _print_metrics("val", default_fixed_val)
    _print_metrics("test", default_fixed_test)
    print("\n[fixed baseline with learned-tuned head]", flush=True)
    _print_metrics("test", tuned_fixed_test)


if __name__ == "__main__":
    main()
