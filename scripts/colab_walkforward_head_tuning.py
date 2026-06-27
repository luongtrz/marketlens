from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import json
import math
from pathlib import Path
import random
import time

import numpy as np


CLASSES = ("BUY", "HOLD", "SELL")
DEFAULT_SEARCH_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)
DEFAULT_RETURN_WEIGHTS = {"1d": 0.40, "3d": 0.30, "7d": 0.15, "15d": 0.10, "30d": 0.05}
K_CHOICES = (3, 5, 7, 10, 15, 20, 25, 30)
RETURN_KEYS = ("1d", "3d", "7d", "15d", "30d")
DEFAULT_OUTPUT = "walkforward_head_tuning_results.json"


@dataclass(frozen=True)
class WindowSpec:
    train_months: int
    val_months: int
    test_months: int
    step_months: int
    expanding: bool


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class HeadConfig:
    k: int
    buy_thr: float
    sell_thr: float
    return_weights: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {
            "k": self.k,
            "buy_thr": self.buy_thr,
            "sell_thr": self.sell_thr,
            "return_weights": self.return_weights,
        }


@dataclass(frozen=True)
class SearchSpace:
    k_choices: tuple[int, ...]
    buy_range: tuple[float, float]
    sell_range: tuple[float, float]
    threshold_grid: tuple[float, ...]
    prior_alpha: tuple[float, float, float, float, float]


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _add_months(day: date, months: int) -> date:
    month_index = (day.year * 12 + (day.month - 1)) + months
    year = month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


def _month_end(day: date) -> date:
    return _add_months(_month_start(day), 1) - timedelta(days=1)


def _build_folds(days: list[date], spec: WindowSpec) -> list[Fold]:
    min_day = min(days)
    max_day = max(days)
    origin = _month_start(min_day)
    first_train_end = _month_end(_add_months(origin, spec.train_months) - timedelta(days=1))
    val_start = _month_start(_add_months(origin, spec.train_months))
    folds: list[Fold] = []
    index = 1

    while True:
        val_end = _month_end(_add_months(val_start, spec.val_months) - timedelta(days=1))
        test_start = _month_start(_add_months(val_start, spec.val_months))
        test_end = _month_end(_add_months(test_start, spec.test_months) - timedelta(days=1))
        if test_start > max_day:
            break
        if spec.expanding:
            train_start = origin
        else:
            train_start = _month_start(_add_months(val_start, -spec.train_months))
        train_end = val_start - timedelta(days=1)
        if train_start < origin:
            train_start = origin
        if train_end < train_start:
            break
        folds.append(
            Fold(
                index=index,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=min(val_end, max_day),
                test_start=test_start,
                test_end=min(test_end, max_day),
            )
        )
        index += 1
        val_start = _month_start(_add_months(val_start, spec.step_months))
        if val_start > max_day:
            break
    return folds


def _normalize_rows(rows: list[dict], key: str, dim: int) -> np.ndarray:
    mat = np.zeros((len(rows), dim), dtype=np.float32)
    for i, row in enumerate(rows):
        values = row.get(key) or []
        if len(values) != dim:
            continue
        arr = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(arr)
        mat[i] = arr if norm <= 1e-12 else arr / norm
    return mat


def _load_dataset(path: Path) -> tuple[list[dict], list[date], np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.sort(key=lambda item: item["date"])
    days = [date.fromisoformat(str(row["date"])) for row in rows]
    futures = np.full((len(rows), 5), np.nan, dtype=np.float32)
    future_keys = (
        "future_return_1d",
        "future_return_3d",
        "future_return_7d",
        "future_return_15d",
        "future_return_30d",
    )
    for i, row in enumerate(rows):
        for j, key in enumerate(future_keys):
            value = row.get(key)
            if value is not None:
                futures[i, j] = float(value)
    return rows, days, futures


def _load_search_weights(path: Path | None) -> tuple[float, float, float]:
    if path is None or not path.exists():
        return DEFAULT_SEARCH_WEIGHTS
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("weights", payload)
    return (
        float(source["w1_factor"]),
        float(source["w2_indicator"]),
        float(source["w3_price"]),
    )


def _load_learned_metric(path: Path) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    block_dims = tuple(int(v) for v in payload["block_dims"])
    diagonal = np.asarray(payload["d"], dtype=np.float64)
    block_scales = np.asarray(payload["block_scales"], dtype=np.float64)
    return block_dims, diagonal, block_scales


def _precompute_learned_blocks(
    event_mat: np.ndarray,
    factor_mat: np.ndarray,
    indicator_mat: np.ndarray,
    price_mat: np.ndarray,
    block_dims: tuple[int, ...],
    diagonal: np.ndarray,
) -> list[np.ndarray]:
    base_blocks = [event_mat, factor_mat, indicator_mat, price_mat]
    blocks: list[np.ndarray] = []
    offset = 0
    for i, dim in enumerate(block_dims):
        block = base_blocks[i].astype(np.float64)
        diag_block = diagonal[offset : offset + dim]
        weighted = block * diag_block[None, :]
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        normalized = np.where(norms > 1e-12, weighted / np.maximum(norms, 1e-12), 0.0)
        blocks.append(normalized.astype(np.float32))
        offset += dim
    return blocks


def _matured_pool(days: list[date], query_index: int, horizon_days: int = 7) -> np.ndarray:
    query_day = days[query_index]
    pool = [
        idx
        for idx in range(query_index)
        if days[idx] + timedelta(days=horizon_days) <= query_day
    ]
    return np.asarray(pool, dtype=np.int32)


def _precompute_neighbors(
    days: list[date],
    futures: np.ndarray,
    factor_mat: np.ndarray,
    indicator_mat: np.ndarray,
    price_mat: np.ndarray,
    learned_blocks: list[np.ndarray],
    block_scales: np.ndarray,
    fixed_weights: tuple[float, float, float],
    kmax: int,
    progress_every: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    fixed_top: dict[int, np.ndarray] = {}
    learned_top: dict[int, np.ndarray] = {}
    query_indices = [idx for idx in range(len(days)) if not np.isnan(futures[idx, 2])]
    started = time.time()

    for count, query_index in enumerate(query_indices, start=1):
        pool = _matured_pool(days, query_index, horizon_days=7)
        if pool.size < 3:
            continue
        fixed_scores = (
            fixed_weights[0] * (factor_mat[pool] @ factor_mat[query_index])
            + fixed_weights[1] * (indicator_mat[pool] @ indicator_mat[query_index])
            + fixed_weights[2] * (price_mat[pool] @ price_mat[query_index])
        )
        learned_scores = np.zeros(pool.shape[0], dtype=np.float32)
        for scale, block in zip(block_scales, learned_blocks):
            if scale <= 1e-12:
                continue
            learned_scores += float(scale) * (block[pool] @ block[query_index])
        fixed_top[query_index] = pool[np.argsort(fixed_scores)[::-1][:kmax]]
        learned_top[query_index] = pool[np.argsort(learned_scores)[::-1][:kmax]]
        if progress_every > 0 and (
            count == 1
            or count % progress_every == 0
            or count == len(query_indices)
        ):
            elapsed = time.time() - started
            print(
                f"[precompute] {count}/{len(query_indices)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return fixed_top, learned_top


def _per_record_avgs(futures: np.ndarray, neighbor_idx: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    weight_arr = np.asarray([weights[key] for key in RETURN_KEYS], dtype=np.float32)
    values = futures[neighbor_idx]
    valid = ~np.isnan(values)
    denom = np.where(valid, weight_arr[None, :], 0.0).sum(axis=1)
    numer = np.where(valid, values * weight_arr[None, :], 0.0).sum(axis=1)
    avgs = numer / np.maximum(denom, 1e-12)
    return avgs[denom > 1e-12]


def _signal(avg: float, buy_thr: float, sell_thr: float) -> str:
    if avg > buy_thr:
        return "BUY"
    if avg < -sell_thr:
        return "SELL"
    return "HOLD"


def _evaluate_indices(
    query_indices: list[int],
    top_map: dict[int, np.ndarray],
    futures: np.ndarray,
    config: HeadConfig,
) -> dict[str, object]:
    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    buy = sell = hold = 0
    buy_correct = sell_correct = hold_correct = 0
    n = 0

    for query_index in query_indices:
        neighbors = top_map.get(query_index)
        if neighbors is None or neighbors.size < config.k:
            continue
        avgs = _per_record_avgs(futures, neighbors[: config.k], config.return_weights)
        if avgs.size == 0:
            continue
        predicted_avg = float(avgs.mean())
        predicted = _signal(predicted_avg, config.buy_thr, config.sell_thr)
        actual_ret = float(futures[query_index, 2])
        actual = _signal(actual_ret, config.buy_thr, config.sell_thr)
        confusion[actual][predicted] += 1
        n += 1

        if predicted == "BUY":
            buy += 1
            buy_correct += int(actual_ret > 0.0)
        elif predicted == "SELL":
            sell += 1
            sell_correct += int(actual_ret < 0.0)
        else:
            hold += 1
            hold_correct += int(-config.sell_thr <= actual_ret <= config.buy_thr)

    active = buy + sell
    total_correct = buy_correct + sell_correct + hold_correct
    return {
        "n": n,
        "buy": buy,
        "sell": sell,
        "hold": hold,
        "coverage_pct": 100.0 * active / n if n else 0.0,
        "buy_da_pct": 100.0 * buy_correct / buy if buy else 0.0,
        "sell_da_pct": 100.0 * sell_correct / sell if sell else 0.0,
        "hold_da_pct": 100.0 * hold_correct / hold if hold else 0.0,
        "overall_da_pct": 100.0 * total_correct / n if n else 0.0,
        "active_acc_pct": 100.0 * (buy_correct + sell_correct) / active if active else 0.0,
        "confusion": confusion,
    }


def _selection_score(metrics: dict[str, object]) -> float:
    buy = float(metrics["buy_da_pct"])
    sell = float(metrics["sell_da_pct"])
    active = float(metrics["active_acc_pct"])
    overall = float(metrics["overall_da_pct"])
    coverage = float(metrics["coverage_pct"])
    return 0.30 * buy + 0.30 * sell + 0.20 * active + 0.10 * overall + 0.10 * coverage


def _gamma_sample(rng: random.Random, alpha: float) -> float:
    return rng.gammavariate(alpha, 1.0)


def _dirichlet_prior(rng: random.Random, alpha: tuple[float, float, float, float, float]) -> list[float]:
    draws = np.asarray([_gamma_sample(rng, max(a, 1e-3)) for a in alpha], dtype=np.float64)
    draws /= draws.sum()
    return [float(value) for value in draws]


def _candidate_configs(trials: int, seed: int, space: SearchSpace) -> list[HeadConfig]:
    rng = random.Random(seed)
    configs: list[HeadConfig] = [
        HeadConfig(k=5, buy_thr=2.0, sell_thr=2.0, return_weights=DEFAULT_RETURN_WEIGHTS)
    ]
    for k in space.k_choices:
        for threshold in space.threshold_grid:
            configs.append(
                HeadConfig(
                    k=k,
                    buy_thr=threshold,
                    sell_thr=threshold,
                    return_weights=DEFAULT_RETURN_WEIGHTS,
                )
            )
    for _ in range(trials):
        weights = _dirichlet_prior(rng, space.prior_alpha)
        configs.append(
            HeadConfig(
                k=rng.choice(space.k_choices),
                buy_thr=round(rng.uniform(*space.buy_range), 2),
                sell_thr=round(rng.uniform(*space.sell_range), 2),
                return_weights={
                    "1d": round(weights[0], 4),
                    "3d": round(weights[1], 4),
                    "7d": round(weights[2], 4),
                    "15d": round(weights[3], 4),
                    "30d": round(weights[4], 4),
                },
            )
        )
    deduped: list[HeadConfig] = []
    seen: set[str] = set()
    for config in configs:
        key = json.dumps(config.to_json(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(config)
    return deduped


def _fold_query_indices(days: list[date], fold: Fold) -> tuple[list[int], list[int]]:
    val_queries = [
        idx for idx, day in enumerate(days)
        if fold.val_start <= day <= fold.val_end
    ]
    test_queries = [
        idx for idx, day in enumerate(days)
        if fold.test_start <= day <= fold.test_end
    ]
    return val_queries, test_queries


def _summarize_fold_metrics(results: list[dict[str, object]]) -> dict[str, float]:
    if not results:
        return {}
    keys = ("buy_da_pct", "sell_da_pct", "hold_da_pct", "overall_da_pct", "active_acc_pct", "coverage_pct")
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(result[key]) for result in results], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_std"] = float(values.std())
    return summary


def _run_walkforward(
    name: str,
    top_map: dict[int, np.ndarray],
    days: list[date],
    futures: np.ndarray,
    folds: list[Fold],
    configs: list[HeadConfig],
    progress_every: int,
) -> dict[str, object]:
    per_fold: list[dict[str, object]] = []
    started = time.time()

    for fold_index, fold in enumerate(folds, start=1):
        val_queries, test_queries = _fold_query_indices(days, fold)
        best_config = configs[0]
        best_val = _evaluate_indices(val_queries, top_map, futures, best_config)
        best_score = _selection_score(best_val)
        trial_log: list[dict[str, object]] = []

        for config_index, config in enumerate(configs, start=1):
            val_metrics = _evaluate_indices(val_queries, top_map, futures, config)
            score = _selection_score(val_metrics)
            trial_log.append(
                {
                    "candidate_index": config_index,
                    "selection_score": score,
                    "config": config.to_json(),
                    "val_metrics": val_metrics,
                }
            )
            if score > best_score:
                best_config = config
                best_val = val_metrics
                best_score = score

        test_metrics = _evaluate_indices(test_queries, top_map, futures, best_config)
        per_fold.append(
            {
                "fold": {
                    "index": fold.index,
                    "train_start": fold.train_start.isoformat(),
                    "train_end": fold.train_end.isoformat(),
                    "val_start": fold.val_start.isoformat(),
                    "val_end": fold.val_end.isoformat(),
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                },
                "best_config": best_config.to_json(),
                "best_val_selection_score": best_score,
                "val_metrics": best_val,
                "test_metrics": test_metrics,
                "top_val_trials": sorted(
                    trial_log,
                    key=lambda item: float(item["selection_score"]),
                    reverse=True,
                )[:10],
            }
        )
        if progress_every > 0 and (
            fold_index == 1
            or fold_index % progress_every == 0
            or fold_index == len(folds)
        ):
            elapsed = time.time() - started
            print(
                f"[{name}] folds {fold_index}/{len(folds)} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return {
        "name": name,
        "fold_count": len(per_fold),
        "per_fold": per_fold,
        "summary": {
            "val": _summarize_fold_metrics([fold["val_metrics"] for fold in per_fold]),
            "test": _summarize_fold_metrics([fold["test_metrics"] for fold in per_fold]),
        },
    }


def _metric_mean_std(metrics_list: list[dict[str, object]], key: str) -> tuple[float, float]:
    values = np.asarray([float(item[key]) for item in metrics_list], dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def _stability_objective(
    metrics_list: list[dict[str, object]],
    variance_penalty: float,
) -> float:
    if not metrics_list:
        return float("-inf")
    score_vals = np.asarray([_selection_score(item) for item in metrics_list], dtype=np.float64)
    overall_vals = np.asarray([float(item["overall_da_pct"]) for item in metrics_list], dtype=np.float64)
    active_vals = np.asarray([float(item["active_acc_pct"]) for item in metrics_list], dtype=np.float64)
    coverage_vals = np.asarray([float(item["coverage_pct"]) for item in metrics_list], dtype=np.float64)
    return float(
        score_vals.mean()
        - variance_penalty * score_vals.std()
        - 0.35 * variance_penalty * overall_vals.std()
        - 0.20 * variance_penalty * active_vals.std()
        - 0.10 * variance_penalty * coverage_vals.std()
    )


def _run_shared_stable_config(
    name: str,
    top_map: dict[int, np.ndarray],
    days: list[date],
    futures: np.ndarray,
    folds: list[Fold],
    configs: list[HeadConfig],
    variance_penalty: float,
) -> dict[str, object]:
    fold_indices = [_fold_query_indices(days, fold) for fold in folds]
    leaderboard: list[dict[str, object]] = []
    best_config = configs[0]
    best_score = float("-inf")
    best_val_metrics: list[dict[str, object]] = []

    for config in configs:
        val_metrics = [
            _evaluate_indices(val_queries, top_map, futures, config)
            for val_queries, _ in fold_indices
        ]
        stable_score = _stability_objective(val_metrics, variance_penalty)
        leaderboard.append(
            {
                "config": config.to_json(),
                "stable_val_score": stable_score,
                "val_summary": _summarize_fold_metrics(val_metrics),
            }
        )
        if stable_score > best_score:
            best_config = config
            best_score = stable_score
            best_val_metrics = val_metrics

    test_metrics = [
        _evaluate_indices(test_queries, top_map, futures, best_config)
        for _, test_queries in fold_indices
    ]
    return {
        "name": name,
        "variance_penalty": variance_penalty,
        "best_config": best_config.to_json(),
        "best_stable_val_score": best_score,
        "summary": {
            "val": _summarize_fold_metrics(best_val_metrics),
            "test": _summarize_fold_metrics(test_metrics),
        },
        "top_configs": sorted(
            leaderboard,
            key=lambda item: float(item["stable_val_score"]),
            reverse=True,
        )[:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Colab-friendly walk-forward tuning for the knn_returns head. "
            "Requires only numpy plus the dataset/artifact JSON files."
        )
    )
    parser.add_argument("--data", default="real_optimizer_finbert.json")
    parser.add_argument("--artifact", default="learned_retriever_finbert.json")
    parser.add_argument("--weights", default="")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--window", choices=("expanding", "rolling", "both"), default="both")
    parser.add_argument("--config-mode", choices=("per_fold", "shared_stable", "both"), default="both")
    parser.add_argument("--search-space", choices=("wide", "focused"), default="focused")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--kmax", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precompute-progress-every", type=int, default=250)
    parser.add_argument("--fold-progress-every", type=int, default=2)
    parser.add_argument("--variance-penalty", type=float, default=0.35)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    trials = args.trials
    if trials is None:
        trials = 60 if args.mode == "quick" else 240

    rows, days, futures = _load_dataset(Path(args.data))
    factor_mat = _normalize_rows(rows, "factor_vec", 75)
    indicator_mat = _normalize_rows(rows, "indicator_vec", 5)
    price_mat = _normalize_rows(rows, "price_vec", 60)
    event_mat = _normalize_rows(rows, "event_vec", 85)
    fixed_weights = _load_search_weights(Path(args.weights) if args.weights else None)
    block_dims, diagonal, block_scales = _load_learned_metric(Path(args.artifact))
    learned_blocks = _precompute_learned_blocks(
        event_mat,
        factor_mat,
        indicator_mat,
        price_mat,
        block_dims,
        diagonal,
    )
    fixed_top, learned_top = _precompute_neighbors(
        days,
        futures,
        factor_mat,
        indicator_mat,
        price_mat,
        learned_blocks,
        block_scales,
        fixed_weights,
        args.kmax,
        args.precompute_progress_every,
    )

    if args.search_space == "focused":
        search_space = SearchSpace(
            k_choices=(3, 5, 7, 10, 15, 20),
            buy_range=(1.25, 3.10),
            sell_range=(1.25, 3.50),
            threshold_grid=(1.5, 2.0, 2.5, 3.0),
            prior_alpha=(1.8, 1.6, 2.4, 1.6, 1.2),
        )
    else:
        search_space = SearchSpace(
            k_choices=K_CHOICES,
            buy_range=(1.0, 4.0),
            sell_range=(1.0, 4.0),
            threshold_grid=(2.0, 2.5, 3.0),
            prior_alpha=(1.0, 1.0, 1.0, 1.0, 1.0),
        )

    configs = _candidate_configs(trials, args.seed, search_space)
    print(f"[search-space] {len(configs)} unique configs", flush=True)

    specs: list[tuple[str, WindowSpec]] = []
    if args.window in {"expanding", "both"}:
        specs.append(
            (
                "expanding",
                WindowSpec(
                    train_months=60 if args.mode == "quick" else 72,
                    val_months=6,
                    test_months=3,
                    step_months=3,
                    expanding=True,
                ),
            )
        )
    if args.window in {"rolling", "both"}:
        specs.append(
            (
                "rolling",
                WindowSpec(
                    train_months=24 if args.mode == "quick" else 36,
                    val_months=6,
                    test_months=3,
                    step_months=3,
                    expanding=False,
                ),
            )
        )

    output: dict[str, object] = {
        "data": args.data,
        "artifact": args.artifact,
        "weights": args.weights or None,
        "mode": args.mode,
        "trial_count": len(configs),
        "kmax": args.kmax,
        "fixed_search_weights": {
            "w_factor": fixed_weights[0],
            "w_indicator": fixed_weights[1],
            "w_price": fixed_weights[2],
        },
        "windows": [],
    }

    for window_name, spec in specs:
        folds = _build_folds(days, spec)
        print(f"[{window_name}] {len(folds)} folds", flush=True)
        output["windows"].append(
            {
                "window": window_name,
                "spec": {
                    "train_months": spec.train_months,
                    "val_months": spec.val_months,
                    "test_months": spec.test_months,
                    "step_months": spec.step_months,
                    "expanding": spec.expanding,
                },
                "per_fold": (
                    {
                        "learned": _run_walkforward(
                            f"{window_name}:learned",
                            learned_top,
                            days,
                            futures,
                            folds,
                            configs,
                            args.fold_progress_every,
                        ),
                        "fixed": _run_walkforward(
                            f"{window_name}:fixed",
                            fixed_top,
                            days,
                            futures,
                            folds,
                            configs,
                            args.fold_progress_every,
                        ),
                    }
                    if args.config_mode in {"per_fold", "both"}
                    else None
                ),
                "shared_stable": (
                    {
                        "learned": _run_shared_stable_config(
                            f"{window_name}:learned:shared_stable",
                            learned_top,
                            days,
                            futures,
                            folds,
                            configs,
                            args.variance_penalty,
                        ),
                        "fixed": _run_shared_stable_config(
                            f"{window_name}:fixed:shared_stable",
                            fixed_top,
                            days,
                            futures,
                            folds,
                            configs,
                            args.variance_penalty,
                        ),
                    }
                    if args.config_mode in {"shared_stable", "both"}
                    else None
                ),
            }
        )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "windows": len(output["windows"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
