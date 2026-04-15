from __future__ import annotations

import argparse
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


DEFAULT_BASELINE_WEIGHTS = (0.35, 0.20, 0.45)
W1_RANGE = (0.20, 0.70)
W2_RANGE = (0.10, 0.45)
W3_RANGE = (0.10, 0.65)
HORIZON_STEP = {
    "1d": 1,
    "7d": 7,
    "30d": 30,
}


@dataclass
class Row:
    date: str
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray
    future_return_1d: float
    future_return_7d: float
    future_return_30d: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] != b.shape[0]:
        return 0.0
    return float(np.dot(a, b))


def weighted_similarity(q: Row, c: Row, w1: float, w2: float, w3: float) -> float:
    return (
        w1 * _cosine(q.factor_vec, c.factor_vec)
        + w2 * _cosine(q.indicator_vec, c.indicator_vec)
        + w3 * _cosine(q.price_vec, c.price_vec)
    )


def _get_horizon_return(row: Row, horizon: str) -> float:
    if horizon == "1d":
        return row.future_return_1d
    if horizon == "30d":
        return row.future_return_30d
    return row.future_return_7d


def _compute_sharpe(returns: list[float], horizon: str, mode: str) -> float:
    if not returns:
        return 0.0

    arr = np.array(returns, dtype=np.float64)
    eval_arr = arr
    if mode == "nonoverlap":
        step = HORIZON_STEP.get(horizon, 1)
        eval_arr = arr[::step]
    if eval_arr.size == 0:
        return 0.0
    std = eval_arr.std()
    return float(eval_arr.mean() / std) if std > 1e-8 else 0.0


def _evaluate_query_against_pool(
    query: Row,
    pool: list[Row],
    w1: float,
    w2: float,
    w3: float,
    k: int,
    horizon: str,
) -> tuple[int, float]:
    scored = [(weighted_similarity(query, p, w1, w2, w3), p) for p in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    if not top:
        return 0, 0.0

    outcomes = [_get_horizon_return(item[1], horizon) for item in top]
    pred = 1 if float(np.mean(outcomes)) > 0 else -1
    actual_ret = _get_horizon_return(query, horizon)
    actual = 1 if actual_ret > 0 else -1

    return (1 if pred == actual else 0), float(pred * actual_ret)


def load_rows(path: Path) -> list[Row]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[Row] = []
    for item in payload:
        rows.append(
            Row(
                date=str(item["date"]),
                factor_vec=np.array(item["factor_vec"], dtype=np.float32),
                indicator_vec=np.array(item["indicator_vec"], dtype=np.float32),
                price_vec=np.array(item["price_vec"], dtype=np.float32),
                future_return_1d=float(item.get("future_return_1d", 0.0)),
                future_return_7d=float(item.get("future_return_7d", 0.0)),
                future_return_30d=float(item.get("future_return_30d", 0.0)),
            )
        )
    rows.sort(key=lambda x: x.date)
    return rows


def validate_rows(rows: list[Row]) -> None:
    if not rows:
        raise ValueError("No records loaded from input data")

    factor_dim = rows[0].factor_vec.shape[0]
    indicator_dim = rows[0].indicator_vec.shape[0]
    price_dim = rows[0].price_vec.shape[0]
    if factor_dim == 0 or indicator_dim == 0 or price_dim == 0:
        raise ValueError("Vectors must be non-empty for all groups")

    for i, row in enumerate(rows):
        if row.factor_vec.shape[0] != factor_dim:
            raise ValueError(
                f"Row {i} has inconsistent factor_vec dim: {row.factor_vec.shape[0]} vs {factor_dim}"
            )
        if row.indicator_vec.shape[0] != indicator_dim:
            raise ValueError(
                f"Row {i} has inconsistent indicator_vec dim: {row.indicator_vec.shape[0]} vs {indicator_dim}"
            )
        if row.price_vec.shape[0] != price_dim:
            raise ValueError(
                f"Row {i} has inconsistent price_vec dim: {row.price_vec.shape[0]} vs {price_dim}"
            )


def walk_forward_evaluate(
    rows: list[Row],
    w1: float,
    w2: float,
    w3: float,
    k: int,
    warmup: int,
    horizon: str = "7d",
    sharpe_mode: str = "nonoverlap",
) -> dict[str, float]:
    correct = 0
    total = 0
    strategy_returns: list[float] = []

    for i in range(warmup, len(rows)):
        query = rows[i]
        pool = rows[:i]
        if not pool:
            continue

        is_correct, strategy_ret = _evaluate_query_against_pool(
            query=query,
            pool=pool,
            w1=w1,
            w2=w2,
            w3=w3,
            k=k,
            horizon=horizon,
        )

        correct += is_correct
        total += 1
        strategy_returns.append(strategy_ret)

    if total == 0:
        return {"da": 0.0, "sharpe": 0.0, "combined": 0.0}

    da = correct / total
    sharpe = _compute_sharpe(strategy_returns, horizon=horizon, mode=sharpe_mode)
    combined = 0.6 * da + 0.4 * min(max(sharpe, -2.0), 2.0) / 2.0
    return {"da": da, "sharpe": sharpe, "combined": combined}


def evaluate(rows: list[Row], w1: float, w2: float, w3: float, k: int, warmup: int) -> dict[str, float]:
    # Keep compatibility for benchmark_weights.py
    return walk_forward_evaluate(rows, w1, w2, w3, k=k, warmup=warmup)


def evaluate_holdout(
    w1: float,
    w2: float,
    w3: float,
    train_rows: list[Row],
    holdout_rows: list[Row],
    k: int,
    horizon: str,
    sharpe_mode: str,
) -> dict[str, float]:
    if not holdout_rows:
        return {"da": 0.0, "sharpe": 0.0, "combined": 0.0}

    correct = 0
    total = 0
    returns: list[float] = []
    for i, query in enumerate(holdout_rows):
        pool = train_rows + holdout_rows[:i]
        if not pool:
            continue

        is_correct, strategy_ret = _evaluate_query_against_pool(
            query=query,
            pool=pool,
            w1=w1,
            w2=w2,
            w3=w3,
            k=k,
            horizon=horizon,
        )
        correct += is_correct
        total += 1
        returns.append(strategy_ret)

    if total == 0:
        return {"da": 0.0, "sharpe": 0.0, "combined": 0.0}

    da = correct / total
    sharpe = _compute_sharpe(returns, horizon=horizon, mode=sharpe_mode)
    combined = 0.6 * da + 0.4 * min(max(sharpe, -2.0), 2.0) / 2.0
    return {"da": da, "sharpe": sharpe, "combined": combined}


def split_train_holdout(
    rows: list[Row], holdout_ratio: float, min_holdout: int
) -> tuple[list[Row], list[Row]]:
    if holdout_ratio <= 0:
        return rows, []

    n = len(rows)
    n_holdout = max(min_holdout, int(round(n * holdout_ratio)))
    if n_holdout >= n:
        raise ValueError(
            f"Holdout size {n_holdout} must be smaller than total records {n}"
        )
    return rows[:-n_holdout], rows[-n_holdout:]


def build_temporal_cv_folds(
    rows: list[Row],
    warmup: int,
    n_folds: int,
    holdout_ratio: float,
    min_holdout: int,
) -> list[tuple[list[Row], list[Row]]]:
    if n_folds <= 1:
        return []

    n = len(rows)
    min_train = warmup + 1
    available = n - min_train
    if available <= 0:
        raise ValueError(
            f"Not enough records for CV: n={n}, warmup={warmup}."
        )

    fold_size = max(min_holdout, int(round(n * holdout_ratio)))
    fold_size = max(1, min(fold_size, available))
    max_non_overlap = max(1, available // fold_size)
    folds_to_use = min(n_folds, max_non_overlap)

    folds: list[tuple[list[Row], list[Row]]] = []
    valid_start0 = n - folds_to_use * fold_size

    for i in range(folds_to_use):
        valid_start = valid_start0 + i * fold_size
        valid_end = valid_start + fold_size
        train_part = rows[:valid_start]
        valid_part = rows[valid_start:valid_end]
        if len(train_part) <= warmup or not valid_part:
            continue
        folds.append((train_part, valid_part))

    if not folds:
        train_part, valid_part = split_train_holdout(rows, holdout_ratio, min_holdout)
        if len(train_part) <= warmup or not valid_part:
            raise ValueError("Unable to build CV folds with current settings")
        folds = [(train_part, valid_part)]

    return folds


def _require_optuna() -> Any:
    if optuna is None:
        raise RuntimeError(
            "Optuna is required for Bayesian optimization. Install with: pip install optuna"
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def select_stable_median_weights(study: Any, top_k: int) -> tuple[float, float, float, dict[str, Any]]:
    o = _require_optuna()
    completed = [
        t
        for t in study.trials
        if t.value is not None
        and t.state == o.trial.TrialState.COMPLETE
        and "w1_factor" in t.params
        and "w2_indicator" in t.params
    ]
    if not completed:
        raise ValueError("No completed trials found for stable weight selection")

    completed.sort(key=lambda t: t.value, reverse=True)
    top_k_eff = max(1, min(top_k, len(completed)))
    top_trials = completed[:top_k_eff]

    w1_vals = np.array([float(t.params["w1_factor"]) for t in top_trials], dtype=np.float64)
    w2_vals = np.array([float(t.params["w2_indicator"]) for t in top_trials], dtype=np.float64)
    w3_vals = 1.0 - w1_vals - w2_vals

    w1_med = float(np.median(w1_vals))
    w2_med = float(np.median(w2_vals))
    w3_med = float(np.median(w3_vals))
    total = w1_med + w2_med + w3_med

    def _best_fallback(mode: str) -> tuple[float, float, float, dict[str, Any]]:
        best = completed[0]
        w1 = float(best.params["w1_factor"])
        w2 = float(best.params["w2_indicator"])
        w3 = 1.0 - w1 - w2
        return w1, w2, w3, {
            "selection_mode": mode,
            "top_k_requested": top_k,
            "top_k_used": top_k_eff,
        }

    if total <= 1e-8:
        return _best_fallback("best_trial_fallback_zero_sum")

    w1 = w1_med / total
    w2 = w2_med / total
    w3 = w3_med / total
    in_bounds = (
        W1_RANGE[0] <= w1 <= W1_RANGE[1]
        and W2_RANGE[0] <= w2 <= W2_RANGE[1]
        and W3_RANGE[0] <= w3 <= W3_RANGE[1]
    )
    if not in_bounds:
        return _best_fallback("best_trial_fallback_out_of_bounds")

    return w1, w2, w3, {
        "selection_mode": "median_top_trials",
        "top_k_requested": top_k,
        "top_k_used": top_k_eff,
    }


def make_objective(
    rows: list[Row],
    horizon: str,
    k: int,
    warmup: int,
    sharpe_mode: str,
    cv_folds: int,
    cv_holdout_ratio: float,
    cv_min_holdout: int,
):
    cv_splits = (
        build_temporal_cv_folds(rows, warmup, cv_folds, cv_holdout_ratio, cv_min_holdout)
        if cv_folds > 1
        else []
    )

    def objective(trial: Any) -> float:
        w1 = trial.suggest_float("w1_factor", W1_RANGE[0], W1_RANGE[1])
        w2 = trial.suggest_float("w2_indicator", W2_RANGE[0], W2_RANGE[1])
        w3 = 1.0 - w1 - w2
        if w3 < W3_RANGE[0] or w3 > W3_RANGE[1]:
            return 0.0

        trial.set_user_attr("w3_price", round(w3, 4))

        if cv_folds <= 1:
            metrics = walk_forward_evaluate(
                rows, w1, w2, w3, k=k, warmup=warmup, horizon=horizon, sharpe_mode=sharpe_mode
            )
            trial.set_user_attr("da", round(metrics["da"], 4))
            trial.set_user_attr("sharpe", round(metrics["sharpe"], 4))
            trial.set_user_attr("combined_mode", "single_walk_forward")
            return metrics["combined"]

        fold_metrics: list[dict[str, float]] = []
        for fold_train, fold_valid in cv_splits:
            fold_metrics.append(
                evaluate_holdout(
                    w1,
                    w2,
                    w3,
                    train_rows=fold_train,
                    holdout_rows=fold_valid,
                    k=k,
                    horizon=horizon,
                    sharpe_mode=sharpe_mode,
                )
            )

        combined_vals = np.array([m["combined"] for m in fold_metrics], dtype=np.float64)
        da_vals = np.array([m["da"] for m in fold_metrics], dtype=np.float64)
        sharpe_vals = np.array([m["sharpe"] for m in fold_metrics], dtype=np.float64)
        combined_median = float(np.median(combined_vals))

        trial.set_user_attr("da", round(float(np.median(da_vals)), 4))
        trial.set_user_attr("sharpe", round(float(np.median(sharpe_vals)), 4))
        trial.set_user_attr("combined_mode", f"median_{len(cv_splits)}_folds")
        trial.set_user_attr("combined_mean", round(float(np.mean(combined_vals)), 4))
        trial.set_user_attr("combined_std", round(float(np.std(combined_vals)), 4))
        return combined_median

    return objective


def grid_search(
    rows: list[Row],
    step: float,
    horizon: str,
    k: int,
    warmup: int,
    sharpe_mode: str,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    w1_range = np.arange(0.20, 0.71, step)
    w2_range = np.arange(0.10, 0.46, step)

    for w1 in w1_range:
        for w2 in w2_range:
            w3 = 1.0 - float(w1) - float(w2)
            if w3 < W3_RANGE[0] or w3 > W3_RANGE[1]:
                continue

            metrics = walk_forward_evaluate(
                rows,
                float(w1),
                float(w2),
                float(w3),
                k=k,
                warmup=warmup,
                horizon=horizon,
                sharpe_mode=sharpe_mode,
            )
            results.append(
                {
                    "w1_factor": round(float(w1), 4),
                    "w2_indicator": round(float(w2), 4),
                    "w3_price": round(float(w3), 4),
                    "da": round(metrics["da"], 4),
                    "sharpe": round(metrics["sharpe"], 4),
                    "combined": round(metrics["combined"], 4),
                }
            )

    results.sort(key=lambda x: x["combined"], reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimize w1/w2/w3 for StockMem weighted similarity")
    parser.add_argument("--data", required=True, help="Path to vectorized dataset JSON")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--horizon", default="7d", choices=["1d", "7d", "30d"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--min-holdout", type=int, default=120)
    parser.add_argument("--sharpe-mode", choices=["overlap", "nonoverlap"], default="nonoverlap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=1)
    parser.add_argument("--cv-holdout-ratio", type=float, default=None)
    parser.add_argument("--cv-min-holdout", type=int, default=None)
    parser.add_argument("--stable-top-k", type=int, default=10)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--output", default="stockmem/config/weights.optimized.json")
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    validate_rows(rows)
    train_rows, holdout_rows = split_train_holdout(rows, args.holdout_ratio, args.min_holdout)

    cv_holdout_ratio = args.cv_holdout_ratio if args.cv_holdout_ratio is not None else args.holdout_ratio
    cv_min_holdout = args.cv_min_holdout if args.cv_min_holdout is not None else args.min_holdout

    if args.cv_folds > 1:
        _ = build_temporal_cv_folds(
            rows=train_rows,
            warmup=args.warmup,
            n_folds=args.cv_folds,
            holdout_ratio=cv_holdout_ratio,
            min_holdout=cv_min_holdout,
        )

    o = _require_optuna()
    study = o.create_study(
        direction="maximize",
        sampler=o.samplers.TPESampler(seed=args.seed),
        study_name="stockmem_weight_optimization",
    )

    objective = make_objective(
        rows=train_rows,
        horizon=args.horizon,
        k=args.k,
        warmup=args.warmup,
        sharpe_mode=args.sharpe_mode,
        cv_folds=args.cv_folds,
        cv_holdout_ratio=cv_holdout_ratio,
        cv_min_holdout=cv_min_holdout,
    )
    study.optimize(objective, n_trials=max(1, args.trials), show_progress_bar=False)

    best = study.best_trial
    w1_best = float(best.params["w1_factor"])
    w2_best = float(best.params["w2_indicator"])
    w3_best = 1.0 - w1_best - w2_best

    stable_w1, stable_w2, stable_w3, stable_meta = select_stable_median_weights(
        study=study,
        top_k=args.stable_top_k,
    )

    baseline = DEFAULT_BASELINE_WEIGHTS

    train_metrics_best = walk_forward_evaluate(
        train_rows,
        w1_best,
        w2_best,
        w3_best,
        k=args.k,
        warmup=args.warmup,
        horizon=args.horizon,
        sharpe_mode=args.sharpe_mode,
    )
    train_metrics_stable = walk_forward_evaluate(
        train_rows,
        stable_w1,
        stable_w2,
        stable_w3,
        k=args.k,
        warmup=args.warmup,
        horizon=args.horizon,
        sharpe_mode=args.sharpe_mode,
    )

    holdout_metrics_best = evaluate_holdout(
        w1_best,
        w2_best,
        w3_best,
        train_rows,
        holdout_rows,
        args.k,
        args.horizon,
        args.sharpe_mode,
    )
    holdout_metrics_stable = evaluate_holdout(
        stable_w1,
        stable_w2,
        stable_w3,
        train_rows,
        holdout_rows,
        args.k,
        args.horizon,
        args.sharpe_mode,
    )
    holdout_metrics_baseline = evaluate_holdout(
        baseline[0],
        baseline[1],
        baseline[2],
        train_rows,
        holdout_rows,
        args.k,
        args.horizon,
        args.sharpe_mode,
    )

    output = {
        "w1_factor": round(stable_w1, 4),
        "w2_indicator": round(stable_w2, 4),
        "w3_price": round(stable_w3, 4),
        "optimized_at": datetime.now().isoformat(),
        "best_score": round(float(best.value), 4),
        "metric": (
            f"0.6*DA + 0.4*Sharpe (horizon={args.horizon}, "
            f"objective={'median_cv' if args.cv_folds > 1 else 'single_walk_forward'})"
        ),
        "da": float(best.user_attrs.get("da", 0.0)),
        "sharpe": float(best.user_attrs.get("sharpe", 0.0)),
        "n_trials": args.trials,
        "n_records": len(rows),
        "train_records": len(train_rows),
        "holdout_records": len(holdout_rows),
        "warmup": args.warmup,
        "sharpe_mode": args.sharpe_mode,
        "seed": args.seed,
        "cv": {
            "folds_requested": args.cv_folds,
            "holdout_ratio": cv_holdout_ratio,
            "min_holdout": cv_min_holdout,
        },
        "stable_selection": {
            "mode": stable_meta["selection_mode"],
            "top_k_requested": stable_meta["top_k_requested"],
            "top_k_used": stable_meta["top_k_used"],
        },
        "best_trial_weights": {
            "w1_factor": round(w1_best, 4),
            "w2_indicator": round(w2_best, 4),
            "w3_price": round(w3_best, 4),
        },
        "train_metrics": {
            "da": round(train_metrics_stable["da"], 4),
            "sharpe": round(train_metrics_stable["sharpe"], 4),
            "combined": round(train_metrics_stable["combined"], 4),
        },
        "train_metrics_best": {
            "da": round(train_metrics_best["da"], 4),
            "sharpe": round(train_metrics_best["sharpe"], 4),
            "combined": round(train_metrics_best["combined"], 4),
        },
        "holdout_metrics_best": {
            "da": round(holdout_metrics_best["da"], 4),
            "sharpe": round(holdout_metrics_best["sharpe"], 4),
            "combined": round(holdout_metrics_best["combined"], 4),
        },
        "holdout_metrics_stable": {
            "da": round(holdout_metrics_stable["da"], 4),
            "sharpe": round(holdout_metrics_stable["sharpe"], 4),
            "combined": round(holdout_metrics_stable["combined"], 4),
        },
        "holdout_metrics_baseline": {
            "da": round(holdout_metrics_baseline["da"], 4),
            "sharpe": round(holdout_metrics_baseline["sharpe"], 4),
            "combined": round(holdout_metrics_baseline["combined"], 4),
        },
    }

    if args.ablation and holdout_rows:
        w12 = w1_best + w2_best
        if w12 > 1e-8:
            w1_ab = w1_best / w12
            w2_ab = w2_best / w12
            w3_ab = 0.0
            ablation = evaluate_holdout(
                w1_ab,
                w2_ab,
                w3_ab,
                train_rows,
                holdout_rows,
                args.k,
                args.horizon,
                args.sharpe_mode,
            )
            output["holdout_metrics_ablation_w3_zero"] = {
                "da": round(ablation["da"], 4),
                "sharpe": round(ablation["sharpe"], 4),
                "combined": round(ablation["combined"], 4),
            }

    if args.grid:
        grid_results = grid_search(
            rows=train_rows,
            step=0.05,
            horizon=args.horizon,
            k=args.k,
            warmup=args.warmup,
            sharpe_mode=args.sharpe_mode,
        )
        grid_path = Path(args.output).parent / "grid_results.json"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_path.write_text(json.dumps(grid_results, indent=2), encoding="utf-8")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
