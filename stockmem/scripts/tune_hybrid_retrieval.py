from __future__ import annotations

import argparse
import json
from dataclasses import replace
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from stockmem.scripts.cem_dataset import LabeledRow, label_rows
from stockmem.scripts.evaluate_hybrid_retrieval import (
    DEFAULT_SEARCH_WEIGHTS,
    MethodEvaluation,
    _fixed_ranked,
    evaluate_ranker,
)
from stockmem.scripts.hybrid_reranking import (
    HybridRankedCandidate,
    HybridRerankWeights,
    rerank_knn_candidates,
)
from stockmem.scripts.optimize_weights import load_rows, validate_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


@dataclass(frozen=True)
class RollingFold:
    index: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date
    embargo_days: int


def _log(message: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} UTC] {message}", flush=True)


def _month_floor(day: date) -> date:
    return date(day.year, day.month, 1)


def _add_months(day: date, months: int) -> date:
    year = day.year + (day.month - 1 + months) // 12
    month = (day.month - 1 + months) % 12 + 1
    return date(year, month, 1)


def _month_end(day: date) -> date:
    return _add_months(day, 1) - timedelta(days=1)


def _generate_weight_grid() -> list[HybridRerankWeights]:
    values_knn = (0.3, 0.4, 0.5, 0.6)
    values_learned = (0.1, 0.2, 0.3, 0.4)
    values_regime = (0.0, 0.1, 0.2)
    values_prior = (0.0, 0.1, 0.2)
    out: list[HybridRerankWeights] = []
    for w_knn in values_knn:
        for w_learned in values_learned:
            for w_regime in values_regime:
                for w_prior in values_prior:
                    if abs((w_knn + w_learned + w_regime + w_prior) - 1.0) > 1e-9:
                        continue
                    out.append(
                        HybridRerankWeights(
                            w_knn=w_knn,
                            w_learned=w_learned,
                            w_regime=w_regime,
                            w_prior=w_prior,
                        )
                    )
    return out


def build_folds(
    labeled: Sequence[LabeledRow],
    *,
    train_months: int = 36,
    val_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
    embargo_days: int = 7,
) -> list[RollingFold]:
    if not labeled:
        return []
    min_date = min(row.parsed_date for row in labeled)
    max_date = max(row.parsed_date for row in labeled)
    start = _month_floor(min_date)
    folds: list[RollingFold] = []
    index = 1
    while True:
        train_start = start
        train_end = _add_months(train_start, train_months) - timedelta(days=1)
        val_start = train_end + timedelta(days=1)
        val_end = _add_months(val_start, val_months) - timedelta(days=1)
        test_start = val_end + timedelta(days=1)
        test_end = _add_months(test_start, test_months) - timedelta(days=1)
        if test_end > max_date:
            break
        folds.append(
            RollingFold(
                index=index,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=embargo_days,
            )
        )
        start = _add_months(start, step_months)
        index += 1
    return folds


def _in_window(day: date, start: date, end: date) -> bool:
    return start <= day <= end


def _candidate_pool_for_fold(
    labeled: Sequence[LabeledRow],
    query: LabeledRow,
    fold: RollingFold,
) -> list[LabeledRow]:
    embargo_cutoff = query.parsed_date - timedelta(days=fold.embargo_days)
    return [
        candidate
        for candidate in labeled
        if fold.train_start <= candidate.parsed_date <= fold.train_end
        and candidate.parsed_date < query.parsed_date
        and candidate.parsed_date <= embargo_cutoff
    ]


def evaluate_fold_hybrid(
    labeled: Sequence[LabeledRow],
    *,
    fold: RollingFold,
    phase: str,
    search_weights: tuple[float, float, float],
    learned_metric: LearnedDiagonalMetric,
    rerank_weights: HybridRerankWeights,
    candidate_pool_size: int,
    top_k: int,
    buy_threshold: float,
    sell_threshold: float,
) -> MethodEvaluation:
    if phase not in {"val", "test"}:
        raise ValueError("phase must be 'val' or 'test'")
    window_start = fold.val_start if phase == "val" else fold.test_start
    window_end = fold.val_end if phase == "val" else fold.test_end

    def ranker(query: LabeledRow, _unused_pool: Sequence[LabeledRow]) -> list[HybridRankedCandidate]:
        pool = _candidate_pool_for_fold(labeled, query, fold)
        fixed_ranked = _fixed_ranked(query, pool, search_weights=search_weights)
        candidate_slice = fixed_ranked[:candidate_pool_size]
        return rerank_knn_candidates(
            query,
            [item.candidate for item in candidate_slice],
            learned_metric=learned_metric,
            baseline_scores=[float(item.score) for item in candidate_slice],
            weights=rerank_weights,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

    filtered = [
        replace(row, split=phase)
        for row in labeled
        if _in_window(row.parsed_date, window_start, window_end)
    ]
    return evaluate_ranker(
        filtered,
        method_name=f"hybrid_{phase}",
        ranker=ranker,
        search_weights=search_weights,
        split=phase,
        top_k=top_k,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )


def _selection_score(evaluation: MethodEvaluation) -> float:
    return (
        0.5 * evaluation.top5_same_d7_sign_rate
        + 0.3 * evaluation.ndcg_at_5
        + 0.2 * evaluation.downstream_da
    )


def _markdown_summary(
    *,
    folds: Sequence[RollingFold],
    best_weights: HybridRerankWeights,
    best_objective: float,
    fold_rows: Sequence[dict],
) -> str:
    lines = [
        "# Hybrid Retrieval Rolling Validation",
        "",
        f"- selected_weights: `{json.dumps(best_weights.as_dict(), sort_keys=True)}`",
        f"- selection_objective: `{best_objective:.6f}`",
        "",
        "## Folds",
        "",
        "| Fold | Train | Validation | Test |",
        "|---|---|---|---|",
    ]
    for fold in folds:
        lines.append(
            f"| {fold.index} | {fold.train_start} -> {fold.train_end} | {fold.val_start} -> {fold.val_end} | {fold.test_start} -> {fold.test_end} |"
        )
    lines.extend(
        [
            "",
            "## Selected Weights by Fold",
            "",
            "| Fold | Val score | Test score | Top5 same D7 sign | nDCG@5 | Downstream DA | Active Acc | Coverage | Weights |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in fold_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold_index"]),
                    f"{row['validation_selection_score']:.4f}",
                    f"{row['test_selection_score']:.4f}",
                    f"{row['test_top5_same_d7_sign_rate']:.4f}",
                    f"{row['test_ndcg_at_5']:.4f}",
                    f"{row['test_downstream_da']:.4f}",
                    f"{row['test_active_acc']:.4f}",
                    f"{row['test_coverage']:.4f}",
                    json.dumps(row["selected_weights"], sort_keys=True),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fold_to_dict(fold: RollingFold) -> dict[str, object]:
    return {
        "index": fold.index,
        "train_start": str(fold.train_start),
        "train_end": str(fold.train_end),
        "val_start": str(fold.val_start),
        "val_end": str(fold.val_end),
        "test_start": str(fold.test_start),
        "test_end": str(fold.test_end),
        "embargo_days": fold.embargo_days,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune hybrid StockMem retrieval weights with rolling validation."
    )
    parser.add_argument("--data", default="stockmem/data/real_optimizer_finbert.json")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--output-dir", default="artifacts/hybrid_retrieval_tuning")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=30)
    parser.add_argument("--buy-threshold", type=float, default=2.0)
    parser.add_argument("--sell-threshold", type=float, default=2.0)
    parser.add_argument("--variance-penalty", type=float, default=1.0)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-grid", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    validate_rows(rows)
    labeled = label_rows(rows, band="fixed", fixed_band=args.buy_threshold)
    learned_metric = LearnedDiagonalMetric.load(args.artifact)
    search_weights = DEFAULT_SEARCH_WEIGHTS
    weights_path = Path(args.weights)
    if weights_path.exists():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        source = payload.get("weights", payload)
        search_weights = (
            float(source["w1_factor"]),
            float(source["w2_indicator"]),
            float(source["w3_price"]),
        )

    folds = build_folds(labeled)
    grid = _generate_weight_grid()
    if args.max_folds is not None:
        folds = folds[: max(0, args.max_folds)]
    if args.max_grid is not None:
        grid = grid[: max(0, args.max_grid)]
    if not folds:
        raise ValueError("No rolling folds available for tuning")
    if not grid:
        raise ValueError("Hybrid weight grid is empty after applying limits")
    _log(
        "Starting rolling hybrid tuning "
        f"folds={len(folds)} grid={len(grid)} top_k={args.top_k} "
        f"candidate_pool_size={args.candidate_pool_size}"
    )
    fold_rows: list[dict] = []
    config_validation_scores: dict[str, list[float]] = {
        json.dumps(weights.as_dict(), sort_keys=True): [] for weights in grid
    }

    for fold in folds:
        _log(
            f"Fold {fold.index} start "
            f"train={fold.train_start}->{fold.train_end} "
            f"val={fold.val_start}->{fold.val_end} "
            f"test={fold.test_start}->{fold.test_end}"
        )
        candidate_scores: list[tuple[HybridRerankWeights, MethodEvaluation, float]] = []
        fold_started_at = datetime.utcnow()
        for idx, rerank_weights in enumerate(grid, start=1):
            val_eval = evaluate_fold_hybrid(
                labeled,
                fold=fold,
                phase="val",
                search_weights=search_weights,
                learned_metric=learned_metric,
                rerank_weights=rerank_weights,
                candidate_pool_size=args.candidate_pool_size,
                top_k=args.top_k,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
            )
            selection_score = _selection_score(val_eval)
            candidate_scores.append((rerank_weights, val_eval, selection_score))
            config_validation_scores[
                json.dumps(rerank_weights.as_dict(), sort_keys=True)
            ].append(selection_score)
            if idx == 1 or idx == len(grid) or idx % max(1, args.progress_every) == 0:
                elapsed = (datetime.utcnow() - fold_started_at).total_seconds()
                _log(
                    f"Fold {fold.index} progress {idx}/{len(grid)} "
                    f"val_score={selection_score:.4f} "
                    f"weights={json.dumps(rerank_weights.as_dict(), sort_keys=True)} "
                    f"elapsed={elapsed:.1f}s"
                )
        selected_weights, selected_val_eval, val_score = max(
            candidate_scores,
            key=lambda item: item[2],
        )
        _log(
            f"Fold {fold.index} selected "
            f"weights={json.dumps(selected_weights.as_dict(), sort_keys=True)} "
            f"val_score={val_score:.4f}"
        )
        test_eval = evaluate_fold_hybrid(
            labeled,
            fold=fold,
            phase="test",
            search_weights=search_weights,
            learned_metric=learned_metric,
            rerank_weights=selected_weights,
            candidate_pool_size=args.candidate_pool_size,
            top_k=args.top_k,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )
        fold_rows.append(
            {
                "fold_index": fold.index,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "val_start": str(fold.val_start),
                "val_end": str(fold.val_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "selected_weights": selected_weights.as_dict(),
                "validation_selection_score": val_score,
                "validation_metrics": asdict(selected_val_eval),
                "test_selection_score": _selection_score(test_eval),
                "test_top5_same_d7_sign_rate": test_eval.top5_same_d7_sign_rate,
                "test_ndcg_at_5": test_eval.ndcg_at_5,
                "test_downstream_da": test_eval.downstream_da,
                "test_active_acc": test_eval.active_acc,
                "test_coverage": test_eval.coverage,
                "test_metrics": asdict(test_eval),
            }
        )
        _log(
            f"Fold {fold.index} test "
            f"score={_selection_score(test_eval):.4f} "
            f"top5_same_d7={test_eval.top5_same_d7_sign_rate:.4f} "
            f"ndcg@5={test_eval.ndcg_at_5:.4f} "
            f"downstream_da={test_eval.downstream_da:.4f}"
        )

    scored_configs: list[tuple[HybridRerankWeights, float, float, float]] = []
    for key, values in config_validation_scores.items():
        if not values:
            continue
        weights = HybridRerankWeights(**json.loads(key))
        vals = np.asarray(values, dtype=np.float64)
        mean_score = float(vals.mean())
        variance_penalty = float(vals.std(ddof=0))
        objective = mean_score - args.variance_penalty * variance_penalty
        scored_configs.append((weights, mean_score, variance_penalty, objective))

    best_weights, mean_score, variance_penalty, best_objective = max(
        scored_configs,
        key=lambda item: item[3],
    )
    _log(
        "Global selection "
        f"weights={json.dumps(best_weights.as_dict(), sort_keys=True)} "
        f"mean_val={mean_score:.4f} std={variance_penalty:.4f} "
        f"objective={best_objective:.4f}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "data": args.data,
            "artifact": args.artifact,
            "weights": args.weights,
            "top_k": args.top_k,
            "candidate_pool_size": args.candidate_pool_size,
            "buy_threshold": args.buy_threshold,
            "sell_threshold": args.sell_threshold,
            "variance_penalty": args.variance_penalty,
            "max_folds": args.max_folds,
            "max_grid": args.max_grid,
            "grid_size": len(grid),
        },
        "summary": {
            "selected_weights": best_weights.as_dict(),
            "mean_validation_score": mean_score,
            "validation_score_std": variance_penalty,
            "selection_objective": best_objective,
        },
        "folds": [_fold_to_dict(fold) for fold in folds],
        "fold_results": fold_rows,
    }
    json_path = output_dir / "rolling_validation.json"
    md_path = output_dir / "rolling_validation.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        _markdown_summary(
            folds=folds,
            best_weights=best_weights,
            best_objective=best_objective,
            fold_rows=fold_rows,
        ),
        encoding="utf-8",
    )
    _log(f"Wrote artifacts json={json_path} markdown={md_path}")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
