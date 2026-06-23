from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stockmem.scripts.cem_dataset import LabeledRow, label_rows, matured_pool, mine_candidates
from stockmem.scripts.optimize_weights import _compute_sharpe, load_rows, validate_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


BLOCK_DIMS = (75, 5, 60)
DEFAULT_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)


@dataclass(frozen=True)
class TrainConfig:
    temperature: float = 0.1
    teacher_temperature: float = 0.1
    ridge: float = 0.01
    hard_negs: int = 8
    positive_count: int = 3
    flat_negs: int = 2
    learning_rate: float = 0.01
    epochs: int = 60
    patience: int = 10
    remine_every: int = 3
    eval_every: int = 5
    k: int = 5
    batch_size: int = 16
    buy_threshold: float = 2.0
    sell_threshold: float = 2.0


def _score_and_grad(
    anchor: LabeledRow,
    candidate: LabeledRow,
    diagonal: np.ndarray,
    scales: np.ndarray,
    block_dims: Sequence[int] = BLOCK_DIMS,
) -> tuple[float, np.ndarray, np.ndarray]:
    if len(anchor.blocks) != len(block_dims) or len(candidate.blocks) != len(block_dims):
        raise ValueError("Training blocks do not match block_dims")
    grad_d = np.zeros_like(diagonal)
    grad_s = np.zeros_like(scales)
    score = 0.0
    offset = 0
    for block_index, (dim, x, y) in enumerate(
        zip(block_dims, anchor.blocks, candidate.blocks)
    ):
        block_d = diagonal[offset : offset + dim]
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        ux = block_d * x64
        uy = block_d * y64
        nx = float(np.linalg.norm(ux))
        ny = float(np.linalg.norm(uy))
        if nx > 1e-12 and ny > 1e-12:
            zx = ux / nx
            zy = uy / ny
            cosine = float(np.dot(zx, zy))
            grad_ux = (zy - cosine * zx) / nx
            grad_uy = (zx - cosine * zy) / ny
            grad_d[offset : offset + dim] = scales[block_index] * (
                x64 * grad_ux + y64 * grad_uy
            )
        else:
            cosine = 0.0
        score += float(scales[block_index]) * cosine
        grad_s[block_index] = cosine
        offset += dim
    return score, grad_d, grad_s


def info_nce_loss_and_grad(
    anchor: LabeledRow,
    candidates: Sequence[LabeledRow],
    diagonal: np.ndarray,
    scales: np.ndarray,
    *,
    temperature: float,
    ridge: float,
    teacher_temperature: float = 0.1,
    positive_weights: Sequence[float] | None = None,
    block_dims: Sequence[int] = BLOCK_DIMS,
) -> tuple[float, np.ndarray, np.ndarray]:
    scores: list[float] = []
    d_grads: list[np.ndarray] = []
    s_grads: list[np.ndarray] = []
    for candidate in candidates:
        score, grad_d, grad_s = _score_and_grad(
            anchor,
            candidate,
            diagonal,
            scales,
            block_dims,
        )
        scores.append(score)
        d_grads.append(grad_d)
        s_grads.append(grad_s)

    logits = np.asarray(scores, dtype=np.float64) / temperature
    logits -= float(logits.max())
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum()
    targets = np.zeros_like(probabilities)
    if positive_weights is None:
        targets[0] = 1.0
    else:
        rewards = np.asarray(positive_weights, dtype=np.float64)
        if rewards.size == 0 or rewards.size > probabilities.size:
            raise ValueError("positive_weights must identify at least one candidate")
        teacher_logits = rewards / teacher_temperature
        teacher_logits -= float(teacher_logits.max())
        teacher_probabilities = np.exp(teacher_logits)
        targets[: rewards.size] = teacher_probabilities / teacher_probabilities.sum()
    loss = -float(np.sum(targets * np.log(np.maximum(probabilities, 1e-12))))
    coefficients = probabilities - targets
    grad_d = sum(
        coefficient * candidate_grad
        for coefficient, candidate_grad in zip(coefficients, d_grads)
    ) / temperature
    grad_s = sum(
        coefficient * candidate_grad
        for coefficient, candidate_grad in zip(coefficients, s_grads)
    ) / temperature
    loss += ridge * float(np.mean(np.square(diagonal - 1.0)))
    grad_d += (2.0 * ridge / diagonal.size) * (diagonal - 1.0)
    return loss, grad_d, grad_s


def _prestack_blocks(rows: Sequence[LabeledRow]) -> list[np.ndarray]:
    """Pre-stack all row blocks into matrices for vectorized scoring."""
    if not rows:
        return []
    n_blocks = len(rows[0].blocks)
    return [
        np.stack([r.blocks[b] for r in rows]).astype(np.float64)
        for b in range(n_blocks)
    ]


def _val_combined_fast(
    labeled_list: list[LabeledRow],
    all_stacked: list[np.ndarray],
    dates_ord: np.ndarray,
    metric: LearnedDiagonalMetric,
    k: int,
    buy_threshold: float = 2.0,
    sell_threshold: float = 2.0,
) -> float:
    """Vectorized combined score (0.6·DA + 0.4·Sharpe) on val split."""
    correct: list[bool] = []
    strategy_returns: list[float] = []
    for i, query in enumerate(labeled_list):
        if query.split != "val" or query.direction == 0:
            continue
        qord = int(dates_ord[i])
        mask = dates_ord <= qord - 7
        pool_idx = np.where(mask)[0]
        if len(pool_idx) == 0:
            continue
        pool_stacked = [s[pool_idx] for s in all_stacked]
        scores = metric.score_batch(query.blocks, pool_stacked)
        k_eff = min(k, len(pool_idx))
        top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
        pred_ret = float(np.mean([labeled_list[pool_idx[j]].row.future_return_7d for j in top_idx]))
        if pred_ret > buy_threshold:
            signal = "BUY"
        elif pred_ret < -sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        actual = query.row.future_return_7d
        is_correct = (
            (signal == "BUY" and actual > 0)
            or (signal == "SELL" and actual < 0)
            or (signal == "HOLD" and -sell_threshold <= actual <= buy_threshold)
        )
        correct.append(is_correct)
        if signal == "BUY":
            strategy_returns.append(actual)
        elif signal == "SELL":
            strategy_returns.append(-actual)
        else:
            strategy_returns.append(0.0)
    if not correct:
        return 0.0
    da = float(np.mean(correct))
    sharpe = _compute_sharpe(strategy_returns, horizon="7d", mode="nonoverlap")
    return 0.6 * da + 0.4 * min(max(sharpe, -2.0), 2.0) / 2.0


def _hit_at_k_fast(
    labeled_list: list[LabeledRow],
    all_stacked: list[np.ndarray],
    dates_ord: np.ndarray,
    metric: LearnedDiagonalMetric,
    k: int,
    split: str = "val",
) -> float:
    """Vectorized hit@k evaluation."""
    hits = total = 0
    for i, query in enumerate(labeled_list):
        if query.split != split or query.direction == 0:
            continue
        qord = int(dates_ord[i])
        mask = dates_ord <= qord - 7
        pool_idx = np.where(mask)[0]
        if len(pool_idx) == 0:
            continue
        pool_stacked = [s[pool_idx] for s in all_stacked]
        scores = metric.score_batch(query.blocks, pool_stacked)
        k_eff = min(k, len(pool_idx))
        top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
        hits += int(any(labeled_list[pool_idx[j]].direction == query.direction for j in top_idx))
        total += 1
    return hits / total if total else 0.0


def train_one_seed(
    labeled: Sequence[LabeledRow],
    config: TrainConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float, list[float]]:
    """Returns (best_d, best_s, best_combined, best_hit_at_k, losses)."""
    rng = np.random.default_rng(seed)
    labeled_list = list(labeled)
    block_dims = tuple(block.size for block in labeled_list[0].blocks)
    diagonal = np.ones(sum(block_dims), dtype=np.float64)
    if len(block_dims) == 4:
        scales = np.asarray(
            [0.05, *(0.95 * np.asarray(DEFAULT_WEIGHTS, dtype=np.float64))],
            dtype=np.float64,
        )
    else:
        scales = np.asarray(DEFAULT_WEIGHTS, dtype=np.float64)
    m_d = np.zeros_like(diagonal)
    v_d = np.zeros_like(diagonal)
    m_s = np.zeros_like(scales)
    v_s = np.zeros_like(scales)
    step = 0
    best_combined = -999.0
    best_d = diagonal.copy()
    best_s = scales.copy()
    stale = 0
    losses: list[float] = []

    # Pre-stack all blocks once for fast vectorized eval
    all_stacked = _prestack_blocks(labeled_list)
    dates_ord = np.array([r.parsed_date.toordinal() for r in labeled_list])

    train_anchors = [row for row in labeled_list if row.split == "train" and row.direction != 0]
    train_pool = [row for row in labeled_list if row.split == "train"]

    # Pre-compute pairwise baseline matrix once (vectorized, O(N²) numpy vs O(N²) Python)
    tp_id_to_idx = {id(r): i for i, r in enumerate(train_pool)}
    tp_dates_ord = np.array([r.parsed_date.toordinal() for r in train_pool])
    _f = np.stack([r.row.factor_vec for r in train_pool]).astype(np.float64)
    _i = np.stack([r.row.indicator_vec for r in train_pool]).astype(np.float64)
    _p = np.stack([r.row.price_vec for r in train_pool]).astype(np.float64)
    w1, w2, w3 = DEFAULT_WEIGHTS
    baseline_full = w1 * (_f @ _f.T) + w2 * (_i @ _i.T) + w3 * (_p @ _p.T)  # (N, N)
    del _f, _i, _p

    mined: list[
        tuple[
            LabeledRow,
            tuple[LabeledRow, ...],
            tuple[float, ...],
            tuple[LabeledRow, ...],
        ]
    ] = []

    for epoch in range(config.epochs):
        if epoch % config.remine_every == 0:
            mined = []
            current_metric = LearnedDiagonalMetric(block_dims, diagonal, scales)
            for anchor in train_anchors:
                ai = tp_id_to_idx[id(anchor)]
                ad = int(tp_dates_ord[ai])
                pool_mask = tp_dates_ord <= ad - 7
                pool_idx_arr = np.where(pool_mask)[0]
                if len(pool_idx_arr) == 0:
                    continue
                pool_rows = [train_pool[pi] for pi in pool_idx_arr]
                scores_vec = baseline_full[ai, pool_idx_arr]
                precomputed = {id(train_pool[pi]): float(scores_vec[j])
                               for j, pi in enumerate(pool_idx_arr)}
                pair = mine_candidates(
                    anchor,
                    pool_rows,
                    weights=DEFAULT_WEIGHTS,
                    hard_negs=config.hard_negs,
                    positive_count=config.positive_count,
                    flat_negs=config.flat_negs,
                    learned_score=lambda left, right: current_metric.score(
                        left.blocks,
                        right.blocks,
                    ),
                    baseline_scores_override=precomputed,
                )
                if pair is not None:
                    mined.append(
                        (
                            anchor,
                            pair.positives,
                            pair.positive_weights,
                            pair.negatives,
                        )
                    )
        rng.shuffle(mined)
        epoch_losses: list[float] = []
        for pair_index, (anchor, positives, positive_weights, negatives) in enumerate(mined):
            batch_start = (pair_index // config.batch_size) * config.batch_size
            batch = mined[batch_start : batch_start + config.batch_size]
            in_batch_negatives = [
                other_positive
                for other_anchor, other_positives, _, _ in batch
                if other_anchor is not anchor
                for other_positive in other_positives
                if other_positive.direction == -anchor.direction
            ]
            loss, grad_d, grad_s = info_nce_loss_and_grad(
                anchor,
                [*positives, *negatives, *in_batch_negatives],
                diagonal,
                scales,
                temperature=config.temperature,
                teacher_temperature=config.teacher_temperature,
                ridge=config.ridge,
                positive_weights=positive_weights,
                block_dims=block_dims,
            )
            step += 1
            beta1, beta2 = 0.9, 0.999
            m_d = beta1 * m_d + (1.0 - beta1) * grad_d
            v_d = beta2 * v_d + (1.0 - beta2) * np.square(grad_d)
            m_s = beta1 * m_s + (1.0 - beta1) * grad_s
            v_s = beta2 * v_s + (1.0 - beta2) * np.square(grad_s)
            m_d_hat = m_d / (1.0 - beta1**step)
            v_d_hat = v_d / (1.0 - beta2**step)
            m_s_hat = m_s / (1.0 - beta1**step)
            v_s_hat = v_s / (1.0 - beta2**step)
            diagonal -= config.learning_rate * m_d_hat / (np.sqrt(v_d_hat) + 1e-8)
            scales -= config.learning_rate * m_s_hat / (np.sqrt(v_s_hat) + 1e-8)
            diagonal = np.clip(diagonal, 0.05, 20.0)
            scales = np.clip(scales, 1e-4, None)
            scales /= scales.sum()
            epoch_losses.append(loss)
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        if (epoch + 1) % config.eval_every == 0 or epoch == config.epochs - 1:
            metric = LearnedDiagonalMetric(block_dims, diagonal, scales)
            val_hit = _hit_at_k_fast(labeled_list, all_stacked, dates_ord, metric, config.k)
            if val_hit > best_combined + 1e-9:
                best_combined = val_hit
                best_d = diagonal.copy()
                best_s = scales.copy()
                stale = 0
            else:
                stale += 1
            if stale >= config.patience:
                break

    best_metric = LearnedDiagonalMetric(block_dims, best_d, best_s)
    best_hit = _hit_at_k_fast(labeled_list, all_stacked, dates_ord, best_metric, config.k)
    return best_d, best_s, best_combined, best_hit, losses


def _require_optuna() -> Any:
    if optuna is None:
        raise RuntimeError("Optuna is required; install the stockmem optional dependencies")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-clean learned diagonal retriever")
    parser.add_argument("--data", default="stockmem/data/real_optimizer.json")
    parser.add_argument("--output", default="stockmem/config/learned_retriever.json")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", default="7d", choices=["7d"])
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    validate_rows(rows)
    o = _require_optuna()

    def objective(trial: Any) -> float:
        band = trial.suggest_categorical("band", ["0.5sigma", "fixed"])
        labeled = label_rows(rows, band=band)
        config = TrainConfig(
            temperature=trial.suggest_float("temperature", 0.04, 0.3, log=True),
            teacher_temperature=trial.suggest_categorical(
                "teacher_temperature",
                [0.05, 0.1, 0.2],
            ),
            ridge=trial.suggest_float("ridge", 1e-4, 0.2, log=True),
            hard_negs=trial.suggest_categorical("hard_negs", [4, 8, 12]),
            positive_count=trial.suggest_categorical("positive_count", [1, 3, 5]),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 3e-2, log=True),
            epochs=args.epochs,
            k=trial.suggest_categorical("k", [3, 5, 7]),
        )
        _, _, val_combined, _, _ = train_one_seed(labeled, config, args.seed + trial.number)
        return val_combined

    study = o.create_study(
        direction="maximize",
        sampler=o.samplers.TPESampler(seed=args.seed),
        study_name="stockmem_learned_retriever",
    )
    study.optimize(objective, n_trials=max(1, args.trials), show_progress_bar=False)
    params = study.best_trial.params
    band = str(params["band"])
    labeled = label_rows(rows, band=band)
    block_dims = tuple(block.size for block in labeled[0].blocks)
    config = TrainConfig(
        temperature=float(params["temperature"]),
        teacher_temperature=float(params["teacher_temperature"]),
        ridge=float(params["ridge"]),
        hard_negs=int(params["hard_negs"]),
        positive_count=int(params["positive_count"]),
        learning_rate=float(params["learning_rate"]),
        epochs=args.epochs,
        k=int(params["k"]),
    )

    diagonals: list[np.ndarray] = []
    scales_list: list[np.ndarray] = []
    val_combineds: list[float] = []
    val_hits_at_k: list[float] = []
    for seed in range(args.seed, args.seed + max(1, args.seeds)):
        diagonal, scales, val_combined, val_hit, losses = train_one_seed(labeled, config, seed)
        diagonals.append(diagonal)
        scales_list.append(scales)
        val_combineds.append(val_combined)
        val_hits_at_k.append(val_hit)
        print(f"seed={seed} val_combined={val_combined:.4f} val_hit@{config.k}={val_hit:.4f} final_loss={losses[-1]:.6f}")

    averaged_d = np.mean(np.vstack(diagonals), axis=0)
    averaged_scales = np.mean(np.vstack(scales_list), axis=0)
    averaged_scales /= averaged_scales.sum()
    payload = {
        "version": "learned_cem_v2",
        "type": "learned_diagonal",
        "dim": int(averaged_d.size),
        "block_dims": list(block_dims),
        "d": averaged_d.tolist(),
        "block_scales": averaged_scales.tolist(),
        "band": band,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "splits": {
            "train": ["2023-01-01", "2024-12-24"],
            "val": ["2025-01-01", "2025-06-23"],
            "test": ["2025-07-01", "2026-05-01"],
            "embargo_days": 7,
        },
        "hyperparameters": {
            "temperature": config.temperature,
            "teacher_temperature": config.teacher_temperature,
            "ridge": config.ridge,
            "hard_negs": config.hard_negs,
            "positive_count": config.positive_count,
            "flat_negs": config.flat_negs,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "k": config.k,
        },
        "val_combined": float(np.mean(val_combineds)),
        "val_hit_at_k": float(np.mean(val_hits_at_k)),
        "val_hit_at_5": float(np.mean(val_hits_at_k)),
        "seed_std": float(np.std(val_hits_at_k)),
        "seeds": list(range(args.seed, args.seed + max(1, args.seeds))),
        "source": args.data,
        "mining_protocol": {
            "version": "outcome_regime_distillation_v1",
            "teacher_relevance": {
                "outcome_magnitude_similarity": 0.45,
                "causal_volatility_similarity": 0.35,
                "fixed_knn_surface_similarity": 0.20,
            },
            "hard_negative_shortlist": "top_4x_fixed_knn_then_current_metric",
            "flat_candidates_retained": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {output} val_combined={np.mean(val_combineds):.4f} "
        f"val_hit@k={np.mean(val_hits_at_k):.4f} "
        f"seed_std={np.std(val_hits_at_k):.4f}"
    )


if __name__ == "__main__":
    main()
