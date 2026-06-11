from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stockmem.scripts.cem_dataset import LabeledRow, label_rows, matured_pool, mine_candidates
from stockmem.scripts.optimize_weights import load_rows, validate_rows
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
    k: int = 5
    batch_size: int = 16


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


def _hit_at_k(
    labeled: Sequence[LabeledRow],
    split: str,
    metric: LearnedDiagonalMetric,
    k: int,
) -> float:
    hits = 0
    total = 0
    for query in labeled:
        if query.split != split or query.direction == 0:
            continue
        pool = matured_pool(labeled, query)
        if not pool:
            continue
        ranked = sorted(pool, key=lambda row: metric.score(query.blocks, row.blocks), reverse=True)[:k]
        hits += int(any(row.direction == query.direction for row in ranked))
        total += 1
    return hits / total if total else 0.0


def train_one_seed(
    labeled: Sequence[LabeledRow],
    config: TrainConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    rng = np.random.default_rng(seed)
    block_dims = tuple(block.size for block in labeled[0].blocks)
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
    best_hit = -1.0
    best_d = diagonal.copy()
    best_s = scales.copy()
    stale = 0
    losses: list[float] = []
    train_anchors = [row for row in labeled if row.split == "train" and row.direction != 0]
    train_pool = [row for row in labeled if row.split == "train"]
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
                pair = mine_candidates(
                    anchor,
                    matured_pool(train_pool, anchor),
                    weights=DEFAULT_WEIGHTS,
                    hard_negs=config.hard_negs,
                    positive_count=config.positive_count,
                    flat_negs=config.flat_negs,
                    learned_score=lambda left, right: current_metric.score(
                        left.blocks,
                        right.blocks,
                    ),
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
        metric = LearnedDiagonalMetric(block_dims, diagonal, scales)
        val_hit = _hit_at_k(labeled, "val", metric, config.k)
        if val_hit > best_hit + 1e-9:
            best_hit = val_hit
            best_d = diagonal.copy()
            best_s = scales.copy()
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    return best_d, best_s, best_hit, losses


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
        _, _, val_hit, _ = train_one_seed(labeled, config, args.seed + trial.number)
        return val_hit

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
    val_hits: list[float] = []
    val_hits_at_5: list[float] = []
    for seed in range(args.seed, args.seed + max(1, args.seeds)):
        diagonal, scales, val_hit, losses = train_one_seed(labeled, config, seed)
        diagonals.append(diagonal)
        scales_list.append(scales)
        val_hits.append(val_hit)
        val_hits_at_5.append(
            _hit_at_k(
                labeled,
                "val",
                LearnedDiagonalMetric(block_dims, diagonal, scales),
                5,
            )
        )
        print(f"seed={seed} val_hit@{config.k}={val_hit:.4f} final_loss={losses[-1]:.6f}")

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
        "val_hit_at_5": float(np.mean(val_hits_at_5)),
        "val_hit_at_k": float(np.mean(val_hits)),
        "seed_std": float(np.std(val_hits_at_5)),
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
        f"wrote {output} val_hit={np.mean(val_hits):.4f} "
        f"seed_std_hit@5={np.std(val_hits_at_5):.4f}"
    )


if __name__ == "__main__":
    main()
