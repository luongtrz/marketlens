from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from stockmem.scripts.train_learned_retriever import info_nce_loss_and_grad
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


TRAIN_END = date(2024, 12, 24)
VAL_START = date(2025, 1, 1)
VAL_END = date(2025, 6, 23)
TEST_START = date(2025, 7, 1)
TEST_END = date(2026, 5, 1)
DEFAULT_KNN_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)


@dataclass(frozen=True)
class HeadConfig:
    name: str
    k: int
    buy_threshold: float
    sell_threshold: float
    return_weights: dict[str, float]


@dataclass(frozen=True)
class TrainRow:
    day: date
    split: str
    event_vec: np.ndarray
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray
    future_returns: dict[str, float]

    @property
    def blocks(self) -> tuple[np.ndarray, ...]:
        return (self.event_vec, self.factor_vec, self.indicator_vec, self.price_vec)


def _l2(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _split(day: date) -> str:
    if day <= TRAIN_END:
        return "train"
    if VAL_START <= day <= VAL_END:
        return "val"
    if TEST_START <= day <= TEST_END:
        return "test"
    return "embargo"


def _direction(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _load_rows(path: Path) -> list[TrainRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[TrainRow] = []
    for item in payload:
        if item.get("future_return_7d") is None:
            continue
        day = date.fromisoformat(str(item["date"]))
        factor = item.get("factor_vec") or []
        indicator = item.get("indicator_vec") or []
        price = item.get("price_vec") or []
        event = item.get("event_vec") or []
        if len(factor) != 75 or len(indicator) != 5 or len(price) != 60:
            continue
        if len(event) != 85:
            event = [0.0] * 85
        rows.append(
            TrainRow(
                day=day,
                split=_split(day),
                event_vec=_l2(event),
                factor_vec=_l2(factor),
                indicator_vec=_l2(indicator),
                price_vec=_l2(price),
                future_returns={
                    "1d": float(item.get("future_return_1d") or 0.0),
                    "3d": float(item.get("future_return_3d") or 0.0),
                    "7d": float(item.get("future_return_7d") or 0.0),
                    "15d": float(item.get("future_return_15d") or 0.0),
                    "30d": float(item.get("future_return_30d") or 0.0),
                },
            )
        )
    rows.sort(key=lambda row: row.day)
    return rows


def _load_head(path: Path) -> HeadConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    head = payload["head"]
    return HeadConfig(
        name=str(payload.get("name", path.stem)),
        k=int(head["k"]),
        buy_threshold=float(head["buy_threshold"]),
        sell_threshold=float(head["sell_threshold"]),
        return_weights={str(k): float(v) for k, v in head["return_weights"].items()},
    )


def _head_value(row: TrainRow, head: HeadConfig) -> float:
    total = 0.0
    total_w = 0.0
    for horizon, weight in head.return_weights.items():
        total += row.future_returns[horizon] * weight
        total_w += weight
    return total / max(total_w, 1e-12)


def _head_signal(rows: Sequence[TrainRow], head: HeadConfig) -> str:
    if not rows:
        return "HOLD"
    value = float(np.mean([_head_value(row, head) for row in rows[: head.k]]))
    if value > head.buy_threshold:
        return "BUY"
    if value < -head.sell_threshold:
        return "SELL"
    return "HOLD"


def _actual_signal(row: TrainRow, threshold: float) -> str:
    direction = _direction(row.future_returns["7d"], threshold)
    if direction > 0:
        return "BUY"
    if direction < 0:
        return "SELL"
    return "HOLD"


def _fixed_score(left: TrainRow, right: TrainRow) -> float:
    return (
        DEFAULT_KNN_WEIGHTS[0] * float(np.dot(left.factor_vec, right.factor_vec))
        + DEFAULT_KNN_WEIGHTS[1] * float(np.dot(left.indicator_vec, right.indicator_vec))
        + DEFAULT_KNN_WEIGHTS[2] * float(np.dot(left.price_vec, right.price_vec))
    )


def _support_score(anchor: TrainRow, candidate: TrainRow, head: HeadConfig, label_threshold: float) -> float:
    direction = _direction(anchor.future_returns["7d"], label_threshold)
    value = _head_value(candidate, head)
    if direction > 0:
        margin = value - head.buy_threshold
    elif direction < 0:
        margin = -value - head.sell_threshold
    else:
        margin = min(head.buy_threshold - value, value + head.sell_threshold)
    if margin <= 0:
        return 0.0
    same_d7 = _direction(candidate.future_returns["7d"], label_threshold) == direction
    surface = (_fixed_score(anchor, candidate) + 1.0) / 2.0
    return float(margin / (abs(margin) + 2.0) + 0.25 * surface + (0.25 if same_d7 else 0.0))


def _matured_pool(rows: Sequence[TrainRow], query: TrainRow) -> list[TrainRow]:
    return [
        row
        for row in rows
        if row.day < query.day and row.day + timedelta(days=7) <= query.day
    ]


def _mine(
    anchor: TrainRow,
    pool: Sequence[TrainRow],
    metric: LearnedDiagonalMetric,
    head: HeadConfig,
    *,
    label_threshold: float,
    positives: int,
    negatives: int,
) -> tuple[list[TrainRow], list[float], list[TrainRow]] | None:
    positive_scored = [
        (_support_score(anchor, candidate, head, label_threshold), candidate)
        for candidate in pool
    ]
    positive_scored = [(score, row) for score, row in positive_scored if score > 0.0]
    if not positive_scored:
        return None
    positive_scored.sort(key=lambda item: item[0], reverse=True)
    chosen_pos = [row for _, row in positive_scored[:positives]]
    pos_weights = [score for score, _ in positive_scored[:positives]]
    positive_ids = {id(row) for row in chosen_pos}
    anchor_direction = _direction(anchor.future_returns["7d"], label_threshold)

    negative_pool = [
        row
        for row in pool
        if id(row) not in positive_ids
        and _direction(row.future_returns["7d"], label_threshold) != anchor_direction
    ]
    if not negative_pool:
        return None
    hard = sorted(
        negative_pool,
        key=lambda row: metric.score(anchor.blocks, row.blocks),
        reverse=True,
    )[:negatives]
    return chosen_pos, pos_weights, hard


def _evaluate(rows: Sequence[TrainRow], metric: LearnedDiagonalMetric, head: HeadConfig, split: str, label_threshold: float) -> dict[str, float]:
    total = overall = active_total = active_correct = covered = hit = 0
    for query in rows:
        if query.split != split:
            continue
        pool = _matured_pool(rows, query)
        if not pool:
            continue
        ranked = sorted(pool, key=lambda row: metric.score(query.blocks, row.blocks), reverse=True)[: head.k]
        predicted = _head_signal(ranked, head)
        actual = _actual_signal(query, label_threshold)
        total += 1
        overall += int(predicted == actual)
        if predicted != "HOLD":
            covered += 1
            active_total += 1
            actual_return = query.future_returns["7d"]
            active_correct += int((predicted == "BUY" and actual_return > 0.0) or (predicted == "SELL" and actual_return < 0.0))
        hit += int(any(_actual_signal(candidate, label_threshold) == actual for candidate in ranked[:5]))
    return {
        "n": float(total),
        "overall_acc": overall / total if total else 0.0,
        "active_acc": active_correct / active_total if active_total else 0.0,
        "coverage": covered / total if total else 0.0,
        "hit_at_5_same_sign": hit / total if total else 0.0,
    }


def train_seed(
    rows: Sequence[TrainRow],
    init_metric: LearnedDiagonalMetric,
    head: HeadConfig,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    label_threshold: float,
    positives: int,
    negatives: int,
    history_writer: csv.DictWriter,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    block_dims = init_metric.block_dims
    diagonal = init_metric.diagonal.astype(np.float64).copy()
    scales = init_metric.block_scales.astype(np.float64).copy()
    m_d = np.zeros_like(diagonal)
    v_d = np.zeros_like(diagonal)
    m_s = np.zeros_like(scales)
    v_s = np.zeros_like(scales)
    step = 0
    best_score = -1.0
    best_d = diagonal.copy()
    best_s = scales.copy()
    best_eval: dict[str, float] = {}
    anchors = [
        row
        for row in rows
        if row.split == "train" and _direction(row.future_returns["7d"], label_threshold) != 0
    ]
    train_rows = [row for row in rows if row.split == "train"]

    for epoch in range(1, epochs + 1):
        metric = LearnedDiagonalMetric(block_dims, diagonal, scales)
        mined = []
        for anchor in anchors:
            pool = _matured_pool(train_rows, anchor)
            pair = _mine(
                anchor,
                pool,
                metric,
                head,
                label_threshold=label_threshold,
                positives=positives,
                negatives=negatives,
            )
            if pair is not None:
                mined.append((anchor, *pair))
        rng.shuffle(mined)
        losses: list[float] = []
        for anchor, pos_rows, pos_weights, neg_rows in mined:
            loss, grad_d, grad_s = info_nce_loss_and_grad(
                anchor,
                [*pos_rows, *neg_rows],
                diagonal,
                scales,
                temperature=0.08,
                teacher_temperature=0.1,
                ridge=0.01,
                positive_weights=pos_weights,
                block_dims=block_dims,
            )
            step += 1
            beta1, beta2 = 0.9, 0.999
            m_d = beta1 * m_d + (1.0 - beta1) * grad_d
            v_d = beta2 * v_d + (1.0 - beta2) * np.square(grad_d)
            m_s = beta1 * m_s + (1.0 - beta1) * grad_s
            v_s = beta2 * v_s + (1.0 - beta2) * np.square(grad_s)
            diagonal -= learning_rate * (m_d / (1.0 - beta1**step)) / (np.sqrt(v_d / (1.0 - beta2**step)) + 1e-8)
            scales -= learning_rate * (m_s / (1.0 - beta1**step)) / (np.sqrt(v_s / (1.0 - beta2**step)) + 1e-8)
            diagonal = np.clip(diagonal, 0.05, 20.0)
            scales = np.clip(scales, 1e-4, None)
            scales /= scales.sum()
            losses.append(loss)

        metric = LearnedDiagonalMetric(block_dims, diagonal, scales)
        val = _evaluate(rows, metric, head, "val", label_threshold)
        selection = 0.45 * val["overall_acc"] + 0.35 * val["active_acc"] + 0.20 * val["coverage"]
        if selection > best_score:
            best_score = selection
            best_d = diagonal.copy()
            best_s = scales.copy()
            best_eval = dict(val)
            best_eval["selection_score"] = selection
            best_eval["epoch"] = float(epoch)
        history_writer.writerow(
            {
                "seed": seed,
                "epoch": epoch,
                "loss": f"{float(np.mean(losses)) if losses else 0.0:.8f}",
                "mined_pairs": len(mined),
                "val_overall_acc": f"{val['overall_acc']:.8f}",
                "val_active_acc": f"{val['active_acc']:.8f}",
                "val_coverage": f"{val['coverage']:.8f}",
                "val_hit_at_5_same_sign": f"{val['hit_at_5_same_sign']:.8f}",
                "selection_score": f"{selection:.8f}",
                "best_selection_score": f"{best_score:.8f}",
            }
        )
        print(
            f"seed={seed} epoch={epoch} loss={float(np.mean(losses)) if losses else 0.0:.6f} "
            f"pairs={len(mined)} val_overall={val['overall_acc']:.4f} "
            f"val_active={val['active_acc']:.4f} val_coverage={val['coverage']:.4f} "
            f"val_hit@5={val['hit_at_5_same_sign']:.4f} best={best_score:.4f}",
            flush=True,
        )
    return best_d, best_s, best_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a retriever against the frozen learned decision head")
    parser.add_argument("--data", default="stockmem/data/real_optimizer_finbert.json")
    parser.add_argument("--init-artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--head", default="stockmem/config/knn_head.learned_finbert_rolling_stable.json")
    parser.add_argument("--output", default="stockmem/config/learned_retriever_head_aligned.json")
    parser.add_argument("--history-output", default="artifacts/train_logs/head_aligned_retriever_history.csv")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--label-threshold", type=float, default=2.0)
    parser.add_argument("--positives", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=10)
    args = parser.parse_args()

    rows = _load_rows(Path(args.data))
    if not rows:
        raise ValueError("No training rows loaded")
    init_metric = LearnedDiagonalMetric.load(args.init_artifact)
    head = _load_head(Path(args.head))
    history_path = Path(args.history_output)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "epoch",
                "loss",
                "mined_pairs",
                "val_overall_acc",
                "val_active_acc",
                "val_coverage",
                "val_hit_at_5_same_sign",
                "selection_score",
                "best_selection_score",
            ],
        )
        writer.writeheader()
        diagonals: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        val_results: list[dict[str, float]] = []
        for seed in range(args.seed, args.seed + args.seeds):
            d, s, val = train_seed(
                rows,
                init_metric,
                head,
                seed=seed,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                label_threshold=args.label_threshold,
                positives=args.positives,
                negatives=args.negatives,
                history_writer=writer,
            )
            handle.flush()
            diagonals.append(d)
            scales.append(s)
            val_results.append(val)

    averaged_d = np.mean(np.vstack(diagonals), axis=0)
    averaged_s = np.mean(np.vstack(scales), axis=0)
    averaged_s /= averaged_s.sum()
    artifact = {
        "version": "head_aligned_diagonal_v1",
        "type": "learned_diagonal",
        "dim": int(averaged_d.size),
        "block_dims": list(init_metric.block_dims),
        "d": averaged_d.tolist(),
        "block_scales": averaged_s.tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": args.data,
        "init_artifact": args.init_artifact,
        "frozen_head": args.head,
        "label_threshold": args.label_threshold,
        "objective": "rank matured candidates whose frozen learned-head weighted returns support the query realized D7 class",
        "hyperparameters": {
            "epochs": args.epochs,
            "seeds": args.seeds,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "positives": args.positives,
            "negatives": args.negatives,
            "temperature": 0.08,
            "teacher_temperature": 0.1,
            "ridge": 0.01,
        },
        "validation": {
            "per_seed": val_results,
            "mean": {
                key: float(np.mean([item.get(key, 0.0) for item in val_results]))
                for key in ("overall_acc", "active_acc", "coverage", "hit_at_5_same_sign", "selection_score")
            },
        },
        "history_output": str(history_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    final_metric = LearnedDiagonalMetric(init_metric.block_dims, averaged_d, averaged_s)
    val = _evaluate(rows, final_metric, head, "val", args.label_threshold)
    test = _evaluate(rows, final_metric, head, "test", args.label_threshold)
    print(
        f"wrote {output} scales={[round(float(x), 4) for x in averaged_s]} "
        f"val_overall={val['overall_acc']:.4f} val_active={val['active_acc']:.4f} "
        f"test_overall={test['overall_acc']:.4f} test_active={test['active_acc']:.4f} "
        f"test_coverage={test['coverage']:.4f} test_hit@5={test['hit_at_5_same_sign']:.4f} "
        f"history={history_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
