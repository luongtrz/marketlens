"""Freeze per-day test-set predictions to artifacts/predictions/.

Reads real_optimizer_v3.json and writes three JSONL files:
  artifacts/predictions/fixed_knn_test.jsonl       -- fixed kNN + simple avg 7d
  artifacts/predictions/knn_returns_test.jsonl     -- fixed kNN + multi-horizon returns
  artifacts/predictions/cem_rag_test.jsonl         -- learned diagonal + tau=0.22

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/freeze_predictions.py
    PYTHONPATH=/home/luong/marketlens python scripts/freeze_predictions.py --data stockmem/data/real_optimizer_v3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stockmem.scripts.cem_dataset import TEST_START, assign_split, label_rows, matured_pool
from stockmem.scripts.evaluate_retriever import DEFAULT_WEIGHTS, _fixed_score
from stockmem.scripts.optimize_weights import load_rows
from stockmem.src.search.learned_metric import load_learned_metric

# ── Constants ─────────────────────────────────────────────────────────────────
SYMBOL = "BTC"
K = 5
CEM_TAU = 0.22
CEM_HORIZON = "7d"
KNN_THRESHOLDS = {"buy": 2.0, "sell": -2.0}
KNN_WEIGHTS = {"1d": 0.15, "3d": 0.25, "7d": 0.35, "15d": 0.15, "30d": 0.10}
HORIZONS = ["1d", "3d", "7d", "15d", "30d"]

OUT_DIR = PROJECT_ROOT / "artifacts" / "predictions"


def _learned_score(query, candidate, metric) -> float:
    q_blocks = [
        np.asarray(query.row.event_vec, dtype=np.float64),
        np.asarray(query.row.factor_vec, dtype=np.float64),
        np.asarray(query.row.indicator_vec, dtype=np.float64),
        np.asarray(query.row.price_vec, dtype=np.float64),
    ]
    c_blocks = [
        np.asarray(candidate.row.event_vec, dtype=np.float64),
        np.asarray(candidate.row.factor_vec, dtype=np.float64),
        np.asarray(candidate.row.indicator_vec, dtype=np.float64),
        np.asarray(candidate.row.price_vec, dtype=np.float64),
    ]
    return float(metric.score(q_blocks, c_blocks))


def _knn_returns_signal(
    neighbors,
    buy_threshold: float = 2.0,
    sell_threshold: float = -2.0,
    weights: dict[str, float] = KNN_WEIGHTS,
) -> tuple[str, float, dict[str, float]]:
    per_rec = []
    for row in neighbors:
        total_w = total_v = 0.0
        for h, w in weights.items():
            val = getattr(row.row, f"future_return_{h}", None)
            if val is not None:
                total_v += val * w
                total_w += w
        if total_w > 0:
            per_rec.append(total_v / total_w)
    if not per_rec:
        return "HOLD", 0.50, {}
    avg = float(np.mean(per_rec))
    if avg > buy_threshold:
        signal = "BUY"
        conf = min(0.55 + min((avg - buy_threshold) / 15.0, 0.35), 0.95)
    elif avg < sell_threshold:
        signal = "SELL"
        conf = min(0.55 + min((sell_threshold - avg) / 15.0, 0.35), 0.95)
    else:
        signal = "HOLD"
        conf = 0.50
    return signal, round(conf, 3), {"weighted_avg_return": round(avg, 4)}


def _cem_rag_signal(
    neighbors, similarity_scores: list[float], tau: float = 0.22, horizon: str = "7d"
) -> tuple[str, float, float, float, float]:
    attr = f"future_return_{horizon}"
    usable = [(sim, nb) for sim, nb in zip(similarity_scores, neighbors)
              if getattr(nb.row, attr, None) is not None]
    if not usable:
        return "HOLD", 1/3, 1/3, 1/3, 0.50
    weights = [(sim + 1) / 2 for sim, _ in usable]
    total_w = sum(weights) + 1e-12
    p_up   = sum(w for w, (_, nb) in zip(weights, usable) if getattr(nb.row, attr) > 0) / total_w
    p_down = sum(w for w, (_, nb) in zip(weights, usable) if getattr(nb.row, attr) < 0) / total_w
    p_hold = max(0.0, 1.0 - p_up - p_down)
    d_up, d_down = p_up - p_down, p_down - p_up
    if d_up >= tau:
        signal = "BUY"; conf = min(0.50 + d_up * 0.80, 0.95)
    elif d_down >= tau:
        signal = "SELL"; conf = min(0.50 + d_down * 0.80, 0.95)
    else:
        signal = "HOLD"; conf = 0.50 + max(d_up, d_down) * 0.30
    return signal, round(p_up, 4), round(p_down, 4), round(p_hold, 4), round(conf, 3)


def run(data_path: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(data_path))
    labeled = label_rows(rows)

    # Load learned metric (225d = event85+factor75+indicator5+price60)
    retriever_path = PROJECT_ROOT / "stockmem" / "config" / "learned_retriever.json"
    learned = load_learned_metric(str(retriever_path))
    if learned is None:
        print(f"WARNING: learned retriever not found at {retriever_path}, skipping cem_rag output")

    fixed_knn_rows: list[dict] = []
    knn_returns_rows: list[dict] = []
    cem_rag_rows: list[dict] = []

    test_queries = [lb for lb in labeled if lb.split == "test"]
    print(f"Test rows: {len(test_queries)}")

    for query in test_queries:
        pool = matured_pool(labeled, query, guard=True)
        if not pool:
            continue

        # ── Fixed kNN scoring ────────────────────────────────────────────────
        scored_fixed = sorted(
            [(float(np.dot(query.row.factor_vec, c.row.factor_vec) * DEFAULT_WEIGHTS[0]
                   + np.dot(query.row.indicator_vec, c.row.indicator_vec) * DEFAULT_WEIGHTS[1]
                   + np.dot(query.row.price_vec, c.row.price_vec) * DEFAULT_WEIGHTS[2]), c)
             for c in pool],
            key=lambda x: x[0], reverse=True,
        )[:K]
        fixed_neighbors = [c for _, c in scored_fixed]
        fixed_sims = [s for s, _ in scored_fixed]

        ret_7d = query.row.future_return_7d
        actual_returns = {h: getattr(query.row, f"future_return_{h}", None) for h in HORIZONS}

        # fixed_knn: simple avg 7d return of neighbors
        avg_7d = float(np.mean([c.row.future_return_7d for c in fixed_neighbors if c.row.future_return_7d is not None] or [0.0]))
        fk_signal = "BUY" if avg_7d > 2.0 else ("SELL" if avg_7d < -2.0 else "HOLD")
        fixed_knn_rows.append({
            "date": query.row.date,
            "symbol": SYMBOL,
            "split": query.split,
            "signal": fk_signal,
            "confidence": round(min(0.55 + abs(avg_7d) / 15.0, 0.95), 3),
            "avg_7d_return_neighbors": round(avg_7d, 4),
            "retrieval_count": len(fixed_neighbors),
            **{f"actual_return_{h}": actual_returns[h] for h in HORIZONS},
        })

        # knn_returns: multi-horizon weighted signal
        kr_signal, kr_conf, kr_meta = _knn_returns_signal(fixed_neighbors)
        knn_returns_rows.append({
            "date": query.row.date,
            "symbol": SYMBOL,
            "split": query.split,
            "signal": kr_signal,
            "confidence": kr_conf,
            **kr_meta,
            "retrieval_count": len(fixed_neighbors),
            **{f"actual_return_{h}": actual_returns[h] for h in HORIZONS},
        })

        # cem_rag: learned retriever + p_up/p_down/p_hold
        if learned is not None:
            scored_learned = sorted(
                [(_learned_score(query, c, learned), c) for c in pool],
                key=lambda x: x[0], reverse=True,
            )[:K]
            learned_neighbors = [c for _, c in scored_learned]
            learned_sims = [s for s, _ in scored_learned]
            cem_signal, p_up, p_down, p_hold, cem_conf = _cem_rag_signal(
                learned_neighbors, learned_sims, tau=CEM_TAU, horizon=CEM_HORIZON
            )
            cem_rag_rows.append({
                "date": query.row.date,
                "symbol": SYMBOL,
                "split": query.split,
                "horizon": CEM_HORIZON,
                "tau": CEM_TAU,
                "signal": cem_signal,
                "confidence": cem_conf,
                "p_up": p_up,
                "p_down": p_down,
                "p_hold": p_hold,
                "retrieval_count": len(learned_neighbors),
                **{f"actual_return_{h}": actual_returns[h] for h in HORIZONS},
            })

    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {len(rows)} rows → {path}")

    _write_jsonl(OUT_DIR / "fixed_knn_test.jsonl", fixed_knn_rows)
    _write_jsonl(OUT_DIR / "knn_returns_test.jsonl", knn_returns_rows)
    if cem_rag_rows:
        _write_jsonl(OUT_DIR / "cem_rag_test.jsonl", cem_rag_rows)

    # Quick accuracy summary
    for name, rows_list in [("fixed_knn", fixed_knn_rows), ("knn_returns", knn_returns_rows), ("cem_rag", cem_rag_rows)]:
        if not rows_list:
            continue
        correct = [
            (r["signal"] == "BUY" and (r.get("actual_return_7d") or 0) > 0)
            or (r["signal"] == "SELL" and (r.get("actual_return_7d") or 0) < 0)
            or (r["signal"] == "HOLD" and abs(r.get("actual_return_7d") or 0) < 2.0)
            for r in rows_list
        ]
        buy_n = sum(1 for r in rows_list if r["signal"] == "BUY")
        sell_n = sum(1 for r in rows_list if r["signal"] == "SELL")
        print(f"  [{name}] n={len(rows_list)}  DA={np.mean(correct):.3f}  BUY={buy_n}  SELL={sell_n}  HOLD={len(rows_list)-buy_n-sell_n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="stockmem/data/real_optimizer_v3.json")
    args = parser.parse_args()
    run(args.data)
