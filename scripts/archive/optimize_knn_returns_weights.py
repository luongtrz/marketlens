"""Bayesian optimization of kNN search weights for the kNN-returns signal.

Objective: maximize Directional Accuracy (DA) on BUY/SELL signals only,
using the exact same logic as _knn_returns_signal() in steps.py:
  - Multi-horizon weighted avg: 1d·0.40 + 3d·0.30 + 7d·0.15 + 15d·0.10 + 30d·0.05
  - Threshold ±2%: BUY / SELL / HOLD
  - DA measured on BUY+SELL days only (HOLD excluded)
  - Coverage penalty: objective penalizes if coverage < min_coverage

Walk-forward (no look-ahead): each query only sees records before its date.

Precomputes pairwise component cosine similarities once → each Optuna trial
is just a weighted combination, making 100+ trials feasible in minutes.

Usage:
    python scripts/optimize_knn_returns_weights.py
    python scripts/optimize_knn_returns_weights.py --trials 200 --warmup 300
    python scripts/optimize_knn_returns_weights.py --buy-thr 2 --sell-thr 2 --k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None  # type: ignore

DB_URL     = "postgresql://postgres:pass@localhost:5432/postgres"
OUT_FILE   = Path(__file__).parent.parent / "stockmem" / "config" / "weights.auto.json"
RETURN_W   = {"1d": 0.40, "3d": 0.30, "7d": 0.15, "15d": 0.10, "30d": 0.05}
HORIZONS   = list(RETURN_W.keys())


# ── Data loading ─────────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _indicator_vec(ms: dict, sentiment: float) -> np.ndarray:
    raw = np.array([
        float(ms.get("msi") or 0.0),
        float(ms.get("rsi") or 50.0),
        sentiment,
        float(ms.get("fear_greed_index") or 50.0),
        float(ms.get("price_change_pct") or 0.0),
    ], dtype=np.float32)
    mean, std = raw.mean(), raw.std() + 1e-8
    return _l2((raw - mean) / std)


def _price_vec(candles: list) -> np.ndarray:
    if len(candles) < 2:
        return np.zeros(60, dtype=np.float32)
    closes  = np.array([float(c.get("close",  0)) for c in candles], np.float32)
    highs   = np.array([float(c.get("high",   0)) for c in candles], np.float32)
    lows    = np.array([float(c.get("low",    0)) for c in candles], np.float32)
    volumes = np.array([float(c.get("volume", 0)) for c in candles], np.float32)
    rets   = np.diff(closes)  / (closes[:-1]  + 1e-8)
    ranges = (highs - lows)   / (closes       + 1e-8)
    vchs   = np.diff(volumes) / (volumes[:-1] + 1e-8)

    def tail20(a: np.ndarray) -> np.ndarray:
        t = a[-20:]
        return np.pad(t, (20 - len(t), 0)).astype(np.float32)

    return _l2(np.concatenate([tail20(rets), tail20(ranges[1:]), tail20(vchs)]))


async def load_data() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dates, factor_mat, indicator_mat, price_mat, returns_mat).

    returns_mat shape: (N, 5) — columns: 1d 3d 7d 15d 30d (NaN if missing).
    """
    conn = await asyncpg.connect(DB_URL)
    rows = await conn.fetch(
        "SELECT record_date, payload FROM stockmem_records "
        "WHERE symbol='BTC' ORDER BY record_date"
    )
    await conn.close()

    dates, fvecs, ivecs, pvecs, rets = [], [], [], [], []
    skipped = 0
    for row in rows:
        p  = json.loads(row["payload"])
        ms = p.get("market_snapshot", {})

        fv_raw = p.get("factor_vector", [])
        if not fv_raw or len(fv_raw) != 75:
            skipped += 1
            continue

        fv = _l2(np.array(fv_raw, dtype=np.float32))
        iv = _indicator_vec(ms, float(p.get("sentiment_score", 0.0)))
        pv = _price_vec(ms.get("recent_candles") or ms.get("candles") or [])

        ret_row = np.array([
            p.get("future_return_1d")  if p.get("future_return_1d")  is not None else np.nan,
            p.get("future_return_3d")  if p.get("future_return_3d")  is not None else np.nan,
            p.get("future_return_7d")  if p.get("future_return_7d")  is not None else np.nan,
            p.get("future_return_15d") if p.get("future_return_15d") is not None else np.nan,
            p.get("future_return_30d") if p.get("future_return_30d") is not None else np.nan,
        ], dtype=np.float64)

        dates.append(str(row["record_date"]))
        fvecs.append(fv)
        ivecs.append(iv)
        pvecs.append(pv)
        rets.append(ret_row)

    if skipped:
        print(f"  Skipped {skipped} records with missing factor_vector")

    return (
        dates,
        np.stack(fvecs),       # (N, 75)
        np.stack(ivecs),       # (N, 5)
        np.stack(pvecs),       # (N, 60)
        np.stack(rets),        # (N, 5)
    )


# ── Precompute pairwise similarities ─────────────────────────────────────────

def precompute_sims(
    F: np.ndarray, I: np.ndarray, P: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dot products of L2-normalized vecs = cosine similarity. Shape (N, N)."""
    print("  Precomputing pairwise cosine similarities...", flush=True)
    F_sim = (F @ F.T).astype(np.float32)
    I_sim = (I @ I.T).astype(np.float32)
    P_sim = (P @ P.T).astype(np.float32)
    print(f"  Done. F_sim {F_sim.shape}, I_sim {I_sim.shape}, P_sim {P_sim.shape}")
    return F_sim, I_sim, P_sim


# ── kNN-returns signal ────────────────────────────────────────────────────────

_RETURN_WEIGHTS = np.array([RETURN_W[h] for h in HORIZONS], dtype=np.float64)


def _knn_avg(neighbor_rets: np.ndarray) -> float | None:
    """Weighted avg of available horizons for a single neighbor. Returns None if all NaN."""
    mask = ~np.isnan(neighbor_rets)
    if not mask.any():
        return None
    w = _RETURN_WEIGHTS[mask]
    v = neighbor_rets[mask]
    return float((v * w).sum() / w.sum())


def signal_from_neighbors(
    top_k_rets: np.ndarray,   # (k, 5) — neighbor return rows
    buy_thr: float,
    sell_thr: float,
) -> str:
    avgs = [a for i in range(len(top_k_rets)) if (a := _knn_avg(top_k_rets[i])) is not None]
    if not avgs:
        return "HOLD"
    overall = sum(avgs) / len(avgs)
    if overall > buy_thr:
        return "BUY"
    if overall < -sell_thr:
        return "SELL"
    return "HOLD"


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def walk_forward(
    F_sim: np.ndarray,
    I_sim: np.ndarray,
    P_sim: np.ndarray,
    returns: np.ndarray,   # (N, 5)
    w1: float,
    w2: float,
    w3: float,
    k: int,
    warmup: int,
    buy_thr: float,
    sell_thr: float,
    eval_horizon_idx: int = 2,  # index into HORIZONS, default 7d
) -> dict:
    N = len(returns)
    correct = total = buy_n = sell_n = 0

    for i in range(warmup, N):
        # Weighted similarity to all prior records
        sims_row = w1 * F_sim[i, :i] + w2 * I_sim[i, :i] + w3 * P_sim[i, :i]

        # Top-k indices from prior pool
        if len(sims_row) <= k:
            top_idx = np.arange(len(sims_row))
        else:
            top_idx = np.argpartition(sims_row, -k)[-k:]

        top_k_rets = returns[top_idx]  # (k, 5)
        sig = signal_from_neighbors(top_k_rets, buy_thr, sell_thr)

        if sig == "HOLD":
            continue

        actual = returns[i, eval_horizon_idx]
        if np.isnan(actual):
            continue

        correct += int((sig == "BUY" and actual > 0) or (sig == "SELL" and actual < 0))
        total   += 1
        buy_n   += int(sig == "BUY")
        sell_n  += int(sig == "SELL")

    da       = correct / total if total > 0 else 0.0
    coverage = total / (N - warmup) if N > warmup else 0.0
    return {"da": da, "coverage": coverage, "total": total, "buy": buy_n, "sell": sell_n}


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(
    F_sim, I_sim, P_sim, returns,
    k, warmup, buy_thr, sell_thr, min_coverage,
):
    def objective(trial: "optuna.Trial") -> float:
        w1 = trial.suggest_float("w1", 0.10, 0.80)
        w2 = trial.suggest_float("w2", 0.05, 0.50)
        w3 = trial.suggest_float("w3", 0.05, 0.70)
        # Normalize so they sum to 1
        total = w1 + w2 + w3
        w1, w2, w3 = w1 / total, w2 / total, w3 / total

        res = walk_forward(
            F_sim, I_sim, P_sim, returns,
            w1, w2, w3, k=k, warmup=warmup,
            buy_thr=buy_thr, sell_thr=sell_thr,
        )
        # Penalize coverage below min_coverage
        coverage_penalty = max(0.0, min_coverage - res["coverage"]) * 2.0
        return res["da"] - coverage_penalty

    return objective


# ── Stable median selection ───────────────────────────────────────────────────

def select_stable_weights(study: "optuna.Study", top_n: int = 10) -> tuple[float, float, float]:
    """Take top-n trials by value and return their median weights."""
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    completed.sort(key=lambda t: t.value or 0.0, reverse=True)
    top = completed[:top_n]

    w1s = [t.params["w1"] / (t.params["w1"] + t.params["w2"] + t.params["w3"]) for t in top]
    w2s = [t.params["w2"] / (t.params["w1"] + t.params["w2"] + t.params["w3"]) for t in top]
    w3s = [t.params["w3"] / (t.params["w1"] + t.params["w2"] + t.params["w3"]) for t in top]

    return float(np.median(w1s)), float(np.median(w2s)), float(np.median(w3s))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    if optuna is None:
        print("ERROR: optuna not installed. Run: pip install optuna")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",       type=int,   default=150)
    ap.add_argument("--warmup",       type=int,   default=250)
    ap.add_argument("--k",            type=int,   default=5)
    ap.add_argument("--buy-thr",      type=float, default=2.0)
    ap.add_argument("--sell-thr",     type=float, default=2.0)
    ap.add_argument("--min-coverage", type=float, default=0.40,
                    help="Minimum BUY+SELL fraction; penalize below this")
    ap.add_argument("--top-n",        type=int,   default=15,
                    help="Top-N trials used for stable median weight selection")
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    print("Loading records from PostgreSQL...", flush=True)
    dates, F, I, P, returns = await load_data()
    N = len(dates)
    print(f"Loaded {N} records ({dates[0]} → {dates[-1]})")

    F_sim, I_sim, P_sim = precompute_sims(F, I, P)

    print(f"\nRunning Optuna ({args.trials} trials, warmup={args.warmup}, "
          f"k={args.k}, threshold=±{args.buy_thr}%, min_coverage={args.min_coverage:.0%})...",
          flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Seed with known good weights
    study.enqueue_trial({"w1": 0.35, "w2": 0.20, "w3": 0.45})  # default
    study.enqueue_trial({"w1": 0.47, "w2": 0.31, "w3": 0.22})  # previous Bayesian

    objective = make_objective(
        F_sim, I_sim, P_sim, returns,
        k=args.k, warmup=args.warmup,
        buy_thr=args.buy_thr, sell_thr=args.sell_thr,
        min_coverage=args.min_coverage,
    )

    def _progress(study, trial):
        if trial.number % 25 == 0 or trial.number < 5:
            best = study.best_value
            t = study.best_trial
            sw = t.params
            tot = sw["w1"] + sw["w2"] + sw["w3"]
            print(f"  trial {trial.number:3d}  best={best:.4f}  "
                  f"w=({sw['w1']/tot:.3f}, {sw['w2']/tot:.3f}, {sw['w3']/tot:.3f})",
                  flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[_progress])

    # Best trial
    best = study.best_trial
    tot = best.params["w1"] + best.params["w2"] + best.params["w3"]
    bw1 = best.params["w1"] / tot
    bw2 = best.params["w2"] / tot
    bw3 = best.params["w3"] / tot

    # Stable median weights from top-N
    sw1, sw2, sw3 = select_stable_weights(study, top_n=args.top_n)
    print(f"\nBest single trial:    w=({bw1:.4f}, {bw2:.4f}, {bw3:.4f})  score={best.value:.4f}")
    print(f"Stable median top-{args.top_n}: w=({sw1:.4f}, {sw2:.4f}, {sw3:.4f})")

    # Evaluate stable weights fully
    res = walk_forward(
        F_sim, I_sim, P_sim, returns,
        sw1, sw2, sw3,
        k=args.k, warmup=args.warmup,
        buy_thr=args.buy_thr, sell_thr=args.sell_thr,
    )
    print(f"\nFinal eval (stable weights, D+7d):")
    print(f"  DA={res['da']:.4f}  coverage={res['coverage']:.2%}  "
          f"total={res['total']}  BUY={res['buy']}  SELL={res['sell']}")

    # Also evaluate default weights for comparison
    res_default = walk_forward(
        F_sim, I_sim, P_sim, returns,
        0.35, 0.20, 0.45,
        k=args.k, warmup=args.warmup,
        buy_thr=args.buy_thr, sell_thr=args.sell_thr,
    )
    print(f"  Default weights:   DA={res_default['da']:.4f}  coverage={res_default['coverage']:.2%}")
    print(f"  Improvement:       ΔDA={res['da'] - res_default['da']:+.4f}")

    # Save
    out = {
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "optimizer": "knn_returns_specific",
        "horizon": "7d",
        "signal": "multi-horizon-weighted-avg",
        "return_weights": RETURN_W,
        "buy_threshold": args.buy_thr,
        "sell_threshold": args.sell_thr,
        "n_records": N,
        "warmup": args.warmup,
        "k": args.k,
        "trials": args.trials,
        "stable_top_n": args.top_n,
        "weights": {
            "w1_factor":    sw1,
            "w2_indicator": sw2,
            "w3_price":     sw3,
        },
        "metrics": {
            "da":       res["da"],
            "coverage": res["coverage"],
        },
        "best_trial_score": best.value,
        "default_weights_da": res_default["da"],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
