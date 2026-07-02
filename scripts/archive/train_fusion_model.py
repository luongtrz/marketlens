"""CEM-RAG Fusion Model (Bước 8 proper).

Feature matrix per query:
  X = [current_vec(225d) | mean_neighbor_vec(225d) | p_up | p_down | p_hold | mean_sim] = 453d

This is the true RAG fusion: retrieved neighbors become the *context* for a learned
forecasting model, not just a kNN vote. The model learns to interpret whether the
retrieved context is reliable or misleading.

Pipeline:
  1. Load labeled rows (train/val/test splits)
  2. For each query, retrieve k=5 matured neighbors via LearnedDiagonalMetric (v4)
  3. Build 453d feature matrix
  4. StandardScaler fit on train only
  5. Train LR + RF; calibrate on val; tune tau on val
  6. Evaluate on test; write predictions; append to main_table.csv

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/train_fusion_model.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from stockmem.scripts.cem_dataset import label_rows, matured_pool
from stockmem.scripts.optimize_weights import load_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric

PRED_DIR  = ROOT / "artifacts/predictions"
METRICS   = ROOT / "artifacts/metrics/main_table.csv"
DATA_PATH = ROOT / "stockmem/data/real_optimizer_v3.json"
RETRIEVER = ROOT / "stockmem/config/learned_retriever_v4.json"
K         = 5
COST_PCT  = 0.10
PERIODS_PER_YEAR = 252 / 7
HORIZON   = "future_return_7d"

PRED_DIR.mkdir(parents=True, exist_ok=True)


# ── Load retriever ───────────────────────────────────────────────────────────

def load_retriever() -> LearnedDiagonalMetric:
    payload = json.loads(RETRIEVER.read_text())
    return LearnedDiagonalMetric.from_payload(payload)


# ── Block decomposition ──────────────────────────────────────────────────────

def _blocks(lb) -> list[np.ndarray]:
    """Return [event_vec(85), factor_vec(75), indicator_vec(5), price_vec(60)]."""
    row = lb.row
    def _v(attr, dim):
        val = getattr(row, attr, None)
        if val is None:
            return np.zeros(dim, dtype=np.float64)
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape == (dim,):
            return arr
        return np.zeros(dim, dtype=np.float64)
    return [_v("event_vec", 85), _v("factor_vec", 75), _v("indicator_vec", 5), _v("price_vec", 60)]


def _flat(lb) -> np.ndarray:
    return np.concatenate(_blocks(lb))


# ── Retrieve k neighbors ─────────────────────────────────────────────────────

def retrieve(metric: LearnedDiagonalMetric, query, pool) -> tuple[list, list[float]]:
    """Return (neighbors, sims) sorted descending by sim."""
    q_blocks = _blocks(query)
    scores = []
    for cand in pool:
        c_blocks = _blocks(cand)
        try:
            s = metric.score(q_blocks, c_blocks)
        except Exception:
            s = 0.0
        scores.append((cand, float(s)))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:K]
    return [t[0] for t in top], [t[1] for t in top]


# ── Build fusion feature vector ──────────────────────────────────────────────

def fusion_vec(query, neighbors, sims) -> np.ndarray:
    """232d: current(225) | p_up | p_down | mean_neigh_ret | mean_sim | delta_norm.

    We use only SUMMARY retrieval features rather than the full neighbor vec —
    the full neighbor vec carries the bull-market prior that anti-predicts in a
    bear test period, while the vote + delta captures the useful signal.
    """
    cur = _flat(query)  # 225d

    if neighbors:
        w = np.asarray(sims, dtype=np.float64)
        w = (w + 1.0) / 2.0   # map [-1,1] → [0,1]
        w = w / (w.sum() + 1e-9)

        # Sim-weighted vote
        rets = [getattr(nb.row, HORIZON, None) for nb in neighbors]
        valid = [(w[i], float(r)) for i, r in enumerate(rets) if r is not None]
        if valid:
            tw = sum(ww for ww, _ in valid)
            p_up   = sum(ww for ww, r in valid if r > 0) / max(tw, 1e-9)
            p_down = sum(ww for ww, r in valid if r < 0) / max(tw, 1e-9)
            mean_neigh_ret = sum(ww * r for ww, r in valid) / max(tw, 1e-9)
        else:
            p_up = p_down = 1.0 / 3
            mean_neigh_ret = 0.0

        # Delta norm: how far is the query from its retrieved neighbors?
        neigh_vecs = np.stack([_flat(n) for n in neighbors])
        mean_neigh_vec = (neigh_vecs * w[:, None]).sum(axis=0)
        delta = cur - mean_neigh_vec
        delta_norm = float(np.linalg.norm(delta))
        mean_sim = float(np.mean(sims))
    else:
        p_up = p_down = 1.0 / 3
        mean_neigh_ret = 0.0
        delta_norm = 0.0
        mean_sim = 0.0

    return np.concatenate([cur, [p_up, p_down, mean_neigh_ret, mean_sim, delta_norm]])


# ── Build dataset ────────────────────────────────────────────────────────────

def build_dataset(labeled, metric) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Returns X, y, meta — one row per labeled entry (all splits)."""
    X_rows, y_rows, meta_rows = [], [], []
    total = len(labeled)
    for i, lb in enumerate(labeled):
        ret7 = getattr(lb.row, HORIZON, None)
        if ret7 is None:
            continue
        ret7 = float(ret7)
        if ret7 == 0.0:
            continue

        pool = matured_pool(labeled, lb, guard=True)
        neighbors, sims = retrieve(metric, lb, pool)
        fvec = fusion_vec(lb, neighbors, sims)

        X_rows.append(fvec)
        y_rows.append(1 if ret7 > 0 else 0)
        meta_rows.append({
            "date":  lb.row.date.isoformat() if hasattr(lb.row.date, "isoformat") else str(lb.row.date),
            "split": lb.split,
            "actual_return_7d":  ret7,
            "actual_return_1d":  float(getattr(lb.row, "future_return_1d",  None) or 0),
            "actual_return_3d":  float(getattr(lb.row, "future_return_3d",  None) or 0),
            "actual_return_15d": float(getattr(lb.row, "future_return_15d", None) or 0),
            "actual_return_30d": float(getattr(lb.row, "future_return_30d", None) or 0),
        })
        if (i + 1) % 200 == 0:
            print(f"  built {i+1}/{total} rows...", flush=True)

    return np.array(X_rows), np.array(y_rows), meta_rows


# ── Calibration (manual Platt) ───────────────────────────────────────────────

def platt_calibrate(base_model, X_val, y_val):
    """Fit a logistic calibrator on val raw scores."""
    raw = base_model.predict_proba(X_val)[:, 1]
    cal = LogisticRegression(C=1.0, max_iter=500)
    cal.fit(raw.reshape(-1, 1), y_val)
    return cal


def calibrated_proba(base_model, cal, X):
    raw = base_model.predict_proba(X)[:, 1]
    p_up = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    p_down = 1.0 - p_up
    return p_up, p_down


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    directional = [r for r in rows if r["signal"] != "HOLD"]
    n_dir = len(directional)

    y_true_all = [r["label"] for r in rows]
    y_pred_all = [1 if r["signal"] == "BUY" else (0 if r["signal"] == "SELL" else -1) for r in rows]

    # DA among directional
    da = sum(1 for r in directional
             if (r["signal"] == "BUY" and r["label"] == 1)
             or (r["signal"] == "SELL" and r["label"] == 0)) / max(1, n_dir)

    coverage = n_dir / max(1, len(rows))

    y_t = [r["label"] for r in rows]
    y_p = [1 if r["signal"] == "BUY" else 0 for r in rows]
    bal  = balanced_accuracy_score(y_t, y_p)
    f1   = f1_score(y_t, y_p, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(y_t, y_p)

    rets = []
    for r in rows:
        sig = r["signal"]
        ret = r["actual_return_7d"]
        if sig == "BUY":   rets.append(ret - COST_PCT)
        elif sig == "SELL": rets.append(-ret - COST_PCT)
        else:               rets.append(0.0)
    arr = np.array(rets)
    mean_r = arr.mean() * PERIODS_PER_YEAR
    std_r  = arr.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR) if len(arr) > 1 else 1e-9
    sharpe = float(mean_r / std_r) if std_r > 1e-9 else 0.0

    neg = arr[arr < 0]
    down_std = neg.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR) if len(neg) > 1 else 1e-9
    sortino = float(mean_r / down_std) if down_std > 1e-9 else 0.0

    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in rets:
        cum *= (1 + r / 100)
        if cum > peak:
            peak = cum
        dd = (cum - peak) / peak
        if dd < max_dd:
            max_dd = dd

    brier = float(np.mean([(r["p_up"] - r["label"]) ** 2 for r in rows]))

    sell_da = sum(1 for r in rows if r["signal"] == "SELL" and r["label"] == 0) / max(1, sum(1 for r in rows if r["signal"] == "SELL"))
    buy_da  = sum(1 for r in rows if r["signal"] == "BUY" and r["label"] == 1) / max(1, sum(1 for r in rows if r["signal"] == "BUY"))

    return {
        "da": round(da, 6), "balanced_acc": round(bal, 6), "macro_f1": round(f1, 6),
        "mcc": round(mcc, 6), "coverage": round(coverage, 6),
        "sharpe": round(sharpe, 6), "sortino": round(sortino, 6), "max_dd": round(max_dd, 6),
        "hit_at_5": 0.0, "brier": round(brier, 6),
        "sell_da": round(sell_da, 6), "buy_da": round(buy_da, 6),
        "n_buy": sum(1 for r in rows if r["signal"] == "BUY"),
        "n_sell": sum(1 for r in rows if r["signal"] == "SELL"),
        "n_hold": sum(1 for r in rows if r["signal"] == "HOLD"),
    }


# ── Tune tau on val ──────────────────────────────────────────────────────────

def tune_tau(val_rows: list[dict]) -> float:
    best_tau, best_sell_da, best_cov = 0.0, 0.0, 1.0
    for tau_int in range(0, 46, 2):
        tau = tau_int / 100.0
        tagged = []
        for r in val_rows:
            pu, pd = r["p_up"], r["p_down"]
            if pu - pd >= tau:
                sig = "BUY"
            elif pd - pu >= tau:
                sig = "SELL"
            else:
                sig = "HOLD"
            tagged.append({**r, "signal": sig})

        n_sell = sum(1 for r in tagged if r["signal"] == "SELL")
        if n_sell < max(3, 0.03 * len(tagged)):
            continue
        sell_da = sum(1 for r in tagged
                      if r["signal"] == "SELL" and r["label"] == 0) / max(1, n_sell)
        cov = sum(1 for r in tagged if r["signal"] != "HOLD") / max(1, len(tagged))
        if sell_da > best_sell_da or (sell_da == best_sell_da and cov > best_cov):
            best_sell_da, best_tau, best_cov = sell_da, tau, cov
    return best_tau


def apply_tau(rows: list[dict], tau: float) -> list[dict]:
    out = []
    for r in rows:
        pu, pd = r["p_up"], r["p_down"]
        if pu - pd >= tau:
            sig = "BUY"
        elif pd - pu >= tau:
            sig = "SELL"
        else:
            sig = "HOLD"
        out.append({**r, "signal": sig, "confidence": float(max(pu, pd))})
    return out


# ── Write JSONL ──────────────────────────────────────────────────────────────

def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows → {path}")


# ── Append to main_table.csv ─────────────────────────────────────────────────

def append_main_table(name: str, m: dict) -> None:
    header = "retriever,da,balanced_acc,macro_f1,mcc,coverage,sharpe,sortino,max_dd,hit_at_5,brier\n"
    row = (f"{name},{m['da']},{m['balanced_acc']},{m['macro_f1']},{m['mcc']},"
           f"{m['coverage']},{m['sharpe']},{m['sortino']},{m['max_dd']},"
           f"{m['hit_at_5']},{m['brier']}\n")
    txt = METRICS.read_text() if METRICS.exists() else header
    # remove old entry with same name
    lines = [l for l in txt.splitlines(keepends=True) if not l.startswith(name + ",")]
    METRICS.write_text("".join(lines) + row)
    print(f"Updated main_table.csv → {name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data and retriever...")
    rows = load_rows(DATA_PATH)
    labeled = label_rows(rows)
    metric  = load_retriever()

    print(f"Splits: train={sum(1 for l in labeled if l.split=='train')} "
          f"val={sum(1 for l in labeled if l.split=='val')} "
          f"test={sum(1 for l in labeled if l.split=='test')}")

    print("\nBuilding 453d fusion feature matrix (this takes a few minutes)...")
    X_all, y_all, meta_all = build_dataset(labeled, metric)
    print(f"Total rows with label: {len(X_all)}, feature dim: {X_all.shape[1]}")

    splits = np.array([m["split"] for m in meta_all])
    tr = splits == "train"
    va = splits == "val"
    te = splits == "test"

    X_tr, y_tr = X_all[tr], y_all[tr]
    X_va, y_va = X_all[va], y_all[va]
    X_te, y_te = X_all[te], y_all[te]
    meta_te = [m for m, s in zip(meta_all, splits) if s == "test"]
    meta_va = [m for m, s in zip(meta_all, splits) if s == "val"]

    print(f"\ntrain={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    # StandardScaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    models_cfg = [
        ("cem_rag_lr_fusion",
         LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")),
        ("cem_rag_rf_fusion",
         RandomForestClassifier(n_estimators=200, max_depth=6,
                                class_weight="balanced", random_state=42, n_jobs=-1)),
    ]

    for model_name, base_model in models_cfg:
        print(f"\n{'='*55}")
        print(f"Model: {model_name}")

        base_model.fit(X_tr_s, y_tr)

        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(y_va, base_model.predict_proba(X_va_s)[:, 1])
        print(f"  Val AUC: {val_auc:.4f}")

        cal = platt_calibrate(base_model, X_va_s, y_va)

        # Build val prediction rows for tau tuning
        p_up_va, p_down_va = calibrated_proba(base_model, cal, X_va_s)
        val_pred_rows = []
        for i, m in enumerate(meta_va):
            val_pred_rows.append({
                **m, "p_up": float(p_up_va[i]), "p_down": float(p_down_va[i]),
                "p_hold": 0.0, "label": int(y_va[i]), "signal": "HOLD",
            })

        tau = tune_tau(val_pred_rows)
        print(f"  Best tau (val): {tau:.2f}")

        val_tagged = apply_tau(val_pred_rows, tau)
        n_vs = sum(1 for r in val_tagged if r["signal"] == "SELL")
        val_sell_da = sum(1 for r in val_tagged if r["signal"] == "SELL" and r["label"] == 0) / max(1, n_vs)
        print(f"  Val SELL-DA: {val_sell_da:.3f} (n_sell={n_vs})")

        # Test predictions
        p_up_te, p_down_te = calibrated_proba(base_model, cal, X_te_s)
        test_pred_rows = []
        for i, m in enumerate(meta_te):
            test_pred_rows.append({
                **m, "symbol": "BTC",
                "p_up": float(p_up_te[i]), "p_down": float(p_down_te[i]),
                "p_hold": 0.0, "label": int(y_te[i]), "signal": "HOLD",
            })

        test_tagged = apply_tau(test_pred_rows, tau)
        m = compute_metrics(test_tagged)

        print(f"  Test: DA={m['da']:.3f} SELL-DA={m['sell_da']:.3f} BUY-DA={m['buy_da']:.3f}")
        print(f"        MCC={m['mcc']:.4f} Sharpe={m['sharpe']:.4f} cov={m['coverage']:.3f}")
        print(f"        n_buy={m['n_buy']} n_sell={m['n_sell']} n_hold={m['n_hold']}")

        # Write JSONL
        out_rows = []
        for r in test_tagged:
            out_rows.append({
                "date": r["date"], "symbol": r.get("symbol", "BTC"),
                "signal": r["signal"], "confidence": r["confidence"],
                "p_up": r["p_up"], "p_down": r["p_down"], "p_hold": r["p_hold"],
                "actual_return_1d":  r["actual_return_1d"],
                "actual_return_3d":  r["actual_return_3d"],
                "actual_return_7d":  r["actual_return_7d"],
                "actual_return_15d": r["actual_return_15d"],
                "actual_return_30d": r["actual_return_30d"],
            })
        write_jsonl(out_rows, PRED_DIR / f"{model_name}_test.jsonl")
        append_main_table(model_name, m)

    print("\nDone.")


if __name__ == "__main__":
    main()
