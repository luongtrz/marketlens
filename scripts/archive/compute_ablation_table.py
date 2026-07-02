"""CEM-RAG ablation study — remove one component at a time.

Ablation variants:
  cem_rag_full          learned retriever (4-block) + tau=0.22          [baseline for ablation]
  ablation_no_event     event block zeroed in retriever + tau=0.22      [event retrieval]
  ablation_no_factor    factor block zeroed in retriever + tau=0.22     [factor retrieval]
  ablation_price_only   event+factor blocks zeroed + tau=0.22           [price+indicator only]
  ablation_fixed_ret    fixed weighted kNN + tau=0.22                   [learned retrieval]
  ablation_no_policy    learned retriever + tau=0 (always BUY/SELL)     [policy calibration]
  ablation_no_retrieval always HOLD (no kNN neighbours used)            [retrieval itself]

All variants use the same test split (2025-07-01 →) and maturity guard.
Output:
  artifacts/predictions/ablation_*.jsonl
  artifacts/metrics/ablation_table.csv

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/compute_ablation_table.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stockmem.scripts.cem_dataset import label_rows, matured_pool
from stockmem.scripts.evaluate_retriever import DEFAULT_WEIGHTS
from stockmem.scripts.optimize_weights import load_rows
from stockmem.src.search.learned_metric import LearnedDiagonalMetric

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOL      = "BTC"
K           = 5
CEM_TAU     = 0.22
HORIZON     = "7d"
HORIZONS    = ["1d", "3d", "7d", "15d", "30d"]
COST_PCT    = 0.10   # 10 bps for trading metrics
PERIODS_PER_YEAR = 252 / 7

DATA_PATH       = PROJECT_ROOT / "stockmem/data/real_optimizer_v3.json"
RETRIEVER_PATH  = PROJECT_ROOT / "stockmem/config/learned_retriever.json"
OUT_PRED_DIR    = PROJECT_ROOT / "artifacts/predictions"
OUT_TABLE       = PROJECT_ROOT / "artifacts/metrics/ablation_table.csv"


# ── Scoring helpers ─────────────────────────────────────────────────────────────

def _fixed_score_fn(weights):
    def _score(q, c):
        return (
            weights[0] * float(np.dot(q.row.factor_vec,    c.row.factor_vec))
          + weights[1] * float(np.dot(q.row.indicator_vec, c.row.indicator_vec))
          + weights[2] * float(np.dot(q.row.price_vec,     c.row.price_vec))
        )
    return _score


def _learned_score_fn(metric: LearnedDiagonalMetric):
    def _score(q, c):
        return metric.score(q.blocks, c.blocks)
    return _score


def _zeroed_block_metric(base: LearnedDiagonalMetric, *zero_indices: int) -> LearnedDiagonalMetric:
    scales = np.asarray(base.block_scales, dtype=np.float64).copy()
    original_total = float(scales.sum())
    for idx in zero_indices:
        scales[idx] = 0.0
    remaining = float(scales.sum())
    if remaining > 1e-12:
        scales *= original_total / remaining   # renormalize so total scale is preserved
    return LearnedDiagonalMetric(
        block_dims=base.block_dims,
        diagonal=base.diagonal.copy(),
        block_scales=scales,
        version=base.version,
    )


# ── CEM-RAG signal ──────────────────────────────────────────────────────────────

def _cem_signal(neighbors, sims: list[float], tau: float, horizon: str = "7d"):
    attr = f"future_return_{horizon}"
    usable = [(s, nb) for s, nb in zip(sims, neighbors)
              if getattr(nb.row, attr, None) is not None]
    if not usable:
        return "HOLD", 1/3, 1/3, 1/3, 0.50
    weights = [(s + 1) / 2 for s, _ in usable]
    total_w = sum(weights) + 1e-12
    p_up   = sum(w for w, (_, nb) in zip(weights, usable) if getattr(nb.row, attr) > 0) / total_w
    p_down = sum(w for w, (_, nb) in zip(weights, usable) if getattr(nb.row, attr) < 0) / total_w
    p_hold = max(0.0, 1.0 - p_up - p_down)
    du, dd = p_up - p_down, p_down - p_up
    if tau <= 0:                    # forced: always commit to majority direction
        if p_up >= p_down:
            return "BUY",  round(min(0.50 + du * 0.80, 0.95), 3), p_up, p_down, p_hold
        else:
            return "SELL", round(min(0.50 + dd * 0.80, 0.95), 3), p_up, p_down, p_hold
    if du >= tau:
        return "BUY",  round(min(0.50 + du * 0.80, 0.95), 3), p_up, p_down, p_hold
    if dd >= tau:
        return "SELL", round(min(0.50 + dd * 0.80, 0.95), 3), p_up, p_down, p_hold
    return "HOLD", round(0.50 + max(du, dd) * 0.30, 3), p_up, p_down, p_hold


# ── Evaluation ──────────────────────────────────────────────────────────────────

def _sharpe(rets: list[float]) -> float:
    if len(rets) < 2: return 0.0
    arr = np.asarray(rets)
    mean = arr.mean() * PERIODS_PER_YEAR
    std  = arr.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)
    return float(mean / std) if std > 1e-9 else 0.0


def _max_dd(rets: list[float]) -> float:
    if not rets: return 0.0
    cum = np.cumprod(1 + np.asarray(rets) / 100) * 100
    peak, max_dd = cum[0], 0.0
    for v in cum:
        if v > peak: peak = v
        dd = (v - peak) / peak
        if dd < max_dd: max_dd = dd
    return float(max_dd * 100)


def _metrics(rows_out: list[dict]) -> dict:
    buy_c = sell_c = hold_c = bc = sc = 0
    rets: list[float] = []
    for r in rows_out:
        s = r["signal"]; ret7 = r.get("actual_return_7d")
        if ret7 is None: continue
        ret7 = float(ret7)
        if s == "BUY":
            buy_c += 1; rets.append(ret7 - COST_PCT)
            bc += int(ret7 > 0)
        elif s == "SELL":
            sell_c += 1; rets.append(-ret7 - COST_PCT)
            sc += int(ret7 < 0)
        else:
            hold_c += 1; rets.append(0.0)
    n = buy_c + sell_c + hold_c
    nd = buy_c + sell_c
    cov    = nd / n if n else 0.0
    buy_da = bc / buy_c  if buy_c  else 0.0
    sell_da= sc / sell_c if sell_c else 0.0
    dir_da = (bc + sc) / nd if nd else 0.0
    sharpe = _sharpe(rets)
    max_dd = _max_dd(rets)

    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
    y_pred = [r["signal"] for r in rows_out if r.get("actual_return_7d") is not None]
    y_true = [
        "BUY" if float(r["actual_return_7d"]) > 0 else "SELL"
        for r in rows_out if r.get("actual_return_7d") is not None
    ]
    ba  = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=["BUY","SELL","HOLD"], average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0

    return dict(
        n=n, n_buy=buy_c, n_sell=sell_c, n_hold=hold_c,
        coverage=round(cov,4), buy_da=round(buy_da,4), sell_da=round(sell_da,4),
        directional_da=round(dir_da,4), balanced_acc=round(ba,4),
        macro_f1=round(mf1,4), mcc=round(mcc,4),
        sharpe=round(sharpe,4), max_drawdown_pct=round(max_dd,3),
    )


# ── Per-variant runner ──────────────────────────────────────────────────────────

def run_variant(
    name: str,
    labeled,
    score_fn: Callable | None,   # None → always HOLD (no retrieval)
    tau: float,
    k: int = K,
) -> list[dict]:
    rows_out: list[dict] = []
    test_queries = [lb for lb in labeled if lb.split == "test"]
    for query in test_queries:
        actual = {h: getattr(query.row, f"future_return_{h}", None) for h in HORIZONS}
        if score_fn is None:
            # no-retrieval ablation: always HOLD
            rows_out.append({
                "date": query.row.date, "symbol": SYMBOL, "variant": name,
                "signal": "HOLD", "confidence": 0.50,
                "p_up": 1/3, "p_down": 1/3, "p_hold": 1/3,
                **{f"actual_return_{h}": actual[h] for h in HORIZONS},
            })
            continue

        pool = matured_pool(labeled, query, guard=True)
        if not pool:
            rows_out.append({
                "date": query.row.date, "symbol": SYMBOL, "variant": name,
                "signal": "HOLD", "confidence": 0.50,
                "p_up": 1/3, "p_down": 1/3, "p_hold": 1/3,
                **{f"actual_return_{h}": actual[h] for h in HORIZONS},
            })
            continue

        scored = sorted(
            [(score_fn(query, c), c) for c in pool],
            key=lambda x: x[0], reverse=True,
        )[:k]
        neighbors = [c for _, c in scored]
        sims      = [float(s) for s, _ in scored]

        signal, conf, p_up, p_down, p_hold = _cem_signal(neighbors, sims, tau, HORIZON)
        rows_out.append({
            "date": query.row.date, "symbol": SYMBOL, "variant": name,
            "signal": signal, "confidence": round(conf, 3),
            "p_up": round(p_up, 4), "p_down": round(p_down, 4), "p_hold": round(p_hold, 4),
            "retrieval_count": len(neighbors),
            **{f"actual_return_{h}": actual[h] for h in HORIZONS},
        })
    return rows_out


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data and retriever…")
    rows    = load_rows(DATA_PATH)
    labeled = label_rows(rows)

    artifact = json.loads(RETRIEVER_PATH.read_text())
    base_metric = LearnedDiagonalMetric.from_payload(artifact)
    block_dims  = base_metric.block_dims      # [85, 75, 5, 60] for 4-block

    # Block indices: 0=event, 1=factor, 2=indicator, 3=price
    assert len(block_dims) == 4, f"Expected 4-block retriever, got {block_dims}"
    EVENT_IDX, FACTOR_IDX = 0, 1

    # Build zeroed-block metrics
    no_event_metric   = _zeroed_block_metric(base_metric, EVENT_IDX)
    no_factor_metric  = _zeroed_block_metric(base_metric, FACTOR_IDX)
    price_only_metric = _zeroed_block_metric(base_metric, EVENT_IDX, FACTOR_IDX)

    variants = [
        ("cem_rag_full",        _learned_score_fn(base_metric),     CEM_TAU),
        ("ablation_no_event",   _learned_score_fn(no_event_metric), CEM_TAU),
        ("ablation_no_factor",  _learned_score_fn(no_factor_metric),CEM_TAU),
        ("ablation_price_only", _learned_score_fn(price_only_metric),CEM_TAU),
        ("ablation_fixed_ret",  _fixed_score_fn(DEFAULT_WEIGHTS),    CEM_TAU),
        ("ablation_no_policy",  _learned_score_fn(base_metric),      0.0),   # tau=0 → always commit
        ("ablation_no_retrieval", None,                              CEM_TAU),
    ]

    fieldnames = [
        "variant", "n", "n_buy", "n_sell", "n_hold",
        "coverage", "buy_da", "sell_da", "directional_da",
        "balanced_acc", "macro_f1", "mcc",
        "sharpe", "max_drawdown_pct",
    ]

    print(f"\n{'Variant':<30} {'cov':>5}  {'BUY-DA':>7}  {'SELL-DA':>7}  {'Dir-DA':>7}  {'MCC':>7}  {'Sharpe':>7}")
    print("-" * 75)

    table_rows: list[dict] = []
    for name, score_fn, tau in variants:
        print(f"  running {name}…", end=" ", flush=True)
        rows_out = run_variant(name, labeled, score_fn, tau)

        # Write JSONL
        out_path = OUT_PRED_DIR / f"{name}_test.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps({
                    k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                    for k, v in r.items()
                }) + "\n")

        m = _metrics(rows_out)
        table_rows.append({"variant": name, **m})
        print(
            f"  {m['coverage']:>5.3f}  {m['buy_da']:>7.3f}  {m['sell_da']:>7.3f}  "
            f"{m['directional_da']:>7.3f}  {m['mcc']:>7.3f}  {m['sharpe']:>7.3f}"
        )

    with OUT_TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"\nWrote {len(table_rows)} rows → {OUT_TABLE}")
    print(f"Wrote JSONL files → {OUT_PRED_DIR}/ablation_*_test.jsonl")


if __name__ == "__main__":
    main()
