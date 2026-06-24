"""Compute statistical significance tests on test-period prediction JSONL files.

Tests:
  McNemar (exact two-tailed) — paired directional correctness
  Block bootstrap 95% CI on DA delta — accounts for time-series autocorrelation
  Block bootstrap 95% CI on Sharpe delta

Comparisons:
  knn_returns vs always_hold      (strong baseline vs trivial)
  cem_rag vs knn_returns          (proposed vs strong baseline)
  cem_rag vs fixed_knn            (proposed vs original)
  ablation_no_factor vs cem_rag   (factor ablation significance)

Output:
  artifacts/metrics/stat_tests.json

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/compute_stat_tests.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

PRED_DIR  = ROOT / "artifacts/predictions"
OUT_PATH  = ROOT / "artifacts/metrics/stat_tests.json"
COST_PCT  = 0.10   # 10 bps (same as trading table)
PERIODS_PER_YEAR = 252 / 7
HORIZON   = "actual_return_7d"


# ── stat functions ──────────────────────────────────────────────────────────────

def mcnemar_exact(left: Sequence[bool], right: Sequence[bool]) -> dict:
    """Two-sided exact McNemar on paired correctness booleans."""
    pairs = [(a, b) for a, b in zip(left, right)]
    lo = sum(1 for a, b in pairs if a and not b)
    ro = sum(1 for a, b in pairs if b and not a)
    disc = lo + ro
    if disc == 0:
        return {"left_only": 0, "right_only": 0, "discordant": 0,
                "p_value": 1.0, "significant_0.05": False, "significant_0.10": False}
    tail = sum(math.comb(disc, i) for i in range(min(lo, ro) + 1))
    p = min(1.0, 2.0 * tail / (2 ** disc))
    return {
        "left_only": lo,
        "right_only": ro,
        "discordant": disc,
        "p_value": round(p, 6),
        "significant_0.05": p < 0.05,
        "significant_0.10": p < 0.10,
    }


def block_bootstrap_ci(
    series_a: np.ndarray,
    series_b: np.ndarray,
    *,
    block_size: int = 7,
    samples: int = 5000,
    seed: int = 42,
) -> dict:
    """95% CI for mean(series_b) - mean(series_a) via block bootstrap."""
    n = min(len(series_a), len(series_b))
    if n < block_size:
        return {"ci_95_low": 0.0, "ci_95_high": 0.0, "delta_positive": False,
                "note": "too few samples"}
    a = np.asarray(series_a[:n], dtype=np.float64)
    b = np.asarray(series_b[:n], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n_blocks = max(1, n // block_size)
    starts = np.arange(0, n - block_size + 1)
    for _ in range(samples):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in chosen])[:n]
        deltas.append(float(b[idx].mean() - a[idx].mean()))
    low  = float(np.quantile(deltas, 0.025))
    high = float(np.quantile(deltas, 0.975))
    return {
        "ci_95_low":       round(low, 6),
        "ci_95_high":      round(high, 6),
        "observed_delta":  round(float(b.mean() - a.mean()), 6),
        "delta_positive":  low > 0,
        "block_size":      block_size,
        "samples":         samples,
    }


def _sharpe_series(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns)
    mean = arr.mean() * PERIODS_PER_YEAR
    std  = arr.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)
    return float(mean / std) if std > 1e-9 else 0.0


# ── load helper ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["date"])


def _is_correct(signal: str, ret7: float | None) -> bool | None:
    if ret7 is None: return None
    if signal == "BUY":  return ret7 > 0
    if signal == "SELL": return ret7 < 0
    return None  # HOLD not included in McNemar


def _returns_after_cost(rows: list[dict]) -> list[float]:
    out = []
    for r in rows:
        sig = r.get("signal", "HOLD")
        ret7 = r.get(HORIZON)
        if ret7 is None: continue
        ret7 = float(ret7)
        if sig == "BUY":   out.append(ret7 - COST_PCT)
        elif sig == "SELL": out.append(-ret7 - COST_PCT)
        else:               out.append(0.0)
    return out


def _directional_correctness(rows: list[dict]) -> list[bool]:
    """Per-day bool: True if BUY/SELL signal was correct."""
    out = []
    for r in rows:
        sig  = r.get("signal", "HOLD")
        ret7 = r.get(HORIZON)
        c    = _is_correct(sig, float(ret7) if ret7 is not None else None)
        if c is not None:
            out.append(c)
    return out


def _da(rows: list[dict]) -> float:
    corr = _directional_correctness(rows)
    return sum(corr) / len(corr) if corr else 0.0


def _paired_align(a_rows: list[dict], b_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return rows for dates present in BOTH files, sorted by date."""
    a_by_date = {r["date"]: r for r in a_rows}
    b_by_date = {r["date"]: r for r in b_rows}
    dates = sorted(set(a_by_date) & set(b_by_date))
    return [a_by_date[d] for d in dates], [b_by_date[d] for d in dates]


# ── paired DA correctness (only directional in at least one) ────────────────────

def _paired_correct(a_rows: list[dict], b_rows: list[dict]) -> tuple[list[bool], list[bool]]:
    """Pair rows where at least one model makes a directional prediction."""
    a_c, b_c = [], []
    for ra, rb in zip(a_rows, b_rows):
        ret7 = ra.get(HORIZON) or rb.get(HORIZON)
        if ret7 is None: continue
        ret7 = float(ret7)
        ca = _is_correct(ra.get("signal","HOLD"), ret7)
        cb = _is_correct(rb.get("signal","HOLD"), ret7)
        if ca is None and cb is None: continue
        a_c.append(bool(ca))
        b_c.append(bool(cb))
    return a_c, b_c


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    files = {p.stem: load_jsonl(p) for p in PRED_DIR.glob("*_test.jsonl")}
    if not files:
        print(f"No JSONL files in {PRED_DIR}")
        return

    def _get(name: str) -> list[dict] | None:
        for stem, rows in files.items():
            if name in stem:
                return rows
        return None

    knn_returns = _get("knn_returns")
    cem_rag     = _get("cem_rag_test") or _get("cem_rag_full")
    fixed_knn   = _get("fixed_knn")
    no_factor   = _get("ablation_no_factor")
    rand_forest = _get("random_forest")
    buy_hold    = _get("buy_and_hold")
    patchtst    = _get("patchtst")

    results: dict = {}

    # ── Pair and test ──────────────────────────────────────────────────────────
    comparisons = [
        ("cem_rag_vs_knn_returns",    knn_returns, cem_rag,    "knn_returns vs cem_rag"),
        ("cem_rag_vs_fixed_knn",      fixed_knn,   cem_rag,    "fixed_knn vs cem_rag"),
        ("ablation_no_factor_vs_full",no_factor,   cem_rag,    "no_factor_ablation vs cem_rag"),
        ("rf_vs_fixed_knn",           fixed_knn,   rand_forest,"fixed_knn vs random_forest"),
        ("rf_vs_buy_and_hold",        buy_hold,    rand_forest,"buy_and_hold vs random_forest"),
        ("rf_vs_patchtst",            patchtst,    rand_forest,"patchtst vs random_forest"),
        ("patchtst_vs_fixed_knn",     fixed_knn,   patchtst,   "fixed_knn vs patchtst"),
    ]

    print(f"\n{'Comparison':<40} {'McNemar p':>10}  {'boot CI DA':>22}  {'sig?':>6}")
    print("-" * 85)

    for key, a_rows, b_rows, label in comparisons:
        if a_rows is None or b_rows is None:
            print(f"  {label:<38}  SKIP (file missing)")
            continue

        a_al, b_al = _paired_align(a_rows, b_rows)
        if not a_al:
            print(f"  {label:<38}  SKIP (no overlapping dates)")
            continue

        a_corr, b_corr = _paired_correct(a_al, b_al)
        mc = mcnemar_exact(a_corr, b_corr)

        a_da = [float(c) for c in a_corr]
        b_da = [float(c) for c in b_corr]
        boot_da = block_bootstrap_ci(a_da, b_da, block_size=7, samples=5000)

        a_ret = np.asarray(_returns_after_cost(a_al))
        b_ret = np.asarray(_returns_after_cost(b_al))
        boot_sharpe = block_bootstrap_ci(a_ret, b_ret, block_size=7, samples=5000)

        results[key] = {
            "label": label,
            "n_paired": len(a_al),
            "a_da":  round(_da(a_al), 4),
            "b_da":  round(_da(b_al), 4),
            "a_sharpe": round(_sharpe_series(_returns_after_cost(a_al)), 4),
            "b_sharpe": round(_sharpe_series(_returns_after_cost(b_al)), 4),
            "mcnemar":    mc,
            "boot_da":    boot_da,
            "boot_sharpe": boot_sharpe,
        }

        sig = "✓" if mc["significant_0.10"] or boot_da["delta_positive"] else "—"
        print(
            f"  {label:<38}  p={mc['p_value']:.4f}    "
            f"[{boot_da['ci_95_low']:+.4f}, {boot_da['ci_95_high']:+.4f}]   {sig}"
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    summary: dict = {}
    for name, rows in sorted(files.items()):
        summary[name] = {
            "da":       round(_da(rows), 4),
            "n":        len(rows),
            "coverage": round(sum(1 for r in rows if r.get("signal","HOLD") != "HOLD") / max(1, len(rows)), 4),
        }

    out = {"comparisons": results, "model_summary": summary}
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote → {OUT_PATH}")


if __name__ == "__main__":
    main()
