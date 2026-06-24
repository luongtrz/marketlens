"""Compute calibration metrics for all prediction files.

For each model's BUY/SELL signals:
  - Brier score: mean((confidence - y_true)^2), y_true=1 if correct direction
  - ECE (Expected Calibration Error): binned |avg_conf - avg_acc|, weighted by count
  - Reliability curve data: 10 bins over [0.5, 1.0]
  - Overconfidence gap: mean_conf - mean_accuracy

Models included: all *_test.jsonl in artifacts/predictions/ with directional signals.

Outputs:
  artifacts/metrics/calibration_table.csv
  artifacts/metrics/calibration_curves.json   (for reliability diagram)

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/compute_calibration_table.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR  = PROJECT_ROOT / "artifacts/predictions"
OUT_TABLE = PROJECT_ROOT / "artifacts/metrics/calibration_table.csv"
OUT_CURVES= PROJECT_ROOT / "artifacts/metrics/calibration_curves.json"

N_BINS = 10
BIN_EDGES = [0.5 + i * (0.5 / N_BINS) for i in range(N_BINS + 1)]  # 0.50 → 1.00


# ── Display order ───────────────────────────────────────────────────────────────
MODEL_ORDER = [
    "fixed_knn_test",
    "knn_returns_test",
    "cem_rag_test",
    "cem_rag_full_test",
    "xgboost_test",
    "xgboost_price_only_test",
    "xgboost_event_only_test",
    "xgboost_no_event_test",
    "ablation_no_event_test",
    "ablation_no_factor_test",
    "ablation_fixed_ret_test",
    "ablation_no_policy_test",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _y_true(signal: str, ret7: float) -> int | None:
    """1 if the directional signal was correct, 0 if wrong, None to skip."""
    if signal == "BUY":
        return 1 if ret7 > 0 else 0
    if signal == "SELL":
        return 1 if ret7 < 0 else 0
    return None


def calibration_metrics(rows: list[dict]) -> dict | None:
    """Compute calibration metrics for directional (BUY/SELL) rows only."""
    pairs: list[tuple[float, int]] = []  # (confidence, y_true)
    for r in rows:
        sig = r.get("signal", "HOLD")
        ret7 = r.get("actual_return_7d")
        conf = r.get("confidence")
        if sig == "HOLD" or ret7 is None or conf is None:
            continue
        y = _y_true(sig, float(ret7))
        if y is None:
            continue
        pairs.append((float(conf), y))

    if len(pairs) < 10:
        return None

    n = len(pairs)
    confs   = [p[0] for p in pairs]
    ys      = [p[1] for p in pairs]

    # Brier score
    brier = sum((c - y) ** 2 for c, y in pairs) / n

    # Mean confidence and accuracy
    mean_conf = sum(confs) / n
    mean_acc  = sum(ys)    / n

    # Binned calibration
    bins: list[list[tuple[float, int]]] = [[] for _ in range(N_BINS)]
    for c, y in pairs:
        idx = min(int((c - 0.5) / (0.5 / N_BINS)), N_BINS - 1)
        bins[idx].append((c, y))

    ece = 0.0
    curve: list[dict] = []
    for i, b in enumerate(bins):
        if not b:
            curve.append({
                "bin_lower": round(BIN_EDGES[i], 3),
                "bin_upper": round(BIN_EDGES[i + 1], 3),
                "avg_conf": None, "avg_acc": None, "count": 0,
            })
            continue
        avg_c = sum(x[0] for x in b) / len(b)
        avg_a = sum(x[1] for x in b) / len(b)
        ece  += (len(b) / n) * abs(avg_c - avg_a)
        curve.append({
            "bin_lower": round(BIN_EDGES[i], 3),
            "bin_upper": round(BIN_EDGES[i + 1], 3),
            "avg_conf": round(avg_c, 4),
            "avg_acc":  round(avg_a, 4),
            "count":    len(b),
        })

    # Reliability: fraction of bins where conf > acc (overconfident)
    active = [(b["avg_conf"], b["avg_acc"]) for b in curve
              if b["avg_conf"] is not None and b["count"] >= 3]
    overconf_bins = sum(1 for c, a in active if c > a)
    total_active  = len(active)

    return {
        "n_directional": n,
        "mean_conf":     round(mean_conf, 4),
        "mean_acc":      round(mean_acc, 4),
        "overconf_gap":  round(mean_conf - mean_acc, 4),
        "brier_score":   round(brier, 5),
        "ece":           round(ece, 5),
        "overconf_bins": overconf_bins,
        "total_active_bins": total_active,
        "curve":         curve,
    }


def main() -> None:
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(PRED_DIR.glob("*_test.jsonl"))
    if not pred_files:
        print(f"No prediction files in {PRED_DIR}")
        return

    # Sort by MODEL_ORDER, then alphabetical for anything not listed
    def sort_key(p: Path) -> tuple[int, str]:
        try:
            return (MODEL_ORDER.index(p.stem), p.stem)
        except ValueError:
            return (len(MODEL_ORDER), p.stem)

    pred_files = sorted(pred_files, key=sort_key)

    fieldnames = [
        "model", "n_directional",
        "mean_conf", "mean_acc", "overconf_gap",
        "brier_score", "ece",
        "overconf_bins", "total_active_bins",
    ]

    table_rows: list[dict] = []
    curves: dict[str, list] = {}

    print(f"\n{'Model':<35} {'n':>5}  {'mean_conf':>9}  {'mean_acc':>9}  {'gap':>7}  {'Brier':>7}  {'ECE':>7}")
    print("-" * 90)

    for path in pred_files:
        rows = _load_jsonl(path)
        m = calibration_metrics(rows)
        if m is None:
            print(f"  {path.stem:<35} — skipped (< 10 directional predictions)")
            continue

        curves[path.stem] = m.pop("curve")
        table_rows.append({"model": path.stem, **m})
        print(
            f"  {path.stem:<35} {m['n_directional']:>5}  "
            f"{m['mean_conf']:>9.4f}  {m['mean_acc']:>9.4f}  {m['overconf_gap']:>7.4f}  "
            f"{m['brier_score']:>7.5f}  {m['ece']:>7.5f}"
        )

    with OUT_TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    with OUT_CURVES.open("w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    print(f"\nWrote {len(table_rows)} rows → {OUT_TABLE}")
    print(f"Wrote reliability curves → {OUT_CURVES}")


if __name__ == "__main__":
    main()
