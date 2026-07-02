"""train_prob_model.py — Calibrated logistic regression + random forest (Bước 8).

Feature vector X: concat([event_vec(85), factor_vec(75), indicator_vec(5), price_vec(60)]) = 225d
Label y: 1 if future_return_7d > 0, 0 if < 0, skip exact zeros

Splits (from cem_dataset.label_rows):
  train: split == "train"  (≤ 2024-12-24)
  val:   split == "val"    (2025-01-01 to 2025-06-23)
  test:  split == "test"   (2025-07-01 to 2026-05-01)

Models:
  1. LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
  2. RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)

Calibration: CalibratedClassifierCV(cv='prefit', method='isotonic') fit on val set.

Tau tuning on val: threshold t in [0.52, 0.80], maximize SELL-DA with coverage >= 5%.
  BUY if p_up >= t, SELL if p_down >= t, else HOLD

Outputs:
  artifacts/predictions/logistic_regression_test.jsonl
  artifacts/predictions/random_forest_test.jsonl
  artifacts/metrics/main_table.csv  (appended)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stockmem.scripts.cem_dataset import label_rows
from stockmem.scripts.optimize_weights import load_rows

DATA_PATH = PROJECT_ROOT / "stockmem" / "data" / "real_optimizer_v3.json"
PRED_DIR = PROJECT_ROOT / "artifacts" / "predictions"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

SYMBOL = "BTC"
HOLD_BAND = 2.0  # ±2% band for is_correct check (consistent with xgboost baseline)
TRANSACTION_COST_PCT = 0.10  # 10 basis points per trade (round-trip)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits():
    # Load raw JSON to get all return horizons (3d, 15d not in Row dataclass)
    import json as _json
    raw_by_date: dict[str, dict] = {}
    for item in _json.loads(DATA_PATH.read_text(encoding="utf-8")):
        raw_by_date[str(item["date"])] = item

    rows = load_rows(DATA_PATH)
    labeled = label_rows(rows, band="0.5sigma")

    X: dict[str, list] = {"train": [], "val": [], "test": []}
    y: dict[str, list] = {"train": [], "val": [], "test": []}
    dates: dict[str, list] = {"train": [], "val": [], "test": []}
    rets: dict[str, list] = {"train": [], "val": [], "test": []}

    for lb in labeled:
        split = lb.split
        if split not in X:
            continue  # embargo gap

        ret7 = lb.row.future_return_7d
        if ret7 == 0.0:
            continue  # skip exact zeros

        lbl = 1 if ret7 > 0 else 0

        # 225d feature: event_vec(85) + factor_vec(75) + indicator_vec(5) + price_vec(60)
        ev = lb.row.event_vec
        if ev.size == 0:
            ev = np.zeros(85, dtype=np.float32)
        feat = np.concatenate([
            ev.astype(np.float32),
            lb.row.factor_vec.astype(np.float32),
            lb.row.indicator_vec.astype(np.float32),
            lb.row.price_vec.astype(np.float32),
        ])

        raw = raw_by_date.get(lb.row.date, {})
        X[split].append(feat)
        y[split].append(lbl)
        dates[split].append(lb.row.date)
        rets[split].append({
            "1d":  float(raw.get("future_return_1d") or lb.row.future_return_1d or 0.0),
            "3d":  float(raw.get("future_return_3d") or 0.0),
            "7d":  float(raw.get("future_return_7d") or lb.row.future_return_7d or 0.0),
            "15d": float(raw.get("future_return_15d") or 0.0),
            "30d": float(raw.get("future_return_30d") or lb.row.future_return_30d or 0.0),
        })

    return X, y, dates, rets


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

def scale_splits(X_parts):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = np.array(X_parts["train"], dtype=np.float32)
    X_val   = np.array(X_parts["val"],   dtype=np.float32)
    X_test  = np.array(X_parts["test"],  dtype=np.float32)

    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s


# ---------------------------------------------------------------------------
# Training + calibration
# ---------------------------------------------------------------------------

def train_and_calibrate(model_name: str, base_model, X_train_s, y_train, X_val_s, y_val):
    """Train base model on train set, calibrate via isotonic regression on val set.

    sklearn >= 1.2 removed cv='prefit'. We replicate it by:
      1. Fit base model on train.
      2. Get raw probabilities on val from base model.
      3. Fit IsotonicRegression directly on those val probabilities.
      4. Wrap as a simple calibrated predictor.
    """
    from sklearn.metrics import roc_auc_score

    print(f"\n[{model_name}] Training base model ...")
    base_model.fit(X_train_s, y_train)

    val_proba_raw = base_model.predict_proba(X_val_s)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba_raw)
    print(f"  val AUC (pre-calibration): {val_auc:.4f}")

    print(f"[{model_name}] Calibrating on val set (Platt/sigmoid, prefit) ...")
    # Use Platt scaling (sigmoid) for small val sets — isotonic overfits with <200 samples.
    # We replicate cv='prefit' manually: fit sigmoid on val predictions from the pre-trained base.
    from sklearn.linear_model import LogisticRegression as _LR
    p_reshaped = val_proba_raw.reshape(-1, 1)
    platt = _LR(C=1e10, solver="lbfgs", max_iter=200)
    platt.fit(p_reshaped, y_val)

    class _CalibratedModel:
        def __init__(self, base, calibrator):
            self._base = base
            self._cal = calibrator

        def predict_proba(self, X):
            p_raw = self._base.predict_proba(X)[:, 1].reshape(-1, 1)
            p_cal = self._cal.predict_proba(p_raw)[:, 1]
            p_cal = np.clip(p_cal, 0.0, 1.0)
            return np.column_stack([1.0 - p_cal, p_cal])

    calibrated = _CalibratedModel(base_model, platt)

    val_proba_cal = calibrated.predict_proba(X_val_s)[:, 1]
    val_auc_cal = roc_auc_score(y_val, val_proba_cal)
    print(f"  val AUC (post-calibration): {val_auc_cal:.4f}")

    return calibrated, val_auc_cal


# ---------------------------------------------------------------------------
# Tau tuning: maximize SELL-DA on val, coverage >= 5%
# ---------------------------------------------------------------------------

def tune_tau(p_up_arr: np.ndarray, y_val: np.ndarray, ret7d_val: list[float]) -> tuple[float, float]:
    """
    Search threshold t in [0.52, 0.80].
    BUY if p_up >= t, SELL if (1-p_up) >= t, else HOLD.
    Maximize SELL-DA with coverage >= 5%.
    Returns (best_tau, best_sell_da).
    """
    best_t = 0.52
    best_sell_da = -1.0
    n = len(p_up_arr)

    for ti in range(52, 81):
        t = ti / 100.0
        sell_correct = sell_total = 0
        buy_sell_total = 0
        for p_up, r in zip(p_up_arr, ret7d_val):
            p_down = 1.0 - float(p_up)
            p_up_f = float(p_up)
            if p_up_f >= t:
                buy_sell_total += 1
            elif p_down >= t:
                buy_sell_total += 1
                sell_total += 1
                if r < 0:
                    sell_correct += 1

        coverage = buy_sell_total / n if n > 0 else 0.0
        if coverage < 0.05:
            continue
        sell_da = sell_correct / sell_total if sell_total > 0 else 0.0
        if sell_da > best_sell_da:
            best_sell_da = sell_da
            best_t = t

    return best_t, best_sell_da


# ---------------------------------------------------------------------------
# Signal application
# ---------------------------------------------------------------------------

def apply_policy(p_up_arr: np.ndarray, t: float) -> tuple[list[str], list[float], list[float], list[float]]:
    """Returns (signals, confidences, p_up_list, p_down_list)."""
    signals, confs, p_ups, p_downs = [], [], [], []
    for p_up in p_up_arr:
        p_up_f = float(p_up)
        p_down_f = 1.0 - p_up_f
        if p_up_f >= t:
            signals.append("BUY")
            confs.append(round(p_up_f, 4))
        elif p_down_f >= t:
            signals.append("SELL")
            confs.append(round(p_down_f, 4))
        else:
            signals.append("HOLD")
            confs.append(round(max(p_up_f, p_down_f), 4))
        p_ups.append(round(p_up_f, 4))
        p_downs.append(round(p_down_f, 4))
    return signals, confs, p_ups, p_downs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sortino(returns: list[float]) -> float:
    arr = np.array(returns, dtype=np.float64)
    downside = arr[arr < 0]
    if len(downside) < 2:
        return 0.0
    return float(np.mean(arr)) * np.sqrt(52) / (float(np.std(downside, ddof=1)) + 1e-12)


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    cum = np.cumprod(1 + np.array(returns, dtype=np.float64) / 100)
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / (roll_max + 1e-12)
    return float(dd.min())


def _macro_f1(predictions: list[str], actuals: list[str]) -> float:
    classes = ["BUY", "SELL", "HOLD"]
    f1s = []
    for cls in classes:
        tp = sum(p == cls and a == cls for p, a in zip(predictions, actuals))
        fp = sum(p == cls and a != cls for p, a in zip(predictions, actuals))
        fn = sum(p != cls and a == cls for p, a in zip(predictions, actuals))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def _mcc_multiclass(predictions: list[str], actuals: list[str]) -> float:
    classes = ["BUY", "SELL", "HOLD"]
    class_idx = {c: i for i, c in enumerate(classes)}
    conf = np.zeros((3, 3), dtype=np.float64)
    for p, a in zip(predictions, actuals):
        if p in class_idx and a in class_idx:
            conf[class_idx[a], class_idx[p]] += 1
    t = conf.sum()
    if t == 0:
        return 0.0
    t_k = conf.sum(axis=1)
    p_k = conf.sum(axis=0)
    c = np.trace(conf)
    numerator = c * t - float(np.dot(t_k, p_k))
    denom_sq = (t**2 - float(np.dot(p_k, p_k))) * (t**2 - float(np.dot(t_k, t_k)))
    if denom_sq <= 0:
        return 0.0
    return float(numerator / np.sqrt(denom_sq))


def _sharpe_with_cost(returns: list[float], signals: list[str], cost_pct: float = TRANSACTION_COST_PCT) -> float:
    """Annualized Sharpe with transaction cost deducted for BUY/SELL trades."""
    if not returns:
        return 0.0
    net = []
    for r, s in zip(returns, signals):
        if s in ("BUY", "SELL"):
            net.append(r - cost_pct)
        else:
            net.append(r)
    arr = np.asarray(net, dtype=np.float64)
    # Weekly rebalancing → 52 periods/year
    std = arr.std(ddof=1)
    if std < 1e-9:
        return 0.0
    return float(arr.mean() * np.sqrt(52) / std)


def compute_metrics(
    signals: list[str],
    ret7d: list[float],
    p_up_arr: np.ndarray,
) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss

    predictions = signals
    actuals_cls = [
        "BUY" if r > HOLD_BAND else ("SELL" if r < -HOLD_BAND else "HOLD")
        for r in ret7d
    ]

    correct = []
    strategy_returns = []
    buy_correct = buy_total = sell_correct = sell_total = hold_correct = hold_total = 0

    for s, r in zip(signals, ret7d):
        is_ok = (
            (s == "BUY"  and r > 0)
            or (s == "SELL" and r < 0)
            or (s == "HOLD" and abs(r) <= HOLD_BAND)
        )
        correct.append(is_ok)
        if s == "BUY":
            strategy_returns.append(r)
            buy_total += 1; buy_correct += int(is_ok)
        elif s == "SELL":
            strategy_returns.append(-r)
            sell_total += 1; sell_correct += int(is_ok)
        else:
            strategy_returns.append(0.0)
            hold_total += 1; hold_correct += int(is_ok)

    n = len(correct)
    da = float(np.mean(correct)) if correct else 0.0
    coverage = (buy_total + sell_total) / n if n > 0 else 0.0
    buy_da  = buy_correct  / buy_total  if buy_total  else 0.0
    sell_da = sell_correct / sell_total if sell_total else 0.0
    hold_da = hold_correct / hold_total if hold_total else 0.0
    balanced_acc = (buy_da + sell_da + hold_da) / 3.0
    macro_f1 = _macro_f1(predictions, actuals_cls)
    mcc = _mcc_multiclass(predictions, actuals_cls)
    sharpe = _sharpe_with_cost(strategy_returns, signals)
    sortino = _sortino(strategy_returns)
    max_dd = _max_drawdown(strategy_returns)

    # Brier score: binary p_up vs actual direction (1=up, 0=down)
    y_binary = [1 if r > 0 else 0 for r in ret7d]
    brier = float(brier_score_loss(y_binary, p_up_arr)) if len(set(y_binary)) > 1 else 0.0

    return dict(
        da=round(da, 6),
        balanced_acc=round(balanced_acc, 6),
        macro_f1=round(macro_f1, 6),
        mcc=round(mcc, 6),
        coverage=round(coverage, 6),
        sharpe=round(sharpe, 6),
        sortino=round(sortino, 6),
        max_dd=round(max_dd, 6),
        hit_at_5=0.0,
        brier=round(brier, 6),
        buy_da=round(buy_da, 6),
        sell_da=round(sell_da, 6),
        hold_da=round(hold_da, 6),
    )


def compute_val_sell_da(signals: list[str], ret7d: list[float]) -> float:
    sell_correct = sell_total = 0
    for s, r in zip(signals, ret7d):
        if s == "SELL":
            sell_total += 1
            if r < 0:
                sell_correct += 1
    return sell_correct / sell_total if sell_total > 0 else 0.0


def compute_val_auc(model, X_val_s, y_val) -> float:
    from sklearn.metrics import roc_auc_score
    p = model.predict_proba(X_val_s)[:, 1]
    return float(roc_auc_score(y_val, p))


# ---------------------------------------------------------------------------
# Write JSONL predictions
# ---------------------------------------------------------------------------

def write_jsonl(
    out_path: Path,
    dates_test: list[str],
    signals: list[str],
    confs: list[float],
    p_ups: list[float],
    p_downs: list[float],
    rets_test: list[dict],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d_str, sig, conf, p_up, p_down, r in zip(dates_test, signals, confs, p_ups, p_downs, rets_test):
            p_hold = round(1.0 - p_up - p_down, 4)  # always 0 for binary
            row = {
                "date": d_str,
                "symbol": SYMBOL,
                "signal": sig,
                "confidence": conf,
                "p_up": p_up,
                "p_down": p_down,
                "p_hold": max(0.0, p_hold),
                "actual_return_1d":  r.get("1d"),
                "actual_return_3d":  r.get("3d"),
                "actual_return_7d":  r.get("7d"),
                "actual_return_15d": r.get("15d"),
                "actual_return_30d": r.get("30d"),
            }
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(dates_test)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Update main_table.csv
# ---------------------------------------------------------------------------

def update_main_table(row_name: str, metrics: dict) -> None:
    path = METRICS_DIR / "main_table.csv"
    fieldnames = ["retriever", "da", "balanced_acc", "macro_f1", "mcc",
                  "coverage", "sharpe", "sortino", "max_dd", "hit_at_5", "brier"]
    existing: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if r["retriever"] != row_name]
    new_row = {
        "retriever": row_name,
        "da":            metrics.get("da", 0.0),
        "balanced_acc":  metrics.get("balanced_acc", 0.0),
        "macro_f1":      metrics.get("macro_f1", 0.0),
        "mcc":           metrics.get("mcc", 0.0),
        "coverage":      metrics.get("coverage", 0.0),
        "sharpe":        metrics.get("sharpe", 0.0),
        "sortino":       metrics.get("sortino", 0.0),
        "max_dd":        metrics.get("max_dd", 0.0),
        "hit_at_5":      metrics.get("hit_at_5", 0.0),
        "brier":         metrics.get("brier", 0.0),
    }
    existing.append(new_row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    print(f"  updated {path}")


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

COLS = ["val_auc", "val_sell_da", "test_sell_da", "test_da", "coverage", "mcc", "sharpe", "brier"]

def _print_summary(summary: dict[str, dict]) -> None:
    header = f"{'model':<26} {'val_auc':>8} {'val_SELL_DA':>11} {'test_SELL_DA':>12} {'test_DA':>8} {'cov':>7} {'mcc':>7} {'sharpe':>8} {'brier':>7}"
    print()
    print("=" * len(header))
    print("SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in summary.items():
        print(
            f"{name:<26} "
            f"{m['val_auc']:>8.4f} "
            f"{m['val_sell_da']:>11.4f} "
            f"{m['test_sell_da']:>12.4f} "
            f"{m['test_da']:>8.4f} "
            f"{m['coverage']:>7.4f} "
            f"{m['mcc']:>7.4f} "
            f"{m['sharpe']:>8.4f} "
            f"{m['brier']:>7.4f}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    print("Loading data ...")
    X_parts, y_parts, dates, rets = load_splits()

    X_train_s, X_val_s, X_test_s = scale_splits(X_parts)
    y_train = np.array(y_parts["train"])
    y_val   = np.array(y_parts["val"])
    y_test  = np.array(y_parts["test"])
    ret7d_val  = [r["7d"] for r in rets["val"]]
    ret7d_test = [r["7d"] for r in rets["test"]]

    print(f"  train={len(X_train_s)}  val={len(X_val_s)}  test={len(X_test_s)}  feat_dim={X_train_s.shape[1]}")
    print(f"  train label dist: UP={sum(y_train==1)}  DOWN={sum(y_train==0)}")
    print(f"  val   label dist: UP={sum(y_val==1)}  DOWN={sum(y_val==0)}")
    print(f"  test  label dist: UP={sum(y_test==1)}  DOWN={sum(y_test==0)}")

    model_defs = [
        (
            "logistic_regression",
            LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42),
        ),
    ]

    summary: dict[str, dict] = {}

    for model_name, base_model in model_defs:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        calibrated, val_auc = train_and_calibrate(
            model_name, base_model, X_train_s, y_train, X_val_s, y_val
        )

        # Val predictions for tau tuning
        val_proba = calibrated.predict_proba(X_val_s)[:, 1]

        print(f"[{model_name}] Tuning tau on val (maximize SELL-DA, coverage>=5%) ...")
        best_t, best_sell_da_val = tune_tau(val_proba, y_val, ret7d_val)
        val_signals, _, _, _ = apply_policy(val_proba, best_t)
        val_sell_da = compute_val_sell_da(val_signals, ret7d_val)
        print(f"  best_t={best_t:.2f}  val_SELL_DA={val_sell_da:.4f}")

        # Test predictions
        test_proba = calibrated.predict_proba(X_test_s)[:, 1]
        test_signals, test_confs, test_p_ups, test_p_downs = apply_policy(test_proba, best_t)

        test_metrics = compute_metrics(test_signals, ret7d_test, test_proba)
        print(f"[{model_name}] Test metrics:")
        print(f"  SELL_DA={test_metrics['sell_da']:.4f}  DA={test_metrics['da']:.4f}  "
              f"coverage={test_metrics['coverage']:.4f}  MCC={test_metrics['mcc']:.4f}  "
              f"Sharpe={test_metrics['sharpe']:.4f}")

        # Write JSONL
        out_path = PRED_DIR / f"{model_name}_test.jsonl"
        write_jsonl(
            out_path,
            dates["test"],
            test_signals,
            test_confs,
            test_p_ups,
            test_p_downs,
            rets["test"],
        )

        # Update main_table.csv
        update_main_table(model_name, test_metrics)

        summary[model_name] = {
            "val_auc":    val_auc,
            "val_sell_da": val_sell_da,
            "test_sell_da": test_metrics["sell_da"],
            "test_da":    test_metrics["da"],
            "coverage":   test_metrics["coverage"],
            "mcc":        test_metrics["mcc"],
            "sharpe":     test_metrics["sharpe"],
            "brier":      test_metrics["brier"],
        }

    _print_summary(summary)


if __name__ == "__main__":
    main()
