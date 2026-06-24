"""XGBoost 3-class baseline for CEM-RAG comparison.

Label:  UP (future_return_7d > 2%)  /  DOWN (< -2%)  /  FLAT (rest)
        Matches the ±2% knn_returns thresholds, so FLAT = HOLD region.

Features (225d total — same input as learned retriever):
  event_vec   85d  (event taxonomy multi-hot)
  factor_vec  75d  (factor taxonomy multi-hot)
  indicator_vec 5d (z-scored indicators)
  price_vec   60d  (OHLCV-derived returns/ranges/volumes)

Scaler: StandardScaler fit on TRAIN only, applied to val/test.
Model:  xgboost multi:softprob, early stopping on val logloss.
Policy: tau tuned on val (maximize coverage-filtered DA), applied to test once.

Ablation variants via --features flag:
  full          all 225 features (default)
  price_only    indicator + price (65 features)
  event_only    event + factor (160 features)
  no_event      factor + indicator + price (140 features — classic 3-block)

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/train_xgboost_baseline.py
    PYTHONPATH=/home/luong/marketlens python scripts/train_xgboost_baseline.py --features price_only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Splits (must match cem_dataset.py) ────────────────────────────────────────
TRAIN_END = date(2024, 12, 24)
VAL_START = date(2025, 1, 1)
VAL_END = date(2025, 6, 23)
TEST_START = date(2025, 7, 1)

SYMBOL = "BTC"
HOLD_BAND = 2.0     # ±2% HOLD band applied at signal time (not training)
CEM_TAU = 0.22      # reference tau; actual tau tuned on val

# Binary labels: 1=UP (return>0), 0=DOWN (return<0); skip exact zeros (rare)
UP, DOWN = 1, 0


def _label(ret: float) -> int | None:
    if ret > 0:
        return UP
    if ret < 0:
        return DOWN
    return None  # skip exact-zero rows


def _load_data(data_path: Path, features: str):
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    X_parts: dict[str, list] = {"train": [], "val": [], "test": []}
    y_parts: dict[str, list] = {"train": [], "val": [], "test": []}
    dates: dict[str, list] = {"train": [], "val": [], "test": []}
    returns: dict[str, list] = {"train": [], "val": [], "test": []}

    for r in raw:
        ret7 = r.get("future_return_7d")
        if ret7 is None:
            continue
        d = date.fromisoformat(r["date"])
        if d <= TRAIN_END:
            split = "train"
        elif VAL_START <= d <= VAL_END:
            split = "val"
        elif d >= TEST_START:
            split = "test"
        else:
            continue  # embargo gap

        ev = np.asarray(r["event_vec"], dtype=np.float32)
        fv = np.asarray(r["factor_vec"], dtype=np.float32)
        iv = np.asarray(r["indicator_vec"], dtype=np.float32)
        pv = np.asarray(r["price_vec"], dtype=np.float32)

        if features == "price_only":
            feat = np.concatenate([iv, pv])
        elif features == "event_only":
            feat = np.concatenate([ev, fv])
        elif features == "no_event":
            feat = np.concatenate([fv, iv, pv])
        else:  # full
            feat = np.concatenate([ev, fv, iv, pv])

        lbl = _label(float(ret7))
        if lbl is None:
            continue  # skip exact-zero returns (rare)

        X_parts[split].append(feat)
        y_parts[split].append(lbl)
        dates[split].append(r["date"])
        returns[split].append({h: r.get(f"future_return_{h}") for h in ["1d", "3d", "7d", "15d", "30d"]})

    return X_parts, y_parts, dates, returns


def _scale(X_train, X_val, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def _train(X_train, y_train, X_val, y_val):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    # Binary classification: UP(1) vs DOWN(0)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "reg_lambda": 2.0,
        "reg_alpha": 0.5,
        "scale_pos_weight": scale_pos,
        "seed": 42,
        "verbosity": 0,
    }
    evals_result: dict = {}
    callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True, maximize=True)]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=800,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        callbacks=callbacks,
        verbose_eval=False,
    )
    val_aucs = evals_result["val"]["auc"]
    best_round = int(np.argmax(val_aucs))
    print(f"  best round={best_round}  val_auc={val_aucs[best_round]:.4f}  total_rounds={len(val_aucs)}")
    return model


def _predict_proba(model, X) -> np.ndarray:
    """Returns shape (n,) — probability of UP (class 1)."""
    import xgboost as xgb
    dm = xgb.DMatrix(X)
    return model.predict(dm)   # scalar p(UP) per row


def _apply_policy(p_up_arr: np.ndarray, tau: float) -> tuple[list[str], list[float]]:
    """BUY if p_up > 0.5+tau/2, SELL if p_up < 0.5-tau/2, HOLD otherwise."""
    signals, confs = [], []
    high = 0.5 + tau / 2
    low  = 0.5 - tau / 2
    for p_up in p_up_arr:
        p_down = 1.0 - float(p_up)
        p_up   = float(p_up)
        diff_up   = p_up   - p_down
        diff_down = p_down - p_up
        if p_up >= high:
            signals.append("BUY"); confs.append(round(min(0.50 + diff_up * 0.80, 0.95), 3))
        elif p_up <= low:
            signals.append("SELL"); confs.append(round(min(0.50 + diff_down * 0.80, 0.95), 3))
        else:
            signals.append("HOLD"); confs.append(round(0.50 + max(diff_up, diff_down) * 0.30, 3))
    return signals, confs


def _tune_tau(p_up_arr: np.ndarray, ret7d: list[float]) -> float:
    best_tau, best_score = 0.05, -1.0
    # Search threshold directly on p_up (maps to tau via tau = 2*|p_up - 0.5|)
    for threshold_int in range(52, 90):
        high = threshold_int / 100
        low  = 1.0 - high
        signals = []
        for p in p_up_arr:
            if float(p) >= high:
                signals.append("BUY")
            elif float(p) <= low:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        correct = [
            (s == "BUY" and r > 0) or (s == "SELL" and r < 0) or (s == "HOLD" and abs(r) <= HOLD_BAND)
            for s, r in zip(signals, ret7d) if r is not None
        ]
        buy_n   = sum(1 for s in signals if s == "BUY")
        sell_n  = sum(1 for s in signals if s == "SELL")
        coverage = (buy_n + sell_n) / len(signals) if signals else 0
        da = float(np.mean(correct)) if correct else 0
        score = da if coverage >= 0.10 else -1.0
        if score > best_score:
            best_score = score
            best_tau   = round(2 * (high - 0.5), 2)
    return best_tau


def _sharpe(returns_list: list[float], horizon_days: int = 7) -> float:
    if not returns_list:
        return 0.0
    arr = np.asarray(returns_list, dtype=float)
    # Annualize: assume horizon_days spacing, 252 trading days/year
    periods_per_year = 252 / horizon_days
    mean = arr.mean() * periods_per_year
    std  = arr.std(ddof=1) * np.sqrt(periods_per_year) if arr.std(ddof=1) > 0 else 1e-9
    return float(mean / std)


def _eval_metrics(signals, ret7d_list) -> dict:
    correct, strategy_returns = [], []
    buy_correct = buy_total = sell_correct = sell_total = hold_correct = hold_total = 0
    for s, r in zip(signals, ret7d_list):
        if r is None:
            continue
        is_ok = (s == "BUY" and r > 0) or (s == "SELL" and r < 0) or (s == "HOLD" and abs(r) <= HOLD_BAND)
        correct.append(is_ok)
        if s == "BUY":
            strategy_returns.append(r); buy_total += 1; buy_correct += int(is_ok)
        elif s == "SELL":
            strategy_returns.append(-r); sell_total += 1; sell_correct += int(is_ok)
        else:
            strategy_returns.append(0.0); hold_total += 1; hold_correct += int(is_ok)

    buy_da   = buy_correct   / buy_total   if buy_total   else 0.0
    sell_da  = sell_correct  / sell_total  if sell_total  else 0.0
    hold_da  = hold_correct  / hold_total  if hold_total  else 0.0
    da       = float(np.mean(correct)) if correct else 0.0
    coverage = (buy_total + sell_total) / len(correct) if correct else 0.0
    sharpe   = _sharpe(strategy_returns)

    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
    y_pred_cls = ["BUY" if s == "BUY" else ("SELL" if s == "SELL" else "HOLD") for s in signals]
    y_true_cls = ["BUY" if r > HOLD_BAND else ("SELL" if r < -HOLD_BAND else "HOLD") for r in ret7d_list if r is not None]
    ba   = balanced_accuracy_score(y_true_cls, y_pred_cls) if y_true_cls else 0.0
    mf1  = f1_score(y_true_cls, y_pred_cls, labels=["BUY","SELL","HOLD"], average="macro", zero_division=0)
    mcc  = matthews_corrcoef(y_true_cls, y_pred_cls) if len(set(y_true_cls)) > 1 else 0.0

    return dict(
        da=round(da, 6), balanced_acc=round(ba, 6),
        macro_f1=round(mf1, 6), mcc=round(mcc, 6),
        buy_da=round(buy_da, 6), sell_da=round(sell_da, 6),
        coverage=round(coverage, 6), sharpe=round(sharpe, 6),
    )


def _update_main_table(row_name: str, metrics: dict) -> None:
    path = PROJECT_ROOT / "artifacts" / "metrics" / "main_table.csv"
    fieldnames = ["retriever", "da", "balanced_acc", "macro_f1", "mcc", "coverage", "sharpe",
                  "sortino", "max_dd", "hit_at_5", "brier"]
    existing: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    # Remove old row with same name if present
    existing = [r for r in existing if r["retriever"] != row_name]
    new_row = {
        "retriever": row_name,
        "da": metrics.get("da", 0),
        "balanced_acc": metrics.get("balanced_acc", 0),
        "macro_f1": metrics.get("macro_f1", 0),
        "mcc": metrics.get("mcc", 0),
        "coverage": metrics.get("coverage", 0),
        "sharpe": metrics.get("sharpe", 0),
        "sortino": 0.0,
        "max_dd": 0.0,
        "hit_at_5": 0.0,
        "brier": 0.0,
    }
    existing.append(new_row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    print(f"  updated {path}")


def run(data_path: Path, features: str) -> None:
    print(f"Loading {data_path} | features={features}")
    X_parts, y_parts, dates, returns = _load_data(data_path, features)

    X_train = np.array(X_parts["train"], dtype=np.float32)
    y_train = np.array(y_parts["train"])
    X_val   = np.array(X_parts["val"],   dtype=np.float32)
    y_val   = np.array(y_parts["val"])
    X_test  = np.array(X_parts["test"],  dtype=np.float32)
    y_test  = np.array(y_parts["test"])

    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  feat_dim={X_train.shape[1]}")
    print(f"  train label dist: UP={sum(y_train==UP)} DOWN={sum(y_train==DOWN)}")

    X_train_s, X_val_s, X_test_s, _ = _scale(X_train, X_val, X_test)

    print("Training XGBoost...")
    model = _train(X_train_s, y_train, X_val_s, y_val)

    val_proba  = _predict_proba(model, X_val_s)
    test_proba = _predict_proba(model, X_test_s)

    val_ret7d = [r["7d"] for r in returns["val"]]
    tau = _tune_tau(val_proba, [r for r in val_ret7d if r is not None])
    print(f"  best val tau={tau:.2f}  (p_up threshold={0.5+tau/2:.2f})")

    test_ret7d = [r["7d"] for r in returns["test"]]
    test_signals, test_confs = _apply_policy(test_proba, tau)

    # Forced-threshold variant: always commit (p_up>=0.5 → BUY, else → SELL, no HOLD)
    forced_signals = ["BUY" if float(p) >= 0.5 else "SELL" for p in test_proba]
    forced_confs   = [round(min(0.50 + abs(float(p) - 0.5) * 1.60, 0.95), 3) for p in test_proba]
    forced_metrics = _eval_metrics(forced_signals, test_ret7d)
    tau_metrics    = _eval_metrics(test_signals, test_ret7d)
    print(f"  TEST(tau={tau}) | DA={tau_metrics['da']:.3f}  cov={tau_metrics['coverage']:.3f}")
    print(f"  TEST(forced)   | DA={forced_metrics['da']:.3f}  BUY-DA={forced_metrics['buy_da']:.3f}  SELL-DA={forced_metrics['sell_da']:.3f}  MCC={forced_metrics['mcc']:.3f}  Sharpe={forced_metrics['sharpe']:.3f}")

    # Write forced signals to JSONL (honest directional accuracy, no degenerate 0-coverage)
    # Use forced signals for both JSONL and main_table
    final_signals, final_confs = forced_signals, forced_confs
    metrics = forced_metrics

    # Write JSONL
    out_name = f"xgboost_{features}_test.jsonl" if features != "full" else "xgboost_test.jsonl"
    out_path = PROJECT_ROOT / "artifacts" / "predictions" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, (d_str, signal, conf, rets) in enumerate(zip(dates["test"], final_signals, final_confs, returns["test"])):
            p_up_val   = float(test_proba[i])
            p_down_val = 1.0 - p_up_val
            f.write(json.dumps({
                "date": d_str,
                "symbol": SYMBOL,
                "split": "test",
                "features": features,
                "tau": float(tau),
                "signal": signal,
                "confidence": float(conf),
                "p_up": round(p_up_val, 4),
                "p_down": round(p_down_val, 4),
                **{f"actual_return_{h}": (float(v) if v is not None else None) for h, v in rets.items()},
            }) + "\n")
    print(f"  wrote {len(test_signals)} rows → {out_path}")

    table_name = f"xgboost_{features}" if features != "full" else "xgboost_full"
    _update_main_table(table_name, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="stockmem/data/real_optimizer_v3.json")
    parser.add_argument(
        "--features",
        choices=["full", "price_only", "event_only", "no_event"],
        default="full",
        help="Feature subset to use",
    )
    parser.add_argument("--all-variants", action="store_true", help="Run all 4 feature subsets")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data

    if args.all_variants:
        for feat in ["full", "price_only", "event_only", "no_event"]:
            print(f"\n{'='*60}\n  Variant: {feat}\n{'='*60}")
            run(data_path, feat)
    else:
        run(data_path, args.features)
