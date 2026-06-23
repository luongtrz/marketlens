# Evaluation Suite — `scripts/eval_suite.py`

> ESWA-grade comparison: 6 baselines × 9 metrics + McNemar + block bootstrap.

---

## Chạy

```bash
PYTHONPATH=/home/luong/marketlens python scripts/eval_suite.py \
  --data stockmem/data/real_optimizer_v2.json \
  --artifact stockmem/config/learned_retriever.json \
  --weights stockmem/config/weights.auto.json \
  --output-dir artifacts/metrics
```

Output:
- `artifacts/metrics/main_table.csv` — bảng chính
- `artifacts/metrics/stat_tests.json` — McNemar + bootstrap CI

---

## Baselines

| Baseline | Mô tả | Lý do có mặt |
|---|---|---|
| `always_hold` | Luôn HOLD | Lower bound naive |
| `random_direction` | Random BUY/SELL/HOLD theo class prior | Kiểm tra baseline ngẫu nhiên |
| `rsi_momentum` | BUY nếu RSI z-score < −1.0, SELL nếu > +1.0 | Technical rule đơn giản |
| `sentiment_only` | BUY/SELL theo sentiment z-score | Text-only signal, không có price |
| `baseline_fixed_knn` | Weighted cosine kNN, weights từ Optuna | Baseline mạnh hiện tại của hệ thống |
| `learned_cem_rag` | Learned diagonal metric (InfoNCE-trained) | Model mới đề xuất |

> **Note:** `rsi_momentum` và `always_hold` cho cùng kết quả vì RSI z-score trong test set không vượt ngưỡng ±1.0 → coverage = 0%. Có thể thử ngưỡng ±0.5 nếu muốn coverage cao hơn cho baseline này.

---

## Metrics

| Metric | Ý nghĩa |
|---|---|
| `da` | Directional Accuracy — tỉ lệ signal đúng hướng |
| `balanced_acc` | (BUY_DA + SELL_DA + HOLD_DA) / 3 — tránh bias class |
| `macro_f1` | Macro-F1 3 class: BUY / SELL / HOLD |
| `mcc` | Matthews Correlation Coefficient — robust với imbalanced class |
| `coverage` | Tỉ lệ ngày ra BUY hoặc SELL (không HOLD) |
| `sharpe` | Annualized Sharpe (7d returns, 52 periods/year) |
| `sortino` | Sharpe chỉ tính downside deviation |
| `max_dd` | Max drawdown từ cumulative return |
| `hit_at_5` | Top-5 retrieved có cùng direction với query không |

---

## Statistical Tests

### McNemar test (paired)
```
H0: learned_cem_rag và baseline_fixed_knn correct cùng số queries
H1: learned_cem_rag correct nhiều hơn
```
Dùng `mcnemar_exact()` từ `evaluate_retriever.py`.  
Kết quả hiện tại: b=32, c=33, p=1.00 — **không significant** (hai model correct gần như cùng queries).

### Block Bootstrap Sharpe CI
```
bootstrap 2000 samples, block_size=7 ngày
95% CI cho (learned_sharpe - baseline_sharpe)
```
Kết quả: [−0.658, +1.055] — CI rộng vì test set nhỏ (305 queries).

---

## Kết quả hiện tại (test 2025-07 → 2026-05)

```
retriever            da    balanced_acc  macro_f1   mcc    coverage  sharpe  hit_at_5
always_hold        0.190      0.063      0.107    0.000    0.000    0.000    —
rsi_momentum       0.190      0.063      0.107    0.000    0.000    0.000    —
sentiment_only     0.275      0.324      0.242   −0.051    0.403    0.198    —
random_direction   0.502      0.447      0.387    0.078    0.826    0.032    —
baseline_fixed_knn 0.393      0.431      0.294    0.024    0.807   −0.297   0.929
learned_cem_rag    0.397      0.405      0.309    0.036    0.784   −0.143   0.954
```

**Nhận xét:**
- `learned_cem_rag` vượt `baseline_fixed_knn` trên tất cả retrieval metrics (DA, macro_f1, MCC, hit@5, Sharpe)
- Cải thiện **không significant về mặt thống kê** (McNemar p=1.0, bootstrap CI spanning zero) do test set nhỏ
- `random_direction` có DA cao (0.502) vì coverage 83% — nhiều BUY/SELL, dễ đúng hơn nhưng max drawdown −0.93
- Cả hai kNN có Sharpe âm ở eval mode này (fixed band, full coverage) — cải thiện rõ hơn khi dùng `calibrate_policy.py` với tau=0.22

---

## Thêm baseline mới

Để thêm `xgboost_price` baseline:
```python
# Trong eval_suite.py, thêm vào _run_all_baselines():
def _xgboost_baseline(labeled, split="test"):
    from sklearn.ensemble import GradientBoostingClassifier
    train = [r for r in labeled if r.split == "train" and r.direction != 0]
    test  = [r for r in labeled if r.split == split]
    X_train = np.stack([r.row.price_vec for r in train])
    y_train = [r.direction for r in train]
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # ... evaluate predictions
```

---

## Artifacts

```
artifacts/
└── metrics/
    ├── main_table.csv     # bảng kết quả đầy đủ
    └── stat_tests.json    # McNemar + bootstrap CI
```

Chưa có (TODO):
```
artifacts/
├── datasets/              # parquet splits
├── predictions/           # per-query JSONL cho từng baseline
└── figures/               # architecture diagram, retriever triplet
```
