# kNN-Returns Backtest Results

**Config:** BTC · 2022-01-06 → 2026-05-09 · k=5 · threshold ±2% · eval horizon D+7d

---

## Config tái lập

```json
{
  "search_weights": { "w_factor": 0.5444, "w_indicator": 0.3091, "w_price": 0.1416 },
  "return_weights": { "1d": 0.40, "3d": 0.30, "7d": 0.15, "15d": 0.10, "30d": 0.05 },
  "buy_threshold_pct": 2.0,
  "sell_threshold_pct": 2.0,
  "k": 5,
  "eval_horizon": "7d"
}
```

```bash
# Reproduce
python scripts/eval_knn_returns.py --horizon 7d --buy-thr 2 --sell-thr 2 \
  --out data/backtests/knn_returns_best.json
# Requires: stockmem/config/weights.auto.json = New Bayesian (commit feat/new-strategy HEAD)
```

---

## Kết Quả Tổng Hợp (D+7d)

| Signal | Count | Share | DA | Avg actual D+7d |
|--------|------:|------:|---:|----------------:|
| BUY | 658 | 42.3% | **59.7%** | +4.46% |
| SELL | 247 | 15.9% | **57.5%** | −3.38% |
| HOLD | 651 | 41.8% | 13.2% | +2.96% |
| **Total** | **1,556** | **100%** | 39.9% | |

**Coverage (BUY+SELL): 58.2%** · **Correct predictions: 616/905 active signals**

---

## Breakdown Theo Năm

| Năm | BUY | BUY DA | SELL | SELL DA | HOLD |
|-----|----:|-------:|-----:|--------:|-----:|
| 2022 | 23 | 39% | 181 | **62%** | 128 |
| 2023 | 202 | **63%** | 21 | 33% | 142 |
| 2024 | 219 | **64%** | 12 | 33% | 135 |
| 2025 | 168 | 52% | 18 | **67%** | 179 |
| 2026 | 46 | **63%** | 15 | 40% | 67 |
| **Total** | **658** | **59.7%** | **247** | **57.5%** | **651** |

**Nhận xét theo năm:**
- **2022 (bear market):** SELL signals nhiều (181) và DA cao (62%) — model nhận dạng tốt downtrend
- **2023–2024 (recovery + bull):** BUY DA cao nhất (63–64%) — model bắt đúng uptrend dài
- **2025:** SELL DA 67% — model phát hiện được các đợt correction trong bull market
- **BUY DA thấp nhất 2022 (39%):** wanh hợp lý vì năm bear market, BUY signal nào cũng khó đúng

---

## So Sánh Với LLM Models (D+7d)

| Model | BUY DA | SELL DA | Coverage | Deterministic |
|-------|-------:|--------:|:--------:|:-------------:|
| kimi-k2.5 | ~51% | ~42% | 29% | ✗ |
| qwen3.5-plus | 52.9% | 37.4% | 41% | ✗ |
| deepseek-v4-flash | 52.4% | 41.8% | 47% | ✗ |
| kNN-returns (default weights) | 59.6% | 54.1% | 62.5% | ✓ |
| **kNN-returns (New Bayesian)** | **59.7%** | **57.5%** | **58.2%** | **✓** |

**kNN-returns New Bayesian vs best LLM (deepseek):**
- BUY DA: +7.3pp (59.7% vs 52.4%)
- SELL DA: +15.7pp (57.5% vs 41.8%)
- Không cần API call, latency ~0ms, cost $0

---

## Data

| File | Mô tả |
|------|-------|
| `data/backtests/knn_returns_best.json` | 1,556 predictions với config, summary, per-date results |
| `stockmem/config/weights.auto.json` | Search weights (New Bayesian, Optuna 150 trials) |
| `scripts/eval_knn_returns.py` | Evaluation script (--out để export) |
| `scripts/optimize_knn_returns_weights.py` | Re-run Bayesian optimization |
