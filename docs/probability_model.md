# Probability Model & Trading Policy — `stockmem/scripts/calibrate_policy.py`

> Biến kết quả kNN retrieval thành xác suất `p_up / p_down / p_hold` và calibrate ngưỡng ra tín hiệu BUY/SELL/HOLD.

---

## Vấn đề với LLM final decision hiện tại

Pipeline cũ dùng LLM để ra BUY/SELL/HOLD. Hạn chế:
- Confidence không có ý nghĩa xác suất (không calibrated)
- Không thể backtest một cách tái lập được
- Guardrail hardcode, không tune từ data

Module này thay thế bước đó bằng **kNN-based probability + policy tuned trên validation**.

---

## Phương pháp: kNN Probability

Với query ngày T, lấy top-k neighbors từ retriever (baseline hoặc learned).  
Mỗi neighbor có `future_return_7d` đã mature.

```python
# Shift similarity [-1, 1] → weight [0, 1]
w_i = (sim_i + 1) / 2

p_up   = Σ w_i · 1[return_i > 0] / Σ w_i
p_down = Σ w_i · 1[return_i < 0] / Σ w_i
p_hold = max(0, 1 - p_up - p_down)
```

Không cần neural network — kNN đủ mạnh với ~2885 training records.

---

## Calibration — Grid search tau trên VAL only

```
signal = BUY  nếu p_up - p_down >= tau
signal = SELL nếu p_down - p_up >= tau
signal = HOLD ngược lại
```

Grid search `tau ∈ [0.02, 0.50]` step 0.02:
- Constraint: coverage (BUY+SELL) giữa 15% và 70%
- Objective: maximize Sharpe trên **validation set** (2025-01 → 2025-06)
- **Không dùng test set để chọn tau**

Kết quả hiện tại: **tau = 0.22**, val_sharpe = 0.137

---

## Kết quả test set (2025-07 → 2026-05, 305 queries)

| Retriever | DA | BUY DA | SELL DA | Coverage | Sharpe | Sortino | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| baseline_fixed_knn | 0.777 | 0.411 | 0.625 | 0.426 | −0.140 | −0.259 | 1.150 | 0.206 |
| **learned_cem_rag** | **0.790** | **0.420** | **0.630** | 0.416 | **+0.041** | **+0.003** | 1.152 | 0.218 |

Learned retriever cải thiện DA +1.3pp, Sharpe từ âm lên dương (+0.18 tuyệt đối).

---

## Output: `stockmem/config/policy.json`

```json
{
  "tau": 0.22,
  "val_sharpe": 0.1367,
  "method": "knn_platt_v1",
  "retriever": "learned_diagonal",
  "k": 5,
  "created_at": "2026-06-23T..."
}
```

---

## Schema mới: `shared/models/prediction.py`

```python
class CEMRAGPrediction(BaseModel):
    horizon: str = "7d"
    p_up: float        # xác suất return > 0
    p_down: float      # xác suất return < 0
    p_hold: float      # xác suất |return| nhỏ
    signal: str        # "BUY" | "SELL" | "HOLD"
    confidence: float  # = max(p_up, p_down, p_hold)
    tau: float         # ngưỡng đã dùng
    explanation: str = ""
    retrieval_count: int = 0
```

---

## Chạy / retrain policy

```bash
# Dùng learned retriever (mặc định)
PYTHONPATH=/home/luong/marketlens python stockmem/scripts/calibrate_policy.py \
  --data stockmem/data/real_optimizer_v2.json \
  --artifact stockmem/config/learned_retriever.json \
  --k 5 \
  --output stockmem/config/policy.json

# Dùng baseline fixed kNN
PYTHONPATH=/home/luong/marketlens python stockmem/scripts/calibrate_policy.py \
  --data stockmem/data/real_optimizer_v2.json \
  --no-learned \
  --output stockmem/config/policy_baseline.json
```

---

## Limitations hiện tại (v1)

- Brier score 1.15 (>1.0 = worse than uniform prior) — probabilities chưa calibrated tốt, ECE = 0.21
- kNN-based probability bị ảnh hưởng bởi class imbalance (nhiều UP hơn DOWN trong bull market)
- Chưa có temperature scaling hay isotonic regression
- Chưa test transaction cost sensitivity (5/10/20 bps)

Để cải thiện (v2): thêm Platt scaling trên val, hoặc XGBoost trên [p_up, p_down, indicator_vec] features.
