# Learned FinBERT Candidate Backtest

**Date:** 2026-06-27  
**Candidate config:** `stockmem/config/knn_head.learned_finbert_rolling_stable.json`  
**Control config:** `stockmem/config/knn_head.fixed_knn_rolling_stable.json`  
**Script:** `scripts/backtest_knn_head_candidates.py`

---

## 1. Backtest Strategy

Ba strategy được chấm cùng một protocol offline:

1. **Learned candidate**
   - retriever: `learned_retriever_finbert.json`
   - head: `learned_finbert_rolling_stable_v1`

2. **Fixed stable control**
   - retriever: fixed-kNN
   - head: `fixed_knn_rolling_stable_v1`

3. **Production baseline**
   - retriever: fixed-kNN
   - head: current repo default `knn_returns`

### Fairness rules

- cùng dataset: `data/exports/stockmem_records.ndjson`
- cùng walk-forward restriction: chỉ search trên lịch sử trước ngày query
- cùng future-return labels
- chỉ thay retriever và head

### Two-layer evaluation

1. **Candidate-confirmation window**
   - `2024-01-01 -> 2026-06-21`
   - mục tiêu: kiểm tra candidate trong regime gần hiện tại

2. **Long-horizon comparison**
   - `2022-01-01 -> 2026-06-21`
   - mục tiêu: so với scope các report kNN trước đó

---

## 2. Important Caveat

Các metric:

- `avg_ret7d`
- `sharpe`
- `sortino`
- `max_drawdown`

trong script hiện tại là **proxy offline**, không phải PnL trading engine đầy đủ.

Lý do:

- mỗi quyết định đang được chấm trực tiếp bằng `actual_return_7d`,
- returns giữa các ngày bị chồng lấn,
- chưa có portfolio sizing / transaction cost / holding constraint.

Vì vậy:

- dùng chúng để so tương đối giữa candidate và baseline thì được,
- nhưng **không dùng như kết luận cuối cùng về trading profitability**.

Đặc biệt `max_drawdown ≈ -100%` ở output hiện tại là artifact của cách compounding proxy return, không phải kết quả deployable.

---

## 3. Results

## 3.1 Candidate-confirmation window: 2024-01-01 -> 2026-06-21

Output: `artifacts/backtests/knn_head_candidates_2024_2026.json`

| Strategy | Coverage | BUY DA | SELL DA | HOLD DA | Overall DA | Active Acc | Avg ret7d |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Learned candidate** | **86.4%** | **56.6%** | 46.7% | 8.4% | **48.5%** | **54.8%** | **2.96%** |
| Fixed stable control | 77.4% | 55.0% | **51.0%** | 12.1% | 44.9% | 54.4% | 2.57% |
| Production baseline | 71.0% | 55.6% | 49.5% | **14.6%** | 43.0% | 54.7% | 2.56% |

### Readout

- Learned candidate đứng đầu ở:
  - `coverage`
  - `BUY_DA`
  - `overall_DA`
  - `active_acc`
  - `avg_ret7d`
- Fixed stable vẫn hơn ở `SELL_DA`
- Production baseline không còn là best option trong cửa sổ gần hiện tại

---

## 3.2 Long-horizon window: 2022-01-01 -> 2026-06-21

Output: `artifacts/backtests/knn_head_candidates_2022_2026.json`

| Strategy | Coverage | BUY DA | SELL DA | HOLD DA | Overall DA | Active Acc | Avg ret7d |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Learned candidate** | **83.1%** | **57.3%** | 52.5% | 9.6% | **47.6%** | 55.4% | 1.90% |
| Fixed stable control | 77.6% | 56.7% | 53.1% | 12.3% | 46.0% | 55.7% | **2.43%** |
| Production baseline | 73.1% | **57.3%** | **53.4%** | **13.0%** | 44.5% | **56.0%** | 2.35% |

### Readout

- Learned candidate vẫn đứng đầu ở:
  - `coverage`
  - `overall_DA`
- BUY side ngang hoặc nhỉnh hơn baseline
- Nhưng:
  - `SELL_DA` chưa vượt production baseline
  - `active_acc` vẫn thấp hơn một chút
  - `avg_ret7d` proxy thấp hơn fixed variants

---

## 4. Interpretation

### What improved

Learned candidate hiện đã chứng minh được:

- policy ổn định hơn so với các run learned trước,
- `overall_DA` tốt hơn cả fixed stable lẫn production baseline ở cả hai cửa sổ,
- coverage tăng mạnh mà không làm active accuracy sụp.

### What is still weak

Learned candidate vẫn chưa phải clean winner ở mọi mặt:

- `SELL_DA` chưa dẫn đầu
- `HOLD_DA` thấp
- proxy return / sharpe chưa vượt fixed stable trong long-horizon window

Nói cách khác:

> Learned candidate tốt hơn về **classification policy quality**, nhưng chưa chắc tốt hơn về **trading-return proxy** trên full 2022–2026.

---

## 5. Decision

### Status

**Promote to production-candidate, not production-default**

### Meaning

- Learned candidate đã vượt qua ngưỡng “research curiosity”
- đáng được đưa vào candidate set chính thức
- nhưng chưa đủ để thay default baseline ngay

### Why

1. thắng `overall_DA` nhất quán ở cả 2 windows
2. coverage cao hơn rõ
3. active accuracy không bị giảm mạnh
4. nhưng SELL / HOLD / return-proxy vẫn còn trade-off

---

## 6. Recommended Next Step

1. Giữ `learned_finbert_rolling_stable_v1` làm candidate chính thức
2. Backtest tiếp bằng protocol giàu hơn:
   - non-overlapping holding logic
   - transaction cost
   - exposure control
3. So thêm:
   - realized trade count
   - net return after cost
   - per-regime breakdown
4. Chỉ sau đó mới quyết định:
   - promote learned candidate thành default
   - hoặc tiếp tục refine retriever / head
