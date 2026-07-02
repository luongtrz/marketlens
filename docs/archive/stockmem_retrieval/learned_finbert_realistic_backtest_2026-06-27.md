# Learned FinBERT Realistic Backtest

**Date:** 2026-06-27  
**Script:** `scripts/backtest_knn_head_realistic.py`  
**Candidate:** `stockmem/config/knn_head.learned_finbert_rolling_stable.json`

---

## 1. Why This Backtest Exists

Mục tiêu của bước này là chuyển từ:

- forecasting quality,
- offline proxy DA,

sang một evaluation gần hơn với execution thực tế:

- fixed hold `7d`,
- `long / short / flat`,
- không cho chồng lệnh,
- có phí giao dịch,
- có evidence top retrieved records cho mỗi quyết định.

Đây là bước cần thiết để trả lời câu hỏi:

> learned candidate có còn đáng tin hơn baseline sau execution constraint và trading cost không?

---

## 2. Execution Protocol

### Signal generation

Mỗi ngày:

1. retrieve top-`k` historical records,
2. aggregate multi-horizon future returns theo head config,
3. map sang `BUY / HOLD / SELL`.

### Position model

- `BUY` -> long
- `SELL` -> short
- `HOLD` -> flat

### Holding rule

- giữ vị thế cố định `7` ngày
- **không chồng lệnh**
- nếu đang giữ vị thế thì bỏ qua signal mới cho đến khi hết hạn hold

### Cost scenarios

1. `no_cost`
2. `fee_10bps_side`
3. `fee_10bps_plus_slippage_5bps_side`

Trong scenario thứ 3:

- fee = `0.10%` mỗi side
- slippage = `0.05%` mỗi side
- round-trip cost = `0.30%`

---

## 3. Evidence Output

Backtest output có 2 lớp row-level:

- `decision_rows`
  - mọi ngày có signal
  - có `weighted_avg_return`
  - có `actual_return_7d`
  - có `evidence` top retrieved cases

- `trade_rows`
  - chỉ những ngày thực sự vào lệnh
  - có `entry_date`, `exit_date`
  - `gross_return_pct`, `net_return_pct`
  - evidence top retrieved cases

Điều này phù hợp với mục tiêu hệ thống:

- decision engine đưa forecast có evidence,
- service xAI khác có thể lấy forecast hiện tại + evidence để giải thích cho user.

---

## 4. Important Caveat

Backtest này **thật hơn trước**, nhưng vẫn chưa phải trading simulator hoàn chỉnh.

Vẫn còn các giới hạn:

1. dùng `future_return_7d` trực tiếp làm realized trade return,
2. chưa có:
   - intraday fill model,
   - funding / borrow cost,
   - partial fills,
   - market impact,
   - portfolio sizing beyond full-notional.

Vì vậy:

- các con số return / Sharpe / MDD dùng được để **so tương đối**
- nhưng chưa nên coi là kết quả deploy trading cuối cùng

Đặc biệt:

- `MDD` rất sâu là dấu hiệu strategy path vẫn rất biến động,
- total return rất lớn ở fixed strategies cho thấy compounding sensitivity cao.

---

## 5. 2024-01-01 -> 2026-06-21

File: `artifacts/backtests/knn_head_realistic_2024_2026.json`

### Cost scenario: 10 bps fee + 5 bps slippage per side

| Strategy | Overall DA | Active Acc | Coverage | Trades | Total Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Learned candidate** | **48.5%** | **54.8%** | **86.4%** | 123 | 99.6% | 0.761 | -93.4% |
| Fixed stable | 44.9% | 54.4% | 77.4% | 122 | **5517.7%** | **2.089** | -74.5% |
| Production baseline | 43.0% | 54.7% | 71.0% | 120 | 3263.4% | 1.877 | -80.9% |

### Readout

Learned candidate đứng đầu về:

- `overall_DA`
- `active_acc`
- `coverage`

Nhưng fixed stable vẫn thắng mạnh về:

- total return
- Sharpe
- drawdown

Tức là:

> learned tốt hơn về decision quality, nhưng fixed mạnh hơn về trade-return profile trong cửa sổ gần hiện tại.

---

## 6. 2022-01-01 -> 2026-06-21

File: `artifacts/backtests/knn_head_realistic_2022_2026.json`

### Cost scenario: 10 bps fee + 5 bps slippage per side

| Strategy | Overall DA | Active Acc | Coverage | Trades | Total Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Learned candidate** | **47.6%** | 55.4% | **83.1%** | 225 | 310.8% | 0.840 | -83.3% |
| Fixed stable | 46.0% | 55.7% | 77.6% | 221 | **25603.9%** | **1.700** | -91.3% |
| Production baseline | 44.5% | **56.0%** | 73.1% | 219 | 5789.2% | 1.382 | -90.2% |

### Readout

Learned candidate vẫn đứng đầu ở:

- `overall_DA`
- `coverage`

Nhưng vẫn không thắng ở:

- Sharpe
- total return
- active accuracy

---

## 7. Interpretation

## 7.1 What is clearly good

Learned candidate hiện đã cho thấy:

- signal coverage cao hơn rõ ràng,
- overall classification quality tốt hơn,
- evidence retrieval có thể dùng trực tiếp cho xAI layer.

Đối với mục tiêu sản phẩm “cung cấp thông tin có căn cứ”, đây là điểm rất mạnh.

## 7.2 What is not yet good enough

Nếu tiêu chí là **trading performance**:

- learned candidate chưa thắng fixed stable
- sau phí vẫn kém hơn khá rõ ở Sharpe / total return

Nói cách khác:

> learned candidate hiện là **better information model** hơn là **better trading strategy**.

Đó không hẳn là thất bại, vì mục tiêu sản phẩm của bạn là:

- dự đoán ổn hơn một chút,
- có evidence hợp lý,
- đủ đáng tin để đưa cho xAI giải thích.

Theo tiêu chí đó, learned candidate đang khá gần mục tiêu.

---

## 8. Research Positioning

Hai hướng nghiên cứu hiện tại đều hợp lý:

### A. Realistic execution backtest

Kết quả cho thấy:

- nếu chỉ nhìn DA thì learned có vẻ hơn,
- nhưng khi đưa vào execution rule + cost thì superiority chưa còn toàn diện.

Đây là insight research thật:

> retrieval improvement không tự động chuyển thành trading improvement.

### B. Stability-aware head selection

Đây là đóng góp methodological rõ:

- single validation block không đủ cho time-series retrieval system,
- shared stable rolling selection giúp learned candidate vượt fixed baseline ở overall DA và coverage,
- và còn tạo ra config ít nhảy hơn, deployable hơn.

Đây là phần novelty có thể viết được như một methodological contribution.

---

## 9. Current Decision

### For product

**Acceptable as evidence-backed forecast candidate**

Lý do:

- overall DA cao hơn baseline
- coverage cao
- có row-level evidence usable cho xAI

### For trading claim

**Not yet acceptable as best trading strategy**

Lý do:

- fixed stable vẫn mạnh hơn rõ ở return-based metrics

---

## 10. Next Research Work

1. giữ learned candidate làm information/evidence model,
2. chưa thay fixed stable nếu mục tiêu là best trading profile,
3. nghiên cứu tiếp:
   - regime-aware policy layer,
   - downstream-aware retriever objective,
   - non-overlapping portfolio simulation tốt hơn,
   - cost-sensitive policy optimization.
