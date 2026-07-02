# Learned FinBERT Rolling-Stable Head Report

**Date:** 2026-06-27  
**Source artifacts:** `wf_full.json`, `wf_rolling_stable.json`  
**Candidate config:** `stockmem/config/knn_head.learned_finbert_rolling_stable.json`

---

## 1. Objective

Sau khi run walk-forward full đầu tiên (`wf_full.json`), learned FinBERT cho thấy:

- có signal,
- nhưng chọn head theo từng fold còn nhảy mạnh,
- và chưa thắng fixed-kNN ổn định.

Mục tiêu của đợt refinement tiếp theo là:

1. chỉ dùng **rolling window**,
2. thu hẹp search space của head,
3. chọn config theo tiêu chí **shared stable**:
   - maximize mean validation quality,
   - đồng thời phạt variance giữa folds.

---

## 2. Protocol

### 2.1 Retriever

- Candidate: `learned_retriever_finbert.json`
- Baseline: fixed-kNN với search weights từ `weights.auto.json`

### 2.2 Decision head

Head `knn_returns` vẫn giữ cùng form:

1. retrieve top-`k` neighbors,
2. tính weighted average future returns theo các horizon:
   - `1d`, `3d`, `7d`, `15d`, `30d`
3. map sang `BUY/HOLD/SELL` theo `buy_threshold`, `sell_threshold`

### 2.3 Windowing

- chỉ dùng `rolling window`
- train window: 36 tháng
- validation window: 6 tháng
- test window: 3 tháng
- step: 3 tháng

### 2.4 Search refinement

So với run full ban đầu:

- search space hẹp hơn:
  - `k ∈ {3, 5, 7, 10, 15, 20}`
  - threshold ranges hẹp hơn
  - return-weight prior không còn uniform
- ưu tiên nhiều hơn cho `7d` / `15d` / `30d`

### 2.5 Stable selection objective

Không còn chỉ chọn best config riêng cho từng fold.

Thay vào đó, với mỗi candidate head:

1. evaluate trên validation của tất cả rolling folds,
2. tính score trung bình,
3. trừ penalty theo độ lệch chuẩn của:
   - selection score
   - `overall_DA`
   - `active_acc`
   - `coverage`

Mục tiêu là chọn **một config chung**, không phải một config “đẹp” cho từng đoạn riêng lẻ.

---

## 3. Results

## 3.1 Per-fold rolling tuning

### Learned FinBERT

- `BUY_DA mean = 51.25%`
- `SELL_DA mean = 42.61%`
- `overall_DA mean = 39.80%`
- `active_acc mean = 51.91%`
- `coverage mean = 63.45%`

### Fixed baseline

- `BUY_DA mean = 49.52%`
- `SELL_DA mean = 39.51%`
- `overall_DA mean = 40.34%`
- `active_acc mean = 53.33%`
- `coverage mean = 62.26%`

### Interpretation

Per-fold tuning cho thấy learned:

- tốt hơn ở `BUY_DA`,
- tốt hơn ở `SELL_DA`,
- coverage nhỉnh hơn,
- nhưng `overall_DA` và `active_acc` chưa vượt fixed rõ ràng.

Điều này xác nhận retriever learned có alpha, nhưng nếu mỗi fold chọn một head khác nhau thì policy vẫn chưa đủ clean.

---

## 3.2 Shared stable config

### Learned best stable config

```json
{
  "k": 5,
  "buy_thr": 1.45,
  "sell_thr": 1.47,
  "return_weights": {
    "1d": 0.0161,
    "3d": 0.1459,
    "7d": 0.4549,
    "15d": 0.1005,
    "30d": 0.2827
  }
}
```

### Learned stable test summary

- `BUY_DA mean = 56.05%`
- `SELL_DA mean = 47.08%`
- `overall_DA mean = 47.88%`
- `active_acc mean = 54.65%`
- `coverage mean = 84.59%`
- `overall_DA std = 10.74`

### Fixed stable baseline summary

- `BUY_DA mean = 55.73%`
- `SELL_DA mean = 46.03%`
- `overall_DA mean = 45.71%`
- `active_acc mean = 54.61%`
- `coverage mean = 77.95%`
- `overall_DA std = 11.81`

### Delta vs fixed

Learned stable config thắng fixed stable baseline ở:

- `BUY_DA`
- `SELL_DA`
- `overall_DA`
- `coverage`
- variance tổng thể cũng thấp hơn nhẹ

`active_acc` gần như ngang nhau.

---

## 4. What Changed vs Earlier Conclusion

Trước khi có stable rolling protocol:

- learned trông promising nhưng chưa đủ ổn định,
- expanding window còn cho kết quả xấu,
- config head nhảy quá mạnh.

Sau khi:

- bỏ expanding,
- thu hẹp search space,
- ép stable selection,

kết quả đổi đáng kể:

- learned không còn chỉ là “có signal”,
- mà đã tạo ra một **production-candidate head** có mean test metrics tốt hơn fixed baseline.

Điểm quan trọng nhất là:

> Vấn đề trước đây không chỉ là retriever, mà chủ yếu là **cách chọn head sai kiểu time-series**.

---

## 5. Recommended Backtest Strategy

## 5.1 What to backtest next

Không nên nhảy thẳng sang “deploy candidate”.

Bước tiếp theo nên là backtest riêng cho:

1. **Candidate A**
   - retriever: learned FinBERT
   - head: `learned_finbert_rolling_stable_v1`

2. **Control B**
   - retriever: fixed-kNN
   - head: stable fixed config từ `wf_rolling_stable.json`

3. **Current production baseline**
   - retriever: fixed-kNN
   - head/config đang dùng trong repo hiện tại

## 5.2 Backtest period

Nên có 2 tầng:

### Tầng 1: candidate-confirmation

- focus: `2024-01-01 -> 2026-06-21`
- lý do:
  - gần regime hiện tại hơn,
  - phản ánh trực tiếp giá trị deploy gần-term,
  - bao gồm bull + volatile post-bull period

### Tầng 2: long-horizon comparison

- full: `2022-01-01 -> 2026-06-21`
- lý do:
  - so trực tiếp với các report kNN / LLM trước đó,
  - tránh overclaim từ recent regime only

## 5.3 Evaluation metrics

Backtest nên báo cáo ít nhất:

- `BUY_DA`
- `SELL_DA`
- `HOLD_DA`
- `overall_DA`
- `active_acc`
- `coverage`
- mean return by signal
- cumulative PnL / simple strategy return
- Sharpe / Sortino
- max drawdown

## 5.4 Rules for fair comparison

Để fair:

1. giữ cùng data universe,
2. cùng maturity guard,
3. cùng future-return labels,
4. chỉ thay:
   - retriever,
   - head config.

Không được:

- tune lại candidate trên chính test period đang chấm,
- thay đổi threshold sau khi nhìn kết quả backtest,
- dùng expanding protocol để cứu một candidate vốn chỉ tốt trong rolling.

---

## 6. Production Recommendation (Current)

### Status

**Promising Go**

### Meaning

- chưa đủ để thay production default ngay,
- nhưng đủ mạnh để làm candidate chính thức cho backtest production-style.

### Why

- learned stable config đã vượt fixed stable baseline trên rolling protocol,
- advantage đến từ quality của policy selection chứ không phải cherry-pick một fold đẹp.

---

## 7. Next Action

1. wire `learned_finbert_rolling_stable_v1` vào backtest script,
2. run candidate-confirmation backtest,
3. compare với:
   - fixed stable,
   - current production baseline,
4. chỉ sau đó mới quyết định:
   - promote candidate,
   - hay cần unfreeze retriever.
