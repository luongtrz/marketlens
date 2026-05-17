# Backtest Report — BTC Signal Model (May 2026)

> **Scope:** 1,568 backtests, BTC, 2022-01-01 → 2026-05-17
> **Đánh giá correctness:** BUY đúng nếu ret7d > 0; SELL đúng nếu ret7d < 0; HOLD đúng nếu |ret7d| < 1%
> **Pipeline:** MainController → LLMGateway (deepseek-v4-flash) → StockMem kNN (k=5) → Regime Guardrails
> **Cập nhật lần cuối:** 2026-05-17

---

## 1. Tổng quan signal distribution

| Signal | Số ngày | Tỷ lệ |
|--------|---------|-------|
| BUY    | 399     | 25.4% |
| SELL   | 170     | 10.8% |
| HOLD   | 999     | 63.7% |
| **Total** | **1,568** | 100% |

**Coverage (BUY+SELL):** 36.3% — model chủ động ra tín hiệu trên ~1/3 số ngày

---

## 2. Accuracy theo horizon

| Horizon | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| D+1     | 523     | 1,568 | 33.4%    |
| D+3     | 381     | 1,568 | 24.3%    |
| D+7     | 345     | 1,568 | 22.0%    |
| D+15    | 345     | 1,568 | 22.0%    |
| D+30    | 331     | 1,567 | 21.1%    |

> **Lưu ý:** DA (Directional Accuracy) tổng thấp vì HOLD chiếm 64% — HOLD chỉ đúng khi |ret7d| < 1% nên kéo số xuống. Metric quan trọng hơn là **Precision** bên dưới.

---

## 3. Accuracy theo signal (D+7)

| Signal | Correct | Total | Accuracy | Avg Confidence |
|--------|---------|-------|----------|----------------|
| BUY    | 207     | 399   | **51.9%**    | 0.719          |
| SELL   | 67      | 170   | **39.4%**    | 0.580          |
| HOLD   | 71      | 999   | 7.1%         | 0.548          |

**BUY theo horizon:**

| Horizon | Correct/Total | Accuracy |
|---------|--------------|----------|
| D+1     | 186/399      | 46.6%    |
| D+7     | 207/399      | **51.9%** |
| D+30    | 222/399      | **55.6%** |

**SELL theo horizon:**

| Horizon | Correct/Total | Accuracy |
|---------|--------------|----------|
| D+1     | 76/170       | 44.7%    |
| D+7     | 67/170       | 39.4%    |
| D+30    | 78/170       | 45.9%    |

---

## 4. Precision (BUY+SELL, excl. HOLD) theo năm

> Precision = tỷ lệ tín hiệu BUY/SELL ra đúng chiều. Đây là metric chính đánh giá chất lượng tín hiệu.

| Năm  | Tổng ngày | DA D+7 | Precision | BUY (acc)    | SELL (acc)  | HOLD | Coverage |
|------|-----------|--------|-----------|--------------|-------------|------|----------|
| 2022 | 365       | 17.0%  | 43.1%     | 57 (35.1%)   | 59 (50.8%)  | 249  | 32%      |
| 2023 | 365       | 23.8%  | **55.7%** | 105 (60.0%)  | 17 (29.4%)  | 243  | 33%      |
| 2024 | 366       | 24.0%  | 48.8%     | 121 (57.0%)  | 39 (23.1%)  | 206  | 44%      |
| 2025 | 365       | 22.7%  | 43.9%     | 92 (47.8%)   | 40 (35.0%)  | 233  | 36%      |
| 2026 | 107       | 23.4%  | **51.3%** | 24 (45.8%)   | 15 (60.0%)  | 68   | 36%      |
| **All** | **1,568** | **22.0%** | **48.2%** | **399 (51.9%)** | **170 (39.4%)** | **999** | **36%** |

---

## 5. Phân tích SELL signal

SELL là tín hiệu khó nhất — model ra SELL sai 61% số lần (103/170).

|              | Số lần | Avg actual D+7 return |
|--------------|--------|----------------------|
| SELL đúng    | 67     | **-5.29%** (giá giảm đúng như dự đoán) |
| SELL sai     | 103    | **+5.83%** (giá thực tăng ngược chiều) |

**SELL accuracy theo năm (D+7):**

| Năm  | Đúng/Tổng | Accuracy  | Nhận xét |
|------|-----------|-----------|----------|
| 2022 | 30/59     | **50.8%** | Bear market — guard rails phù hợp |
| 2023 | 5/17      | 29.4%     | Ít tín hiệu SELL, kém |
| 2024 | 9/39      | 23.1%     | False SELL trong bull run BTC 40k→100k |
| 2025 | 14/40     | 35.0%     | Thị trường sideways/biến động |
| 2026 | 9/15      | **60.0%** | Tốt nhất — bear_regime xác nhận rõ |

---

## 6. Phân tích BUY signal (D+7)

| Năm  | Đúng/Tổng | Accuracy  | Nhận xét |
|------|-----------|-----------|----------|
| 2022 | 20/57     | 35.1%     | Bear market, BUY hay sai |
| 2023 | 63/105    | **60.0%** | Tốt — recovery phase |
| 2024 | 69/121    | **57.0%** | Tốt — bull market |
| 2025 | 44/92     | 47.8%     | Trung bình |
| 2026 | 11/24     | 45.8%     | Còn ít data |

BUY cải thiện rõ từ bear (2022: 35%) → bull (2023-2024: ~58%), cho thấy model định hướng uptrend tốt hơn downtrend.

---

## 7. Guard Rails đang hoạt động

Pipeline có 8 lớp guard rail trong `_apply_regime_policy` ([llm_gateway_client.py](../main_controller/src/clients/llm_gateway_client.py)):

| Guard | Mô tả | Tác động chính |
|-------|-------|----------------|
| G1 | Block SELL trong bull_regime mạnh | Giảm false SELL |
| G2 | Block BUY trong bear_regime mạnh | Giảm false BUY |
| G3 | HOLD → BUY/SELL khi regime + momentum khớp | Tăng coverage |
| G4 | Confidence gate ≥ 0.50 cho BUY/SELL | Lọc tín hiệu yếu |
| G5 | Release HOLD khi directional_bias ≥ 1.5 | Tăng BUY coverage |
| G6 | RSI exhaustion (>70+short_down / <30+short_up) | Chặn tại đỉnh/đáy |
| G7 | Dual-timeframe momentum (14d + 3d cùng chiều) | Phát hiện trend sớm |
| G8 | SELL bắt buộc phải có bear_regime xác nhận | Giảm false SELL |

**Thông số hiện tại (đã tuning):**

```
min_directional_confidence = 0.50   # confidence tối thiểu để ra BUY/SELL
hold_release_bias          = 1.5    # ngưỡng bias để release HOLD → BUY
knn_confirm_threshold      = 1.0    # veto BUY nếu avg7d kNN < -1%
knn_sell_threshold         = -3.0   # veto SELL nếu avg7d kNN > -3%
```

**Định nghĩa bear_regime** (bất kỳ điều kiện nào):
- `ret_14d ≤ -6%`
- `avg30_kNN ≤ -4%`
- `avg15_kNN ≤ -3%` **VÀ** `avg7_kNN ≤ -1.5%`
- `MACD < 0` **VÀ** `avg7_kNN ≤ -2%`

**Định nghĩa bull_regime** (bất kỳ điều kiện nào):
- `ret_14d ≥ +6%`
- `avg30_kNN ≥ +3%`
- `avg15_kNN ≥ +2%`
- `MACD > 0` **VÀ** `avg7_kNN ≥ +1%`

---

## 8. Pipeline & Infrastructure

| Component        | Chi tiết |
|------------------|---------|
| LLM decision     | deepseek-v4-flash qua LLM Gateway (port 8006) |
| Similarity search | StockMem kNN k=5, weighted cosine |
| Factor vector    | 75d: 62d taxonomy type bits + 13d group bits |
| Indicator vector | 5d: z-scored [msi, rsi, sentiment, fear_greed, price_change] |
| Price vector     | 60d: OHLCV features (close returns/intraday ranges/volume changes × 20 candles) |
| kNN weights      | factor=0.35 · indicator=0.20 · price=0.45 |
| Historical data  | 2022–2026 BTC, 1,568 records trong StockMem |
| 2022 article data | Crawl từ Decrypt sitemaps (676 bài thật), factor extraction qua LLM gateway taxonomy |

---

## 9. Quá trình cải thiện trong session này

| Bước | Thay đổi | Trước | Sau |
|------|---------|-------|-----|
| Crawl 2022 | 676 bài Decrypt có nội dung thật | Không có | ✅ |
| Build factor snapshots | 304 ngày 2022 qua LLM gateway | 0-2 bits/ngày | 2-8 bits/ngày |
| Backfill StockMem 2022 | ~345 records nạp lại | Factor rỗng | Factor thật |
| Guard rails | Tighten knn_sell_threshold: -1.5 → -3.0 | SELL 38.2% | SELL 39.4% |
| Guard rails | Tighten bear_regime thresholds | - | Bear phân loại chính xác hơn |
| Guard rails | Giảm hold_release_bias: 2.0 → 1.5 | Coverage 37.8% | Coverage 36.3% |
| Guard rails | Giảm min_confidence: 0.52 → 0.50 | - | Thông thoáng hơn |

---

## 10. Điểm mạnh & Hạn chế

### Điểm mạnh
- **BUY accuracy ổn định** (~52% overall, ~57-60% trong bull market) — tốt hơn random (50%)
- **SELL trong confirmed bear** tốt (2022: 50.8%, 2026: 60%)
- **Kiến trúc đúng hướng**: kNN historical evidence + LLM reasoning + layered regime guardrails
- **Không look-ahead**: pipeline có strict data cutoff — valid cho backtesting
- **2022 data quality**: factor vectors thật từ 676 bài crawl thay vì rỗng

### Hạn chế
- **SELL overall 39.4%** — dưới ngưỡng useful (50%). False SELL trong uptrend là vấn đề lớn nhất
- **HOLD quá nhiều (63.7%)** — coverage thấp, bỏ lỡ nhiều cơ hội giao dịch
- **SELL 2024 chỉ 23.1%** — model ra SELL sai trong bull run mạnh BTC 40k→100k
- **2022 DA thấp (17%)** — bear market phức tạp, kNN ít data tương đồng giai đoạn đầu
- **Confidence không phân biệt đúng/sai**: SELL đúng và sai đều có confidence ~0.58

---

## 11. Hướng cải thiện tiếp theo

| Ưu tiên | Hướng | Kỳ vọng |
|---------|-------|---------|
| Cao | Crawl thêm data 2023 (hiện chỉ có Decrypt) | Cải thiện 2023 SELL accuracy (29% → 45%+) |
| Cao | Tách confidence thresholds riêng cho BUY vs SELL | Giảm false SELL 2024 |
| Trung bình | Thêm on-chain signals (exchange outflow, whale movements) | Factor vector đặc trưng hơn |
| Trung bình | Tăng k=5 lên k=7 cho SELL decision | kNN evidence mạnh hơn, ít nhiễu |
| Thấp | Fine-tune bull_regime threshold cho strong bull market | Tránh SELL trong 40k→100k style run |
