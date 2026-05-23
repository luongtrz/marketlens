# Backtest Report — BTC Signal Model (May 2026, v3)

> **Scope:** 1,569 backtests, BTC, 2022-01-01 → 2026-05-17
> **Đánh giá correctness:** BUY đúng nếu ret7d > 0; SELL đúng nếu ret7d < 0; HOLD đúng nếu |ret7d| < 1%
> **Pipeline:** MainController → LLMGateway (deepseek-v4-flash) → StockMem kNN (k=5) → Regime Guardrails
> **Cập nhật lần cuối:** 2026-05-19

---

## 1. Tổng quan signal distribution

| Signal | Số ngày | Tỷ lệ |
|--------|---------|-------|
| BUY    | 558     | 35.6% |
| SELL   | 170     | 10.8% |
| HOLD   | 841     | 53.6% |
| **Total** | **1,569** | 100% |

**Coverage (BUY+SELL):** 46.4% — tăng mạnh so với v2 (37.5%), model chủ động hơn trong uptrend nhờ prompt mới nhận ra trend context rõ ràng hơn.

---

## 2. Accuracy theo horizon

| Horizon | Correct | Total | Accuracy | Ghi chú |
|---------|---------|-------|----------|---------|
| D+1     | 721     | 1,569 | 46.0%    | HOLD kéo DA xuống (đúng khi \|ret\|<1%, hiếm) |
| D+3     | 586     | 1,569 | 37.3%    | |
| D+7     | 509     | 1,569 | 32.4%    | Horizon chính để đánh giá |
| D+15    | 458     | 1,569 | 29.2%    | |
| D+30    | 433     | 1,569 | 27.6%    | |

> **Lưu ý:** DA tổng thấp vì HOLD chiếm 53.6% — HOLD chỉ đúng khi |ret7d| < 1% (rất hiếm trong crypto). Metric quan trọng hơn là **Precision** bên dưới.

---

## 3. Accuracy theo signal (D+7)

| Signal | Correct | Total | Accuracy | Avg Return | Avg Confidence |
|--------|---------|-------|----------|------------|----------------|
| BUY    | 302     | 558   | **54.1%**    | +0.58%     | 0.712          |
| SELL   | 67      | 170   | **39.4%**    | +1.45%     | 0.580          |
| HOLD   | 140     | 841   | 7.1%         | +0.24%     | 0.547          |

**BUY theo horizon:**

| Horizon | Correct/Total | Accuracy |
|---------|--------------|----------|
| D+1     | 279/558      | 50.0%    |
| D+3     | 295/558      | 52.9%    |
| D+7     | 302/558      | **54.1%** |
| D+15    | 288/558      | 51.6%    |
| D+30    | 308/558      | **55.2%** |

**SELL theo horizon:**

| Horizon | Correct/Total | Accuracy |
|---------|--------------|----------|
| D+1     | 76/170       | 44.7%    |
| D+3     | 68/170       | 40.0%    |
| D+7     | 67/170       | 39.4%    |
| D+15    | 84/170       | **49.4%** |
| D+30    | 78/170       | 45.9%    |

> SELL accuracy tốt nhất ở D+15 (49.4%) — SELL signal có độ trễ, downtrend thường cần 2 tuần để materialize.

---

## 4. Precision (BUY+SELL, excl. HOLD) theo năm

> Precision = tỷ lệ tín hiệu BUY/SELL ra đúng chiều. Đây là metric chính đánh giá chất lượng tín hiệu.

| Năm  | Tổng ngày | Precision | BUY (acc, avg_ret)    | SELL (acc, avg_ret)        | HOLD | Coverage |
|------|-----------|-----------|----------------------|---------------------------|------|----------|
| 2022 | 365       | 43.8%     | 39.8% (103, -2.5%)   | 50.8% (59, -0.1%)          | 203  | 44.4%    |
| 2023 | 365       | **55.1%** | 58.2% (141, +1.9%)   | 29.4% (17, +9.2%)          | 207  | 43.3%    |
| 2024 | 366       | 50.0%     | 56.6% (159, +1.5%)   | 23.1% (39, +3.3%)          | 168  | 54.1%    |
| 2025 | 365       | 51.9%     | 57.4% (122, +0.6%)   | 35.0% (40, +0.5%)          | 203  | 44.4%    |
| 2026 | 107       | **58.3%** | 57.6% (33, -0.0%)    | **60.0%** (15, -3.7%)      | 60   | 44.4%    |
| **All** | **1,569** | **50.7%** | **54.1% (558)** | **39.4% (170)** | **841** | **46.4%** |

---

## 5. Phân tích SELL signal

SELL là tín hiệu khó nhất — model ra SELL sai 60.6% số lần (103/170).

|              | Số lần | Avg actual D+7 return |
|--------------|--------|----------------------|
| SELL đúng    | 67     | **-5.29%** (giá giảm đúng như dự đoán) |
| SELL sai     | 103    | **+5.83%** (giá thực tăng ngược chiều) |

**SELL accuracy theo năm (D+7):**

| Năm  | Đúng/Tổng | Accuracy  | Avg return khi sai | Nhận xét |
|------|-----------|-----------|-------------------|----------|
| 2022 | 30/59     | **50.8%** | — | Bear market — guard rails phù hợp |
| 2023 | 5/17      | 29.4%     | +9.2% | Ít tín hiệu SELL, kém |
| 2024 | 9/39      | 23.1%     | +3.3% | False SELL trong bull run BTC 40k→100k |
| 2025 | 14/40     | 35.0%     | +0.5% | Thị trường sideways/biến động |
| 2026 | 9/15      | **60.0%** | — | Tốt nhất — bear_regime xác nhận rõ |

**Vấn đề confidence:** SELL đúng và SELL sai đều có avg_confidence = 0.580 — model không thể tự phân biệt SELL tốt và SELL xấu thông qua confidence score. Đây là dấu hiệu cần ML classifier thay cho rule-based guardrails.

---

## 6. Phân tích BUY signal (D+7)

| Năm  | Đúng/Tổng | Accuracy  | Avg return | Nhận xét |
|------|-----------|-----------|------------|----------|
| 2022 | 41/103    | 39.8%     | -2.5%      | Bear market, BUY hay sai nhưng cải thiện vs baseline |
| 2023 | 82/141    | **58.2%** | +1.9%      | Tốt — recovery phase |
| 2024 | 90/159    | **56.6%** | +1.5%      | Tốt — bull market |
| 2025 | 70/122    | 57.4%     | +0.6%      | Ổn định |
| 2026 | 19/33     | 57.6%     | -0.0%      | Còn ít data |

BUY accuracy ổn định ~57% trong 2023-2026, chỉ yếu trong 2022 (bear market). BUY là signal đáng tin cậy của mô hình.

---

## 7. So sánh 3 versions

| Metric | Baseline (v1) | Dense+Regime (v2) | Prompt+Trend (v3) | v1→v3 |
|--------|--------------|------------------|------------------|-------|
| BUY D+7 | 51.9% (399) | 53.5% (399) | **54.1% (558)** | +2.2% |
| SELL D+7 | 39.4% (170) | 41.1% (170) | 39.4% (170) | ±0 |
| Precision | 48.2% | 49.8% | **50.7%** | +2.5% |
| Coverage | 36.3% | 37.5% | **46.4%** | +10.1% |
| SELL 2022 | 50.8% | 54.7% | 50.8% | ±0 |
| BUY 2024 | 57.0% | 59.3% | 56.6% | -0.4% |
| SELL 2026 | 60.0% | 60.0% | 60.0% | ±0 |

**Quan sát:**
- Dense embeddings + regime search (v2) cải thiện SELL accuracy nhưng coverage thấp
- Prompt với trend context (v3) tăng BUY signal count (+159 signals) và đưa precision vượt 50%
- SELL accuracy không cải thiện qua cả hai iteration — vấn đề structural, không giải quyết bằng embedding hay prompt

---

## 8. Guard Rails & Pipeline

Pipeline có 8 lớp guard rail trong `_apply_regime_policy`:

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

**Thông số hiện tại:**

```
min_directional_confidence = 0.50
hold_release_bias          = 1.5
knn_confirm_threshold      = 1.0
knn_sell_threshold         = -3.0
```

**Định nghĩa bear_regime:**
- `ret_14d ≤ -6%`
- `avg30_kNN ≤ -4%`
- `avg15_kNN ≤ -3%` AND `avg7_kNN ≤ -1.5%`
- `MACD < 0` AND `avg7_kNN ≤ -2%`

**Định nghĩa bull_regime:**
- `ret_14d ≥ +6%`
- `avg30_kNN ≥ +3%`
- `avg15_kNN ≥ +2%`
- `MACD > 0` AND `avg7_kNN ≥ +1%`

---

## 9. Pipeline & Infrastructure

| Component        | Chi tiết |
|------------------|---------|
| LLM decision     | deepseek-v4-flash qua LLM Gateway (port 8006) |
| Similarity search | StockMem kNN k=5, weighted cosine |
| Factor vector    | 75d dense: 62d type (1.0=active, 0.3=same-group) + 13d group intensity [0.6,1.0] |
| Indicator vector | 5d: z-scored [msi, rsi, sentiment, fear_greed, price_change] |
| Price vector     | 60d: OHLCV features (close returns/intraday ranges/volume changes × 20 candles) |
| kNN weights      | factor=0.678 · indicator=0.101 · price=0.222 (Bayesian re-optimized) |
| Regime bonus     | ±0.15 similarity bonus/penalty khi query và candidate cùng/khác regime |
| Prompt           | ret_14d + trend label + ⚠ SELL WARNING khi kNN bullish |
| Historical data  | 2022–2026 BTC, 1,569 records trong StockMem |

---

## 10. Cải tiến trong v3

| Thay đổi | Mô tả | Tác động |
|----------|-------|---------|
| Dense factor vector | active=1.0, same-group=0.3 thay binary | SELL 2022: +3.9%, cosine fix |
| Regime-conditional search | ±0.15 bonus/penalty theo bull/bear/neutral | Tách signal 2022 vs 2024 |
| Explicit trend context | `ret_14d + trend=STRONG_BULL/...` trong prompt | BUY signals +159 (+40%) |
| SELL WARNING trong prompt | Cảnh báo khi kNN bullish, force justify | Coverage +8.9% |
| SELL gate trong system prompt | 3 điều kiện bắt buộc trước khi SELL | Chưa cải thiện SELL acc |
| kNN weight re-optimization | Bayesian: factor↑, indicator↓, price↓ | Sharpe +29% (0.131→0.169) |

---

## 11. Điểm mạnh & Hạn chế

### Điểm mạnh
- **Precision vượt 50% lần đầu tiên** (50.7%) — model tốt hơn random khi nhìn tổng thể
- **BUY accuracy ổn định** (~54-58% trong 2023-2026) — signal đáng tin cậy
- **Coverage cao** (46.4% vs 36.3% baseline) — model ra signal trong ~một nửa số ngày
- **SELL trong confirmed bear** tốt (2022: 50.8%, 2026: 60%)
- **Kiến trúc đúng hướng**: dense kNN + regime + LLM + layered guardrails
- **Không look-ahead**: strict data cutoff mọi bước

### Hạn chế
- **SELL overall 39.4%** — dưới random (50%). False SELL trong uptrend là vấn đề lớn nhất
- **SELL confidence vô nghĩa**: đúng và sai đều conf=0.580, không phân biệt được
- **2024 SELL 23.1%** — model ra SELL sai trong bull run mạnh BTC 40k→100k, prompt mới không fix
- **HOLD 53.6%** — mặc dù coverage tăng so với v1, vẫn bỏ lỡ ~50% ngày giao dịch
- **Vấn đề LLM**: deepseek-v4-flash thấy tin xấu → SELL, bất kể kNN hay trend

---

## 12. Hướng cải thiện tiếp theo

| Ưu tiên | Hướng | Kỳ vọng | Effort |
|---------|-------|---------|--------|
| Cao | **ML reranker cho SELL** — LightGBM dùng 1,569 labeled rows hiện có để filter SELL output của LLM | SELL 39% → 48-52% | 1-2 ngày |
| Cao | **Ensemble voting cho SELL** — 3 prompt variations, SELL chỉ ra khi ≥2/3 đồng ý | SELL → 43-46%, ít noise | 0.5 ngày |
| Trung bình | **On-chain signals** (exchange netflow, funding rate, whale position) làm hard gate cho SELL | SELL → 55%+ | 3-5 ngày |
| Trung bình | **Reversal detection** thay SELL — hỏi "có breakdown không?" thay "BUY/SELL/HOLD?" | SELL → 45-50% | 1 ngày |
| Thấp | **Crawl thêm 2020-2021** — thêm ~730 ngày kNN data | Cải thiện nhỏ | 2-3 ngày |
