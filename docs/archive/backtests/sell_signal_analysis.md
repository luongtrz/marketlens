# Phân Tích SELL Signal — Vấn Đề & Xu Hướng

> **Scope:** 170 SELL signals, BTC, 2022–2026 (1,569 ngày backtest)
> **Cập nhật:** 2026-05-19 | Pipeline: v3 (dense embeddings + regime search + prompt trend context)

---

## 1. Tóm tắt vấn đề

SELL là signal kém nhất trong mô hình — **sai 60.6% số lần**, dưới mức random (50%).

| Kết quả | Số lần | Avg D+7 return thực tế | Avg D+30 return |
|---------|--------|------------------------|-----------------|
| SELL đúng (ret7d < 0) | 67 (39.4%) | **-5.29%** | **-5.6%** |
| SELL sai (ret7d ≥ 0)  | 103 (60.6%) | **+5.83%** | **+6.3%** |

Khi model ra SELL sai, giá không chỉ tăng nhẹ mà tăng **mạnh** — avg +5.83% trong 7 ngày và +6.3% trong 30 ngày. Đây không phải nhiễu ngẫu nhiên mà là pattern: model ra SELL trong **sustained uptrend**.

---

## 2. Phân bổ mức độ sai của SELL

Trong 103 SELL sai:

| Giá tăng thực tế (D+7) | Số lần | Tỷ lệ | Mức độ |
|------------------------|--------|-------|--------|
| +0% → +2%              | 30     | 29.1% | Nhẹ — correction rồi phục hồi |
| +2% → +5%              | 33     | 32.0% | Vừa — rõ ràng sai hướng |
| +5% → +10%             | 25     | 24.3% | Nặng — bỏ lỡ rally lớn |
| +10% → +20%            | 11     | 10.7% | Rất nặng |
| > +20%                 | 4      | 3.9%  | Nghiêm trọng — đỉnh bull run |

**70.9% SELL sai** có giá tăng > +2% — không phải nhiễu mà là real opportunity cost.

---

## 3. SELL accuracy theo năm (xu hướng đảo ngược với thị trường)

| Năm | Regime thị trường | SELL Acc | Sai/Tổng | Avg return khi sai | Avg sentiment |
|-----|-------------------|----------|-----------|-------------------|---------------|
| 2022 | **Bear** (BTC -65%) | **50.8%** | 29/59 | +5.2% | +0.000 |
| 2023 | Recovery (+155%) | 29.4% | 12/17 | **+14.1%** | -0.361 |
| 2024 | **Bull** (+120%) | **23.1%** | 30/39 | +5.5% | -0.066 |
| 2025 | Sideways/volatile | 35.0% | 26/40 | +3.8% | +0.018 |
| 2026 | **Bear** (từ peak) | **60.0%** | 6/15 | +2.7% | -0.151 |

**Quan sát rõ ràng:** SELL accuracy đảo ngược hoàn toàn với trend thị trường:
- Bear market (2022, 2026): SELL đúng 50-60%
- Bull market (2023-2024): SELL đúng chỉ 23-29%

Model sinh SELL từ **tin tức tiêu cực**, nhưng trong bull market, tin xấu bị hấp thụ và giá vẫn tăng. Trong bear market, tin xấu xác nhận downtrend → SELL đúng.

---

## 4. Xu hướng thay đổi qua 3 versions

### 4.1 SELL count — cố định ở 170 dù thay đổi mọi thứ

| Version | Thay đổi | SELL Count | SELL Acc |
|---------|----------|-----------|----------|
| v1 Baseline | — | **170** | 39.4% |
| v2 Dense+Regime | Dense embedding, ±0.15 regime bonus | **170** | 41.1% |
| v3 Prompt+Trend | ret_14d + SELL WARNING + 3-condition gate | **170** | 39.4% |

SELL count **không đổi** qua 3 versions — 170/1,569 = 10.8% mỗi lần.

**Lý do:** Guard rail G8 (`SELL requires bear_regime`) là gate cuối cùng và quyết định. Số ngày `bear_regime=True` trong 2022–2026 cố định ở mức 170. Mọi thay đổi ở embedding, regime bonus, hay prompt đều không ảnh hưởng đến con số này vì G8 lọc ở tầng cuối dựa trên price data, không phải LLM output.

### 4.2 SELL accuracy theo năm — gần như không đổi

| Year/Version | v1 | v2 | v3 | Thay đổi |
|---|---|---|---|---|
| 2022 | 50.8% | **54.7%** | 50.8% | v2 tốt hơn nhờ regime bonus |
| 2023 | 29.4% | 29.4% | 29.4% | Không đổi |
| 2024 | 23.1% | 23.1% | 23.1% | **Không đổi** — vấn đề cốt lõi |
| 2025 | 35.0% | 35.0% | 35.0% | Không đổi |
| 2026 | 60.0% | 60.0% | 60.0% | Không đổi |

Chỉ có **2022 SELL cải thiện** (+3.9%) với v2 nhờ regime-conditional search — kNN tìm đúng bear cases hơn. Tất cả năm còn lại **bất biến** qua 3 versions.

### 4.3 Tại sao mỗi cách tiếp cận thất bại với SELL

| Cách tiếp cận | Lý do thất bại |
|---|---|
| **Dense factor embeddings** | Cải thiện kNN quality, nhưng SELL gate (G8) vẫn cho qua 170 ngày như cũ. Khi kNN tốt hơn, LLM vẫn ra SELL vì thấy tin xấu |
| **Regime-conditional search** | Giúp 2022 (+3.9%) vì phân tách đúng bear/bull kNN. Nhưng 2024 không đổi — bear_regime không trigger trong bull run dù SELL sai |
| **Prompt SELL WARNING** | LLM vẫn ra SELL signal ban đầu vì thấy sentiment âm. Guard rails sau đó giữ nguyên quyết định nếu bear_regime=True. Prompt cảnh báo không reach được những ngày đó |
| **min_sell_confidence=0.60** (đã revert) | Block SELL tốt cùng với SELL xấu — accuracy không tăng, coverage giảm |

---

## 5. Root cause phân tích

### 5.1 LLM phản ứng với tin tức, không phải trend

```
Sentiment âm (avg -0.084 khi sai, -0.109 khi đúng)
→ Gần như giống nhau → Sentiment không phân biệt được SELL đúng/sai
→ LLM dùng cùng 1 logic cho cả hai trường hợp
```

**Bằng chứng:** Avg sentiment khi SELL sai (-0.084) và đúng (-0.109) chênh nhau chỉ 0.025. Model không thể phân biệt "tin xấu trong bull market" vs "tin xấu xác nhận downtrend" thông qua sentiment score.

### 5.2 Confidence hoàn toàn vô dụng cho SELL discrimination

```
SELL đúng  → avg confidence = 0.5800
SELL sai   → avg confidence = 0.5800  ← giống hệt, đến 4 chữ số thập phân
```

Confidence scoring của guard rails được tính từ các factors (bull_regime, bear_regime, kNN averages) — nhưng những ngày SELL lọt qua G8 đều có `bear_regime=True`, nên evidence score gần như giống nhau cho tất cả SELL. Confidence không chứa thông tin để phân loại SELL đúng/sai.

**So sánh với BUY:**
```
BUY đúng  avg confidence = 0.711
BUY sai   avg confidence = 0.714 ← cũng gần như giống (chênh 0.003)
```

BUY cũng có vấn đề tương tự nhưng ít nghiêm trọng hơn vì BUY accuracy 54% (trên random) thay vì 39% (dưới random).

### 5.3 Guard rails là necessary but not sufficient

G8 cần `bear_regime=True` để cho phép SELL. Định nghĩa bear_regime hiện tại:
- `ret_14d ≤ -6%`, OR
- `avg30_kNN ≤ -4%`, OR
- `avg15_kNN ≤ -3% AND avg7_kNN ≤ -1.5%`, OR
- `MACD < 0 AND avg7_kNN ≤ -2%`

Vấn đề: trong 2024 bull run (BTC $40K→$100K), có những đợt **correction -10 đến -15%** trong 14 ngày (bear_regime trigger) rồi **tiếp tục tăng mạnh**. Model ra SELL tại đáy correction → giá phục hồi → SELL sai. Bear_regime phát hiện correction nhưng không biết correction là tạm thời hay bắt đầu downtrend thật.

---

## 6. Đặc điểm của SELL sai vs đúng

| Feature | SELL đúng (n=67) | SELL sai (n=103) | Phân biệt được? |
|---------|-----------------|-----------------|-----------------|
| Avg confidence | 0.580 | 0.580 | ❌ Không |
| Avg sentiment | -0.109 | -0.084 | ❌ Quá gần |
| Avg D+30 return | -5.6% | **+6.3%** | ✅ Rõ ràng |
| Market year | 2022,2026 dominant | 2024,2023 dominant | ✅ Rõ ràng |
| kNN avg_7d | Cần < -3.0% (G threshold) | Vượt gate nhờ bear_regime | Partial |

**Key insight:** D+30 return phân biệt rõ ràng nhất — nhưng đây là **future data**, không thể dùng trực tiếp. Tuy nhiên, nếu có thể **dự đoán D+30 context** từ features hiện tại (trend, on-chain data, kNN D+30 history), đây là signal tiềm năng mạnh nhất.

---

## 7. Tại sao cần thay đổi kiến trúc

Ba iterations cải thiện đều **không thay đổi** SELL 2024 (23.1%) và SELL overall (~39-41%). Lý do cơ bản:

```
Current flow:
LLM (sees bad news) → SELL
Guard rails (bear_regime check) → Pass/Block
Result: 170 SELL, ~39% accuracy

Problem: Bear_regime check là binary rule
→ Không học được "correction trong bull run" ≠ "real downtrend"
→ Confidence không phân biệt được case nào là đúng
→ Sentiment không đủ signal để phân loại
```

Để vượt qua 50% SELL accuracy cần một **learning component** — thứ có thể học pattern phức tạp từ data, không chỉ follow rules.

---

## 8. Roadmap cải thiện SELL

### Option A: ML Reranker (Khuyến nghị — 1-2 ngày)

Thêm LightGBM classifier sau guard rails, trước khi emit SELL:

```
LLM → SELL? → Guard rails (G1-G8) → [NEW] LightGBM → P(SELL đúng) ≥ 0.55? → Emit SELL : HOLD
```

**Training data:** 1,569 labeled rows từ backtest (170 SELL, 67 đúng / 103 sai).

**Features đề xuất:**
```python
features = [
    knn_avg_7d,        # avg D+7 return của 5 similar cases
    knn_avg_30d,       # avg D+30 return — proxy cho sustained trend
    knn_bullish_count, # số cases có ret7d > 0
    ret_14d,           # trend 14 ngày hiện tại
    ret_3d,            # short-term momentum
    rsi,               # RSI hiện tại
    macd_hist,         # MACD histogram
    sentiment_score,   # news sentiment
    fear_greed_index,  # FGI
    bear_regime_strength, # 0-4 (bao nhiêu điều kiện bear_regime đúng)
    month,             # seasonality (BTC thường dump tháng 5, pump tháng 10-11)
    days_since_ath,    # khoảng cách từ ATH — proxy cho bull/bear phase
]
label = actual_return_7d < 0  # 1=SELL đúng, 0=SELL sai
```

**Kỳ vọng:** SELL accuracy 39% → 48-53%. Dùng cross-validation để tránh overfitting trên 170 samples.

---

### Option B: Ensemble Voting (0.5 ngày)

Gọi LLM 3 lần với prompt variations:
- Prompt A: thiên về kNN evidence (conservative)
- Prompt B: thiên về technical indicators
- Prompt C: thiên về trend context (current v3)

SELL chỉ emit khi ≥ 2/3 votes = SELL. BUY chỉ cần 1/3.

**Trade-off:** Cost x3, latency x3. Kỳ vọng SELL → 43-47%.

---

### Option C: On-chain Signals Gate (3-5 ngày)

Thêm data source từ Glassnode/CryptoQuant làm **hard gate** bổ sung cho SELL:

```python
sell_allowed = (
    bear_regime                          # G8 hiện tại
    AND (
        exchange_netflow_7d > threshold  # BTC đang vào sàn (distribution)
        OR funding_rate < -0.01%         # funding âm = bearish sentiment thật
        OR whale_net_selling             # whale đang sell, không accumulate
    )
)
```

**Lý do:** On-chain data phân biệt được "correction trong bull run" (whale vẫn accumulate, funding dương) vs "real distribution" (whale bán, exchange inflow tăng). Đây là signal mà price + news không capture được.

**Kỳ vọng:** SELL accuracy → 55%+, nhưng SELL count giảm (chỉ ra khi có xác nhận on-chain).

---

## 9. Kết luận

| Câu hỏi | Trả lời |
|---------|---------|
| SELL có thể vượt 50% không? | Có, nhưng cần architecture change, không phải tuning |
| Dense embeddings có giúp SELL không? | Nhỏ (+1.7% ở 2022), không fix core problem |
| Prompt improvement có giúp SELL không? | Không — SELL count và accuracy không đổi |
| Guard rails có đủ không? | Không — binary rules không học được "correction vs downtrend" |
| Fix nhanh nhất là gì? | ML reranker dùng 1,569 labeled rows sẵn có |
| Fix tốt nhất dài hạn là gì? | On-chain signals gate — phát hiện real distribution |

Mô hình hiện tại nên được sử dụng như **BUY-only signal system** cho đến khi SELL vượt 50%. Tắt SELL hoặc tăng ngưỡng `bear_regime` thêm sẽ giảm false SELL nhưng cũng mất true SELL — trade-off không thuận lợi.
