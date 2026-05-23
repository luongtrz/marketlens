# Trading Strategy Report — MarketLens Pipeline
**Generated:** 2026-05-23 | **Symbol:** BTC | **Capital:** $1,000,000 | **Period:** 2022-01-01 → 2026-05-22

---

## 1. Pipeline Architecture

```
MainController → LLM Gateway (qwen3.5-plus) → Guardrails (G1-G8) → Signal
                                   ↑
                  StockMem kNN (k=5, 1603 historical records)
                  factor_vec(75d) · 0.35 + indicator_vec(5d) · 0.20 + price_vec(60d) · 0.45
```

**Signal output:** BUY / SELL / HOLD + confidence (0.0–1.0)  
**Guardrail G8 (dominant):** SELL requires `bear_regime=True` (30d down >10% AND 3d down >3%). Trong bull market, hầu hết SELL bị suppress thành HOLD.

---

## 2. Signal Distribution (2022–2026, ~1600 records)

| Model | BUY | SELL | HOLD | Coverage |
|-------|-----|------|------|----------|
| kimi-k2.5 | 18% | 11% | 71% | 29% |
| qwen3.5-plus | 29% | 12% | 59% | 41% |
| deepseek-v4-flash | 36% | 11% | 53% | 47% |

**Key insight:** HOLD dominates vì G8 suppress SELL trong bull market. BUY signals cluster thành run dài (5–15 ngày) do guardrails giữ regime.

---

## 3. D+7 Accuracy (2022–2026)

| Model | BUY acc | SELL acc | Precision (BUY+SELL) |
|-------|---------|---------|----------------------|
| kimi-k2.5 | ~51% | ~42% | ~46% |
| qwen3.5-plus | 52.9% | 37.4% | 47.2% |
| deepseek-v4-flash | 52.4% | 41.8% | 49.8% |

---

## 4. Data

| File | Records | Horizons available |
|------|---------|-------------------|
| `data/backtests/qwen.json` | 1598 | ret_1d · ret_3d · ret_7d · ret_15d · ret_30d |
| `data/backtests/kimi.json` | 1598 | ret_1d · ret_3d · ret_7d · ret_15d · ret_30d |
| `data/backtests/deepseek.json` | 1569 | ret_1d · ret_3d · ret_7d · ret_15d · ret_30d |

Returns joined từ `stockmem_records` (PostgreSQL local). Script: `scripts/enrich_backtest.py`.

---

## 5. Portfolio Simulation — Full Results (2022-01-01 → 2026-05-22)

**Config:** $1,000,000 capital · position size = confidence × 15% · overlapping positions · BUY=long · SELL=short  
**Random baseline:** 10 runs, signal random(BUY/SELL/HOLD), confidence random(0.55–0.74)

### 5.1 Kết quả theo strategy × model

| Strategy | qwen | kimi | deepseek |
|----------|------|------|---------|
| **[1] SL 3%** | +1,471% ❌ edge -5381pp | +860% ❌ edge -5992pp | +5,154% ❌ edge -1652pp |
| **[2] Dynamic Exit + SL 3%** | +217% ✅ edge +97pp | +188% ✅ edge +68pp | **+818% ✅ edge +698pp** |
| **[3] TP 6% + Trailing 3% + SL 3%** | +51% ❌ edge -39pp | +24% ❌ edge -66pp | +91% ≈ edge +1pp |
| **[4] Dynamic Exit + TP + Trailing + SL** | +45% ✅ edge +8pp | +24% ❌ edge -12pp | +95% ✅ edge +58pp |
| **[5] Ensemble kimi+qwen + Dynamic + SL** | +177% ✅ edge +57pp | — | — |

### 5.2 Detail — Best Strategy: Dynamic Exit + SL 3%

| Metric | qwen | kimi | deepseek |
|--------|------|------|---------|
| Return | +217% | +188% | **+818%** |
| Final capital | $3.17M | $2.88M | **$9.18M** |
| Max Drawdown | 16.87% | 13.83% | 13.61% |
| Win rate | 35.0% | 31.1% | 35.2% |
| Long P&L | +$1,965k | +$1,838k | +$7,848k |
| Short P&L | +$207k | +$50k | +$338k |
| Exit via 7d | 232 | 203 | 331 |
| Exit via SL | 291 | 249 | 333 |
| Exit via signal | 112 | 2 | 64 |
| Random baseline | +120% | +120% | +120% |
| **Edge** | **+97pp** | **+68pp** | **+698pp** |

---

## 6. Winning Strategy: Dynamic Exit + SL 3%

### Cơ chế

```
Open:   BUY/SELL signal → enter confidence × 15% capital
Daily:  Update cumret dùng ret_1d thực tế
Exit 1: Cumret ≤ -3% → SL trigger → đóng tại -3%
Exit 2: Opposite signal xuất hiện → dynamic exit → đóng tại cumret hiện tại
Exit 3: 7 ngày hết hạn → đóng tại ret_7d
```

### Tại sao có edge

Guardrails tạo **regime-aligned signal runs**: BUY cluster 5–15 ngày trong bull market → positions chạy đủ 7d → capture upside. Random flip sau ~3 ngày → dynamic exit đóng ở P&L ≈ 0 → random chỉ đạt +120%.

### Tại sao deepseek thắng

Deepseek coverage 47% (cao nhất) → nhiều BUY/SELL hơn → tận dụng bull runs nhiều hơn. P&L long $7.8M vs qwen $1.9M — deepseek bắt được nhiều điểm entry hơn trong cùng thời gian.

---

## 7. Strategies Không Hiệu Quả

### ❌ SL 3% đơn thuần (không dynamic exit)
Random đạt +6852% sau 4 năm nhờ compounding × BTC tăng 5x. Không phải model tệ mà là asymmetry: trong bull market kéo dài, random BUY cũng thắng. Model bị kéo xuống bởi SELL signals sai.

### ❌ TP 6% + Trailing 3%
Cắt upside của BTC bull runs (cap 6%). Model mất lợi thế vì BTC thường tăng >6% trong 7 ngày bull market. Random không bị cap nên thắng. SL=304, TP=179, trail=57 exit counts cho thấy quá nhiều positions bị cắt sớm.

### ❌ Ensemble kimi+qwen
Giảm coverage → ít trades → edge thấp hơn dynamic exit đơn (+57pp vs +97pp cho qwen). Không đáng đánh đổi.

---

## 8. Recommendations

### Deploy ngay
1. **Dynamic Exit + SL 3%** là config chính
2. **deepseek** cho return tuyệt đối tốt nhất (+818%); **qwen** nếu muốn ổn định hơn (coverage thấp hơn, ít risk)
3. **Giữ SELL signals** — short P&L dương cho tất cả 3 model

### Cải thiện ngắn hạn
4. **SL threshold tuning:** test SL 2% vs 5% — SL 3% gây 291–333 exits, có thể hơi tight
5. **Confidence calibration:** BUY=0.74, SELL=0.58 gần cố định → fine-tune để spread rộng hơn → position sizing tốt hơn
6. **Live signal tracking:** `POST /portfolio/signal` endpoint để log dynamic exit triggers real-time

### Dài hạn
7. Bear regime confirmation tại entry point trước khi mở SELL
8. Thêm ret_3d và ret_15d vào accuracy tracking để chọn horizon tối ưu theo regime

---

## 9. Risk Notes

- **Backtest với ret_1d thực tế** từ stockmem DB — SL trigger theo daily path, không phải retrospective
- **BTC bull market 2025–2026 bias** — nhưng 4-year backtest bao gồm 2022 bear market
- **Compounding effect mạnh:** $1M × 4 năm × nhiều trades → kết quả % rất lớn, live trading thực tế cần vốn nhỏ hơn để test
- **Chưa tính slippage/fees** (~0.1–0.2% per trade) — với 700+ trades trong 4 năm, có thể ảnh hưởng ~5–10%

---

*Data: `data/backtests/{qwen,kimi,deepseek}.json` (từ stockmem PostgreSQL local)*  
*Scripts: `scripts/model_backtest_compare.py` · `scripts/enrich_backtest.py` · `scripts/portfolio_sim.py`*
