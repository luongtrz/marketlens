# kNN-Returns Strategy Report
**Generated:** 2026-05-26 | **Symbol:** BTC | **Dataset:** 2022-01-01 → 2026-05-24 | **Records:** 1576

---

## 1. Motivation

LLM-based signal generation (qwen, kimi, deepseek) có vấn đề cơ bản: **non-deterministic**. Cùng một input, mỗi lần gọi có thể trả ra kết quả khác nhau, khiến signal không đáng tin cho systematic trading.

Giải pháp: thay LLM bằng **weighted average of future returns của top-k similar historical days** từ StockMem. Hoàn toàn deterministic, không cần API call, và dựa trên data thực tế đã xảy ra.

---

## 2. Cơ Chế Hoạt Động

```
StockMem kNN search (k=5, before_date)
        ↓
Top-5 similar days → future_return_1d / 3d / 7d / 15d / 30d
        ↓
Weighted average per record (normalize nếu thiếu horizon):
  w1d=0.40  w3d=0.30  w7d=0.15  w15d=0.10  w30d=0.05
        ↓
Overall avg = mean across 5 records
        ↓
Signal:  avg > +2%  → BUY
         avg < -2%  → SELL
         otherwise  → HOLD
        ↓
Confidence = 0.55 + min(distance_from_thr / 15, 0.35) + consensus_bonus
  consensus_bonus = (fraction agreeing − 0.5) × 0.10
  clamp → [0.50, 0.95]
```

**Similarity weights (kNN search):** factor_vec·0.35 + indicator_vec·0.20 + price_vec·0.45

---

## 3. Directional Accuracy (DA) — Full Backtest

### 3.1 D+7d (primary trading horizon)

| Signal | Count | Share | DA | Avg actual D+7d |
|--------|-------|-------|----|----------------|
| BUY | 656 | 42.2% | **59.6%** | +3.77% |
| SELL | 316 | 20.3% | **54.1%** | −1.33% |
| HOLD | 584 | 37.5% | 11.8% | +3.39% |
| **ALL** | **1556** | **100%** | **40.6%** | |

**Coverage (BUY+SELL): 62.5%**

> Note: HOLD "đúng" được định nghĩa là actual return nằm trong [−2%, +2%]. BTC thường di chuyển >2% trong 7 ngày nên HOLD DA thấp — điều này cho thấy model đang bỏ lỡ nhiều ngày tốt (HOLD avg = +3.39%).

### 3.2 D+7d — Threshold ±3% (để so sánh)

| Signal | Count | Share | DA | Avg actual D+7d |
|--------|-------|-------|----|----------------|
| BUY | 477 | 30.7% | **60.6%** | +3.63% |
| SELL | 222 | 14.3% | **56.8%** | −2.61% |
| HOLD | 857 | 55.1% | 20.3% | +3.36% |
| **ALL** | **1556** | **100%** | **37.9%** | |

**Coverage (BUY+SELL): 44.9%**

> HOLD DA (20.3%) thấp vì threshold ±3% nhưng BTC thường di chuyển >3% trong 7 ngày. Điểm mạnh: BUY DA cao nhất (60.6%) vì chỉ chọn những ngày kNN avg rất rõ ràng.

### 3.3 DA theo horizon (threshold ±2%)

| Horizon | BUY DA | SELL DA | HOLD DA | Overall DA | Coverage |
|---------|--------|---------|---------|-----------|----------|
| D+1d | 52.8% | 50.7% | 72.8% | 63.5% | 44.8% |
| D+3d | 52.8% | 54.5% | — | 53.4% | 44.8% |
| **D+7d** | **59.6%** | **54.1%** | 11.8% | 40.6% | **62.5%** |
| D+15d | 55.5% | 55.0% | — | 55.3% | 44.9% |

> D+7d là horizon tốt nhất cho BUY/SELL signals, dù weights bias về ngắn hạn (w1d=40%). Điều này phản ánh BTC regime persistence: xu hướng ngắn hạn thường kéo dài 7+ ngày.

### 3.4 So sánh với LLM models (D+7d, threshold ±3%)

| Model | BUY DA | SELL DA | Coverage |
|-------|--------|---------|----------|
| qwen3.5-plus | 52.9% | 37.4% | 41% |
| deepseek-v4-flash | 52.4% | 41.8% | 47% |
| kimi-k2.5 | ~51% | ~42% | 29% |
| **kNN-returns (3%)** | **60.6%** | **56.8%** | 44.9% |
| **kNN-returns (2%)** | **59.6%** | **54.1%** | **62.5%** |

**kNN-returns vượt LLM ~7–10pp trên BUY DA và ~12–19pp trên SELL DA**, đồng thời không cần gọi LLM API.

---

## 4. Threshold Analysis: 3% vs 2%

| Config | BUY | SELL | HOLD | Coverage | BUY DA | SELL DA |
|--------|-----|------|------|----------|--------|---------|
| thr = 3% | 477 (30.7%) | 222 (14.3%) | 857 (55.1%) | 44.9% | 60.6% | 56.8% |
| **thr = 2%** | **656 (42.2%)** | **316 (20.3%)** | **584 (37.5%)** | **62.5%** | **59.6%** | **54.1%** |

- **Threshold 2%** được chọn làm default: HOLD giảm từ 55% → 37.5%, coverage tăng từ 45% → 62.5%
- DA giảm nhẹ (−1pp BUY, −2.7pp SELL) vì các ngày borderline 2–3% avg ít chắc chắn hơn
- Đánh đổi được chấp nhận: nhiều signals hơn mà không mất đáng kể độ chính xác

---

## 5. Cấu Hình Production

```python
# main_controller/src/config.py
predict_provider: str = "knn_returns"   # default
knn_buy_threshold: float = 2.0          # avg > +2% → BUY
knn_sell_threshold: float = -2.0        # avg < -2% → SELL
knn_return_w1d: float = 0.40
knn_return_w3d: float = 0.30
knn_return_w7d: float = 0.15
knn_return_w15d: float = 0.10
knn_return_w30d: float = 0.05
```

Env vars để override:
```bash
MAIN_CONTROLLER_PREDICT_PROVIDER=knn_returns
MAIN_CONTROLLER_KNN_BUY_THRESHOLD=2.0
MAIN_CONTROLLER_KNN_SELL_THRESHOLD=-2.0
MAIN_CONTROLLER_KNN_RETURN_W1D=0.40
# ... etc
```

Fallback về LLM:
```bash
MAIN_CONTROLLER_PREDICT_PROVIDER=llm_gateway   # hoặc aihub
```

---

## 6. Điểm Mạnh và Hạn Chế

### Điểm mạnh
- **Deterministic**: cùng input → cùng output, không có variance từ LLM sampling
- **Không cần API call**: giảm latency ~500ms–2s, không có API cost
- **Interpretable**: explanation trả ra "top 5 similar days → avg +4.2% → BUY"
- **SELL accuracy cao hơn LLM**: 54.1% vs 37–42% — LLM thường bị G8 guardrail suppress SELL

### Hạn chế
- **HOLD nhiều ngày tốt bị bỏ qua**: HOLD avg return = +3.39% ≈ BUY avg +3.77% → model "bỏ lỡ" nhiều upside
- **Phụ thuộc vào stockmem coverage**: cần ≥k records trước ngày cần predict
- **Weights chưa được optimize**: w1d=40% là heuristic, chưa qua hyperparameter tuning
- **Không capture context news**: LLM có thể đọc sentiment bài viết, kNN chỉ dựa vào historical pattern

---

## 7. Recommendations

### Ngắn hạn (deploy ngay)
1. **Giữ threshold ±2%** — coverage 62.5% tốt hơn đáng kể so với 44.9% ở ±3%
2. **Monitor SELL signals** trong live — không có G8 guardrail suppress nên SELL sẽ fire nhiều hơn trước

### Cải thiện tiếp theo
3. **Weight optimization**: grid search w1d/w3d/w7d để maximize D+7d BUY DA
4. **Asymmetric thresholds**: test BUY_thr=1.5%, SELL_thr=2.5% (BTC bullish bias → easier BUY)
5. **Ensemble**: kNN-returns signal + news sentiment score → tránh BUY khi news rất bearish
6. **Confidence calibration**: kiểm tra correlation giữa confidence và actual accuracy

### Dài hạn
7. **Adaptive weights**: điều chỉnh w1d/w7d theo regime (bear/bull market)
8. **Return prediction** thay vì chỉ direction: dùng weighted avg return trực tiếp cho position sizing

---

## 8. Files

| File | Mô tả |
|------|-------|
| `main_controller/src/orchestrator/steps.py` | `_knn_returns_signal()` implementation |
| `main_controller/src/config.py` | Thresholds + weights config |
| `main_controller/src/orchestrator/pipeline.py` | `PipelineConfig` với knn params |
| `main_controller/src/api.py` | Wire config → PipelineConfig |
| `scripts/eval_knn_returns.py` | Offline DA evaluation script |

**Eval command:**
```bash
python scripts/eval_knn_returns.py --horizon 7d --buy-thr 2 --sell-thr 2
python scripts/eval_knn_returns.py --horizon 7d --buy-thr 3 --sell-thr 3   # compare
```

---

*Data source: `stockmem_records` PostgreSQL (1603 records, BTC 2022–2026)*
*Embedding: factor_vec(75d)·0.35 + indicator_vec(5d)·0.20 + price_vec(60d)·0.45*
