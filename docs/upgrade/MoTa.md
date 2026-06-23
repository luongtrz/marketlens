# Mô tả đề tài CEM-RAG cho MarketLens

Tài liệu này viết lại cách tiếp cận của paper theo hướng dễ hiểu, để có thể review bằng domain knowledge. Góc nhìn ở đây là "happy path": nếu thu thập dữ liệu thành công, huấn luyện đúng quy trình, không bị rò rỉ dữ liệu, và kết quả vượt baseline theo gate đã đặt ra, thì paper có câu chuyện khá mạnh để nộp đồ án tốt nghiệp.

Lưu ý quan trọng: tài liệu này không khẳng định chắc chắn được accept. "Có khả năng accept" chỉ hợp lý khi kết quả thực nghiệm thỏa các điều kiện: vượt baseline mạnh, có ablation rõ, có kiểm định thống kê, và có implementation/test đầy đủ.

---

## 1. Tên paper đề xuất

Tên ngắn:

**CEM-RAG: Crypto Event Memory Retrieval-Augmented Forecasting**

Tên đầy đủ hơn:

**CEM-RAG: A Point-in-Time Crypto Event Memory and Learned Retrieval Framework for Multi-Horizon Cryptocurrency Movement Forecasting**

Cách hiểu đơn giản:

> CEM-RAG là một hệ thống dự báo giá crypto bằng cách đọc tin tức, gom các tin cùng nói về một sự kiện, ghi nhớ các sự kiện lịch sử tương tự, rồi kết hợp với giá/chỉ báo kỹ thuật để đưa ra xác suất BUY, SELL, HOLD cho các horizon như 1 ngày, 3 ngày, 7 ngày, 15 ngày, 30 ngày.

Nhiệm vụ chính của paper:

- Primary task: dự báo hướng biến động 7 ngày của crypto.
- Output research: xác suất `UP`, `DOWN`, `NO_TRADE/HOLD`, không chỉ là một label BUY/SELL/HOLD.
- Output trading: chuyển xác suất thành trading signal bằng policy được tune trên validation.
- Output giải thích: top events, retrieved historical cases, và evidence trace.

---

## 2. Vấn đề hiện tại của MarketLens

Kiến trúc MarketLens gốc đã đúng hướng:

- `Crawler` lấy tin tức.
- `AIHub` tính sentiment, factor và prediction.
- `MarketData` lấy OHLCV/indicator.
- `FactorLedge` chuẩn hóa factor.
- `StockMem` lưu ký ức lịch sử và search kNN.
- `MainController` điều phối pipeline.

Nhưng research core hiện tại còn yếu nếu nộp paper:

- kNN đang là weighted cosine cố định, chưa học từ label tương lai.
- LLM tham gia quá nhiều vào final decision, dễ khó verify và khó chống hallucination.
- Guardrails hardcoded, khó chứng minh là tối ưu.
- Backtest BTC 2022-01-01 -> 2026-05-17 cho thấy D+7 accuracy tổng yếu, SELL precision yếu, HOLD quá nhiều.
- Tin tức mới đang ở dạng factor riêng lẻ, chưa thành "event memory" có tính lan tỏa, tính mới, và cluster theo ngày.

Do đó paper không nên bán câu chuyện "LLM + kNN + guardrails". Paper nên bán câu chuyện:

> Tài sản crypto di chuyển theo sự kiện. Một tin đơn lẻ không đủ, nhưng một cụm sự kiện có nhiều nguồn, có tính mới, và giống các cụm sự kiện lịch sử từng dẫn đến biến động giá thì có giá trị dự báo.

---

## 3. Ý tưởng cốt lõi của CEM-RAG

CEM-RAG gồm 5 thành phần:

1. **Event extraction**
   - Đọc tin tức và factor.
   - Biến text thành event có cấu trúc.
   - Ví dụ: "SEC delays Ethereum ETF decision" -> event group `regulatory`, event type `etf_delay`, polarity âm.

2. **Daily event memory**
   - Gom các bài cùng ngày thành cụm sự kiện.
   - Tính `article_count`, `source_count`, `source_diversity`, `novelty_7d`, `novelty_30d`.
   - Ví dụ: 1 tin nhỏ về ETF có thể yếu, nhưng 35 bài từ 12 nguồn trong cùng ngày về ETF approval là event mạnh.

3. **Learned retrieval**
   - Thay vì tìm ngày lịch sử bằng cosine cố định, học retriever từ label.
   - Positive pair: hai cửa sổ lịch sử có cùng hướng 7 ngày và regime biến động gần nhau.
   - Hard negative: nhìn bên ngoài rất giống nhau nhưng kết quả 7 ngày trái ngược.

4. **Multimodal forecasting**
   - Price encoder đọc 30 ngày OHLCV, RSI, MACD, volume, Fear & Greed, return.
   - Event encoder đọc FinBERT/CryptoBERT embedding và event features.
   - Retrieval encoder đọc các historical cases được lấy ra.
   - Fusion/cross-attention kết hợp 3 nguồn evidence.

5. **Calibrated trading policy**
   - Model trả xác suất `p_up`, `p_down`, `p_hold`.
   - Policy:
     - BUY nếu `p_up - p_down >= tau`.
     - SELL nếu `p_down - p_up >= tau`.
     - HOLD ngược lại.
   - `tau` chỉ được tune trên validation, không tune trên test.

---

## 4. Ví dụ để hiểu nhanh

### Ví dụ 1: Event tích cực rõ ràng, giá cũng ủng hộ

Ngày T có các headline:

- "BlackRock spot Bitcoin ETF sees record inflow"
- "Bitcoin ETF daily volume reaches new high"
- "Institutional demand pushes BTC higher"

Event memory tạo:

```json
{
  "date": "2025-03-18",
  "symbol": "BTC",
  "event_group": "institutional_adoption",
  "event_type": "etf_inflow",
  "polarity": 0.82,
  "article_count": 28,
  "source_count": 11,
  "novelty_score": 0.74,
  "confidence": 0.88
}
```

Market state:

- RSI = 58, chưa quá mua.
- MACD histogram dương.
- 14-day return = +5.2%.
- Volume tăng 30%.

Retriever tìm được các ngày lịch sử:

| Ngày lịch sử | Event tương tự | Similarity | Future 7d return |
|---|---|---:|---:|
| 2024-02-12 | ETF inflow + volume breakout | 0.91 | +6.4% |
| 2024-03-04 | Institutional demand | 0.87 | +4.1% |
| 2023-10-24 | ETF optimism | 0.84 | +8.7% |

Model output:

```json
{
  "horizon": "7d",
  "p_up": 0.61,
  "p_down": 0.21,
  "p_hold": 0.18,
  "signal": "BUY",
  "policy_tau": 0.12,
  "net_return_estimate": 4.8
}
```

Giải thích để đọc trong paper:

> Model BUY vì tin ETF có breadth cao, novelty cao, price momentum không bị overbought, và các memory tương tự trong quá khứ đều có 7-day return dương.

### Ví dụ 2: Tin xấu regulatory, SELL hợp lý

Ngày T có headline:

- "SEC sues major crypto exchange"
- "Exchange token falls after enforcement action"
- "Regulatory pressure spreads across altcoins"

Event:

```json
{
  "event_group": "regulatory",
  "event_type": "enforcement_action",
  "polarity": -0.86,
  "article_count": 41,
  "source_count": 15,
  "novelty_score": 0.67,
  "confidence": 0.91
}
```

Giá:

- BTC giảm 3 ngày liên tiếp.
- MACD histogram âm.
- Funding/market sentiment xấu.

Retriever:

| Ngày lịch sử | Event tương tự | Similarity | Future 7d return |
|---|---|---:|---:|
| 2023-06-05 | SEC enforcement | 0.89 | -5.9% |
| 2022-11-09 | Exchange crisis | 0.82 | -12.4% |
| 2025-02-03 | Regulatory shock | 0.78 | -3.7% |

Output:

```json
{
  "p_up": 0.19,
  "p_down": 0.58,
  "p_hold": 0.23,
  "signal": "SELL"
}
```

Giải thích:

> Model SELL vì event regulatory có độ lan tỏa cao, các case lịch sử cùng event group và cùng regime đều có negative future return, và price encoder cũng đang bearish.

### Ví dụ 3: Tin tích cực nhưng giá quá nóng, HOLD tốt hơn BUY

Headline:

- "Solana ecosystem activity hits yearly high"
- "Developers announce major upgrade"

Event có polarity dương, nhưng:

- RSI = 82.
- 14-day return = +32%.
- Retrieved cases cho thấy sau các đợt pump quá nóng, 7-day return thường mixed hoặc mean-revert.

Output:

```json
{
  "p_up": 0.43,
  "p_down": 0.34,
  "p_hold": 0.23,
  "signal": "HOLD",
  "policy_tau": 0.12
}
```

Tại sao HOLD?

- `p_up - p_down = 0.09`, nhỏ hơn `tau = 0.12`.
- Tin tốt có thật, nhưng risk reversal cao.
- Happy path của model không phải lúc nào có tin tốt cũng BUY.

### Ví dụ 4: Giá bullish nhưng event xấu, model cân conflict

Headline:

- "Bridge exploit drains $200M"
- "Security concern spreads to DeFi tokens"

Giá:

- BTC vẫn trên SMA, momentum 14 ngày dương.

Evidence:

- Price encoder bullish.
- Event encoder bearish.
- Retrieval trace mixed.

Output mong muốn:

```json
{
  "p_up": 0.38,
  "p_down": 0.37,
  "p_hold": 0.25,
  "signal": "HOLD"
}
```

Điểm hay để viết paper:

> CEM-RAG biết tránh giao dịch khi các modality mâu thuẫn, thay vì ép ra BUY/SELL.

### Ví dụ 5: Tin nhiều nhưng không mới, novelty thấp

Headline lặp lại:

- "Bitcoin trades sideways ahead of Fed meeting"
- "Investors await macro data"

Trong 30 ngày qua đã có nhiều event `macro_wait_and_see`, nên:

```json
{
  "event_group": "macro",
  "event_type": "fed_expectation",
  "article_count": 17,
  "source_count": 8,
  "novelty_score": 0.08,
  "polarity": -0.05
}
```

Output:

```json
{
  "p_up": 0.34,
  "p_down": 0.32,
  "p_hold": 0.34,
  "signal": "HOLD"
}
```

Ý nghĩa:

> Tin nhiều không đồng nghĩa tin có giá trị. Novelty giúp phân biệt "tin mới có shock" với "tin lặp lại".

---

## Bổ sung: Ý nghĩa kết quả kNN-Returns hiện tại

Báo cáo `knn_returns_strategy_report.md` cho thấy một kết quả rất quan trọng cho hướng đồ án: **chỉ cần dùng historical memory từ StockMem, không cần LLM final decision, hệ thống đã tạo được tín hiệu BUY/SELL tốt hơn các mô hình LLM hiện tại** trên BTC giai đoạn 2022-01-01 -> 2026-05-24.

Tóm tắt kết quả chính:

| Cấu hình | BUY DA | SELL DA | BUY avg D+7d | SELL avg D+7d | Coverage |
|---|---:|---:|---:|---:|---:|
| Default weights, threshold ±2% | 59.6% | 54.1% | +3.77% | -1.33% | 62.5% |
| Old Bayesian, threshold ±2% | 58.5% | 54.9% | +4.11% | -2.23% | 59.6% |
| New Bayesian, threshold ±2% | 59.7% | 57.5% | +4.46% | -3.38% | 58.2% |
| New Bayesian, threshold ±2.5% | 60.1% | 58.0% | +4.77% | -4.08% | 50.0% |
| New Bayesian, threshold ±3% | 61.3% | 57.1% | +4.91% | -4.28% | 42.8% |

Diễn giải dễ hiểu:

- Khi kNN-Returns nói BUY, khoảng 6/10 lần BTC thật sự tăng sau 7 ngày.
- Khi kNN-Returns nói SELL, khoảng 5.7/10 lần BTC thật sự giảm sau 7 ngày với cấu hình New Bayesian ±2%.
- Tín hiệu SELL cải thiện rõ so với LLM hiện tại, vì LLM/guardrail cũ thường quá dè dặt với SELL trong bull market.
- D+7 là horizon tốt nhất: D+1 nhiễu quá nhiều, còn D+7 phản ánh regime persistence tốt hơn.
- Search weights mới cho thấy `factor_vec` và `indicator_vec` quan trọng hơn `price_vec`: weight price giảm từ 0.45 xuống 0.1416, trong khi factor tăng lên 0.5444.

Insight quan trọng cho CEM-RAG:

> Kết quả này là bằng chứng ban đầu rằng "memory retrieval" có giá trị thật. Nếu một kNN deterministic còn vượt LLM final-decision, thì hướng nâng cấp lên event-memory retriever học được, có event novelty/source breadth và calibration, là một hướng có cơ sở.

### kNN-Returns đang làm gì?

Thuật toán hiện tại có thể hiểu đơn giản như sau:

1. Với ngày hiện tại T, hệ thống tìm top-5 ngày lịch sử giống nhất.
2. Độ giống được tính từ 3 nhóm feature:
   - `factor_vec`: tin tức/factor/event taxonomy.
   - `indicator_vec`: RSI, MACD, sentiment, Fear & Greed, price change.
   - `price_vec`: pattern giá/volume 20 ngày gần nhất.
3. Với mỗi ngày lịch sử giống nhất, xem sau ngày đó BTC đã tăng/giảm bao nhiêu ở các horizon 1d, 3d, 7d, 15d, 30d.
4. Tính weighted average return.
5. Nếu average return > +2% thì BUY, < -2% thì SELL, còn lại HOLD.

Ví dụ ngắn:

```text
Ngày hiện tại: 2025-03-18
Top-5 ngày lịch sử tương tự:
  Case 1: future weighted return +4.2%
  Case 2: future weighted return +3.1%
  Case 3: future weighted return +6.0%
  Case 4: future weighted return +1.8%
  Case 5: future weighted return +2.9%

overall_avg = (+4.2 + 3.1 + 6.0 + 1.8 + 2.9) / 5 = +3.6%
threshold = +2%
=> BUY
```

Ý nghĩa: ngày hiện tại giống các ngày lịch sử mà sau đó BTC thường tăng khá mạnh, nên model phát tín hiệu BUY.

### Ví dụ minh họa 1: BUY đúng

Giả sử ngày T có bối cảnh:

- Tin ETF inflow tích cực.
- RSI = 58, chưa quá mua.
- MACD histogram dương.
- Fear & Greed tăng nhưng chưa cực đoan.

Top-5 ngày tương tự:

| Case | Bối cảnh lịch sử | Weighted future return | D+7 return |
|---|---|---:|---:|
| 2024-02-12 | ETF inflow + volume breakout | +4.2% | +6.4% |
| 2024-03-04 | Institutional demand | +3.1% | +4.1% |
| 2023-10-24 | ETF optimism | +6.0% | +8.7% |
| 2024-02-26 | Momentum continuation | +1.8% | +2.5% |
| 2023-12-04 | Bullish factor cluster | +2.9% | +3.8% |

Tính:

```text
overall_avg = +3.6%
BUY threshold = +2.0%
=> Signal = BUY
```

Nếu sau 7 ngày BTC tăng +4.6%, đây là BUY đúng. Trong report, nhóm BUY của New Bayesian ±2% có average D+7 return +4.46%, nghĩa là các BUY signal không chỉ đúng hướng mà còn chọn được các ngày có upside khá tốt.

### Ví dụ minh họa 2: SELL đúng

Giả sử ngày T có bối cảnh:

- Tin regulatory enforcement lan rộng.
- MACD histogram âm.
- BTC giảm 3 ngày liên tiếp.
- Các factor giống các giai đoạn thị trường từng giảm.

Top-5 ngày tương tự:

| Case | Bối cảnh lịch sử | Weighted future return | D+7 return |
|---|---|---:|---:|
| 2023-06-05 | SEC enforcement | -3.5% | -5.9% |
| 2022-11-09 | Exchange crisis | -8.2% | -12.4% |
| 2025-02-03 | Regulatory shock | -2.7% | -3.7% |
| 2022-05-09 | Risk-off cascade | -4.1% | -6.3% |
| 2024-04-12 | Leverage flush | -1.9% | -2.8% |

Tính:

```text
overall_avg = -4.08%
SELL threshold = -2.0%
=> Signal = SELL
```

Nếu sau 7 ngày BTC giảm -3.5%, đây là SELL đúng. Đây là điểm mạnh nhất của report: SELL DA của New Bayesian ±2% đạt 57.5%, và ±2.5% đạt 58.0%, cao hơn đáng kể so với LLM models trong báo cáo.

### Ví dụ minh họa 3: HOLD vì tín hiệu chưa đủ mạnh

Giả sử ngày T có:

- Một số tin tích cực nhưng không mới.
- RSI trung tính.
- Retrieved cases lẫn lộn: có ngày tăng, có ngày giảm.

Top-5 weighted future return:

```text
Case 1: +1.4%
Case 2: -0.7%
Case 3: +0.9%
Case 4: +1.8%
Case 5: -0.2%

overall_avg = +0.64%
```

Với threshold ±2%:

```text
-2% < +0.64% < +2%
=> HOLD
```

Ý nghĩa: hệ thống không thấy edge đủ lớn để trade. Điều này giúp tránh overtrading, nhưng cũng có mặt trái: report cho thấy HOLD avg D+7 vẫn dương khoảng +3.3% trong giai đoạn BTC có bullish drift, tức là một phần upside bị bỏ lỡ.

### Ví dụ minh họa 4: Threshold cao hơn thì ít trade nhưng chắc hơn

Cùng một ngày có `overall_avg = +2.6%`:

```text
Threshold ±2.0%  => BUY
Threshold ±2.5%  => BUY
Threshold ±3.0%  => HOLD
```

Trade-off:

- Threshold ±2%: coverage cao hơn, nhiều cơ hội hơn, nhưng DA thấp hơn một chút.
- Threshold ±3%: chỉ trade khi historical evidence rất mạnh, DA cao hơn, nhưng bỏ lỡ nhiều cơ hội.

Report phản ánh đúng trade-off này:

| Threshold New Bayesian | BUY DA | SELL DA | Coverage |
|---|---:|---:|---:|
| ±2% | 59.7% | 57.5% | 58.2% |
| ±2.5% | 60.1% | 58.0% | 50.0% |
| ±3% | 61.3% | 57.1% | 42.8% |

Nếu mục tiêu là production có đủ tín hiệu, ±2% hợp lý. Nếu mục tiêu là case study chất lượng cao hoặc demo ít trade nhưng chắc hơn, ±2.5% hoặc ±3% dễ giải thích hơn.

### Ý nghĩa đối với hướng đồ án

Kết quả này nên được đưa vào đồ án như một milestone:

- **Baseline mới mạnh hơn LLM**: kNN-Returns nên trở thành baseline chính thay cho LLM + guardrails.
- **Ủng hộ event-memory retrieval**: vì factor/indicator đóng vai trò lớn hơn price pattern, việc xây event memory chi tiết hơn có khả năng cải thiện tiếp.
- **Ủng hộ learned retriever**: Bayesian weights mới đã tốt hơn heuristic; bước tiếp theo là học retriever bằng supervised contrastive learning thay vì chỉ optimize 3 trọng số.
- **Ủng hộ 7-day task**: D+7 là horizon có ý nghĩa nhất trong report, phù hợp chọn làm primary task.

### Cảnh báo khi diễn giải kết quả

Không nên kết luận ngay rằng strategy đã production-ready. Cần kiểm tra thêm:

1. **Temporal holdout**
   - Nếu weights và threshold được optimize trên toàn bộ 2022-2026 rồi báo cáo lại trên cùng tập đó, kết quả có thể optimistic.
   - Cần train/validation/test theo thời gian: ví dụ train 2022-2023, validation 2024, test 2025-2026.

2. **Leakage từ future returns chưa matured**
   - Khi dự báo ngày T, nếu dùng `future_return_30d` của một ngày lịch sử T' mà `T' + 30d > T`, thì tại thời điểm T ta chưa thể biết return 30 ngày đó.
   - Quy tắc an toàn: nếu dùng horizon 30d của case lịch sử, case đó phải có `case_date + 30d <= prediction_date`.

3. **Trading PnL sau phí**
   - DA tốt chưa chắc Sharpe tốt.
   - Cần tính net cumulative return, Sharpe, Sortino, max drawdown, turnover, transaction cost.

4. **So với buy-and-hold**
   - HOLD avg D+7 cũng dương vì BTC có positive drift.
   - Cần chứng minh BUY/SELL policy tạo edge sau phí, không chỉ hưởng trend chung của BTC.

5. **Multi-asset generalization**
   - Kết quả hiện tại là BTC-only.
   - Đồ án mạnh hơn nếu mở rộng sang ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, DOT.

---

## 5. Contribution đề xuất cho paper

### Contribution 1: Point-in-time crypto event memory

Đóng góp:

- Xây dựng `DailyEventState` cho từng ngày, từng symbol.
- Mỗi event có `event_type`, `event_group`, `entities`, `polarity`, `novelty_score`, `source_count`, `article_count`, `confidence`.
- Bảo đảm point-in-time: ngày T chỉ được dùng tin xuất hiện trước cutoff ngày T.

Ví dụ:

Thay vì lưu:

```text
"ETF inflow bullish, BlackRock, BTC, institutional demand"
```

Hệ thống lưu:

```json
{
  "event_group": "institutional_adoption",
  "event_type": "etf_inflow",
  "entities": ["BlackRock", "Bitcoin ETF", "BTC"],
  "polarity": 0.82,
  "source_count": 11,
  "article_count": 28,
  "novelty_score": 0.74
}
```

Giá trị research:

- Làm cho tin tức thành feature có thể kiểm định.
- Giảm phụ thuộc vào LLM final reasoning.
- Tạo được explanation trace cho paper và system.

### Contribution 2: Dissemination-aware event features cho crypto

Ý tưởng:

- Một event có tác động mạnh hơn nếu nó được nhiều nguồn độc lập đưa tin.
- `source_count` và `source_diversity` có thể đo market attention.

Ví dụ:

| Case | Article count | Source count | Novelty | Diễn giải |
|---|---:|---:|---:|---|
| 1 blog nhỏ nói BTC bullish | 1 | 1 | 0.3 | Yếu |
| 30 bài từ nhiều nguồn nói ETF inflow kỷ lục | 30 | 12 | 0.8 | Mạnh |
| 40 bài lặp lại chuyện Fed đã nói 2 tuần | 40 | 15 | 0.1 | Nhiều nhưng không mới |

Giá trị research:

- Khác với sentiment đơn thuần.
- Gắn với ý tưởng news breadth/dissemination trong financial forecasting.

### Contribution 3: Learned event-memory retriever

Đóng góp:

- Baseline cũ `MarketLens-kNN`: weighted cosine trên factor, indicator, price.
- CEM-RAG: học retriever để lấy ra các lịch sử có giá trị dự báo hơn.

Ví dụ triplet training:

Query:

```text
2024-03-04 BTC, ETF inflow, volume breakout, RSI 61, ret_14d +8%
future_return_7d = +6.4%
```

Positive:

```text
2023-10-24 BTC, ETF optimism, volume breakout, future_return_7d = +8.7%
```

Hard negative:

```text
2024-01-11 BTC, ETF news, RSI 83, sell-the-news, future_return_7d = -7.2%
```

Mục tiêu:

- Query gần Positive hơn Hard negative.
- Hard negative rất quan trọng vì crypto hay có "same news, opposite outcome" tùy theo regime.

### Contribution 4: Multimodal price-event-retrieval fusion

Đóng góp:

- Kết hợp price tokens, event tokens, retrieved cases.
- Model không chỉ đọc text, không chỉ đọc giá.

Ví dụ fusion:

| Evidence | Tín hiệu | Điểm |
|---|---|---:|
| Price encoder | Momentum 14 ngày dương, MACD dương | + |
| Event encoder | ETF inflow breadth cao | + |
| Retrieval encoder | 5 case tương tự, avg future return +4.8% | + |
| Policy | `p_up - p_down >= tau` | BUY |

Ví dụ ngược:

| Evidence | Tín hiệu | Điểm |
|---|---|---:|
| Price encoder | RSI 84, pump quá nóng | - |
| Event encoder | Tin upgrade tích cực | + |
| Retrieval encoder | Case tương tự mixed | 0 |
| Policy | chênh lệch xác suất nhỏ | HOLD |

### Contribution 5: Calibrated trading policy thay guardrails hardcoded

Đóng góp:

- Không dùng rule cảm tính như "nếu RSI > 70 thì..."
- Model trả xác suất, policy tune threshold trên validation.
- Dùng coverage constraint để tránh model HOLD quá nhiều hoặc trade quá nhiều.

Ví dụ:

```text
tau grid: 0.05, 0.06, ..., 0.50
constraint: directional coverage 25% -> 60%
objective: maximize validation net Sharpe
```

Nếu `tau=0.08`:

- Coverage 75%, trade quá nhiều, cost cao.

Nếu `tau=0.30`:

- Coverage 12%, quá ít signal, không có tác dụng trading.

Nếu `tau=0.14`:

- Coverage 42%, Sharpe validation cao nhất.

Thì dùng `tau=0.14` cho test, không tune lại.

### Contribution 6: Auditable intelligent system, hợp scope đồ án tốt nghiệp

Paper không chỉ là model offline. Nó là một intelligent system có:

- Microservice implementation.
- Data contracts.
- Prediction trace.
- Retrieval trace.
- Backtesting và ablation.
- Trading policy có calibration.

Đây là điểm phù hợp đồ án tốt nghiệp hơn một paper "chỉ fine-tune model".

---

## 6. Tại sao có khả năng phù hợp đồ án tốt nghiệp

Theo scope chính thức của đồ án tốt nghiệp, journal quan tâm đến thiết kế, phát triển, testing, implementation của expert/intelligent systems và các ứng dụng như finance, stock trading, information retrieval, text mining, neural networks. CEM-RAG có thể map trực tiếp vào các điểm đó:

| đồ án tốt nghiệp fit | CEM-RAG đáp ứng như thế nào |
|---|---|
| Intelligent system design | Kiến trúc Crawler, FactorLedge, StockMem, AIHub, MainController |
| Finance/stock trading | Bài toán dự báo crypto movement và trading signal |
| Text mining | Structured event extraction từ news |
| Information retrieval | Learned event-memory retrieval |
| Neural networks | Price encoder, event encoder, cross-attention fusion |
| Implementation/testing | Microservice code, endpoint, schema, backtest, ablation |
| Practical guidelines | Policy threshold, point-in-time constraint, transaction cost |

Câu chuyện paper mạnh hơn nếu chứng minh được:

1. **Không chỉ accuracy**: có MCC, calibration, coverage, Sharpe, Sortino, drawdown, turnover.
2. **Không chỉ one asset**: có BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, DOT.
3. **Không chỉ one split**: có train/validation/test theo thời gian và walk-forward.
4. **Không chỉ model mới**: có ablation cho event memory và learned retrieval.
5. **Không chỉ LLM story**: LLM dùng để extract/explain, final signal do probabilistic model và policy quyết định.

Happy path cho accept:

- CEM-RAG vượt XGBoost/PatchTST/TFT/FinBERT text-only/current MarketLens.
- MCC tăng >= 0.05 trên test.
- Net Sharpe tăng >= 0.25.
- Bootstrap CI/McNemar ủng hộ cải tiến.
- Ablation cho thấy bỏ event memory hoặc learned retrieval thì metric giảm rõ.
- Paper có case study để giải thích đúng/sai signal.

---

## 7. Cách làm thí nghiệm từng bước

### Bước 0: Đóng băng câu hỏi nghiên cứu

Đặt 4 research questions:

RQ1. Event memory có cải thiện crypto movement forecasting so với price-only và sentiment-only không?

RQ2. Learned retriever có lấy historical cases tốt hơn fixed weighted kNN không?

RQ3. Cross-modal fusion của price, event, retrieval có tốt hơn từng modality riêng lẻ không?

RQ4. Trading policy được tune trên validation có cải thiện net Sharpe sau transaction cost không?

Giả thuyết happy path:

- H1: CEM-RAG có MCC cao hơn baseline mạnh nhất ít nhất 0.05.
- H2: CEM-RAG có net Sharpe cao hơn baseline mạnh nhất ít nhất 0.25.
- H3: Removing event memory làm giảm macro-F1/MCC.
- H4: Learned retriever tốt hơn random/BM25/numeric/kNN retrieval.

### Bước 1: Thu thập dữ liệu

Tài sản:

```text
BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, DOT
```

Khoảng thời gian happy path:

```text
2018-01-01 -> 2026-05-17
```

Nguồn nội bộ:

- Supabase news.
- Daily factor snapshots.
- Binance OHLCV.
- Fear & Greed.
- Current StockMem records.

Nguồn public có thể thêm:

- CryptoLin corpus.
- DLT-Sentiment-News.
- CrypTop12.
- CryptoVision nếu license cho phép.

Output mong muốn sau bước này:

```text
artifacts/raw/news/{symbol}.jsonl
artifacts/raw/ohlcv/{symbol}.parquet
artifacts/raw/factors/{symbol}.jsonl
artifacts/raw/fear_greed.csv
```

Ví dụ 1 row news:

```json
{
  "symbol": "BTC",
  "url": "https://example.com/btc-etf-inflow",
  "source": "CoinDesk",
  "published_at": "2024-03-04T10:15:00Z",
  "title": "Bitcoin ETF inflows hit record high",
  "text": "...",
  "crawl_time": "2024-03-04T10:20:00Z"
}
```

### Bước 2: Tạo point-in-time daily snapshot

Nguyên tắc:

- Bạn đang dự báo tại ngày T.
- Chỉ được dùng tin có `published_at <= cutoff_time(T)`.
- Chỉ được dùng giá/indicator đã biết tại ngày T.
- Không dùng future return khi tạo feature.
- Retrieval phải có `before_date < T`.

Ví dụ:

```text
Predict date: 2025-06-10
Cutoff: 2025-06-10 23:59:59 UTC
Allowed news: news published before cutoff
Forbidden news: article published 2025-06-11
Allowed retrieval: records before 2025-06-10
Forbidden retrieval: 2025-06-10 or later
```

Output 1 daily record:

```json
{
  "date": "2025-06-10",
  "symbol": "BTC",
  "price_window_30d": "...",
  "indicators": {
    "rsi": 57.2,
    "macd_hist": 1.15,
    "volume_change": 0.31,
    "fear_greed": 68
  },
  "news_ids": ["n1", "n2", "n3"],
  "event_state": "...",
  "future_return_7d": 4.6
}
```

`future_return_7d` chỉ dùng cho training/evaluation, không được đưa vào model inference.

### Bước 3: Extract event từ tin tức

Dùng LLM/FinBERT/CryptoBERT để biến text thành event có cấu trúc.

Input:

```text
Title: Ethereum ETF decision delayed by SEC
Text: The SEC postponed a decision on multiple spot Ethereum ETF applications...
```

Output event:

```json
{
  "symbol": "ETH",
  "event_group": "regulatory",
  "event_type": "etf_delay",
  "entities": ["SEC", "Ethereum ETF"],
  "polarity": -0.62,
  "confidence": 0.84
}
```

Thêm ví dụ event mapping:

| Text | Event group | Event type | Polarity |
|---|---|---|---:|
| "Binance resumes withdrawals" | exchange | withdrawal_resume | +0.45 |
| "Stablecoin depegs to $0.94" | stablecoin | depeg | -0.90 |
| "Fed signals fewer rate cuts" | macro | hawkish_fed | -0.55 |
| "Protocol completes major upgrade" | technology | network_upgrade | +0.50 |
| "Bridge exploit drains funds" | security | exploit | -0.85 |
| "Whale transfers BTC to exchange" | on_chain | exchange_inflow | -0.40 |

### Bước 4: Cluster same-day news thành event memory

Nếu 20 bài cùng nói về ETF inflow, không nên coi là 20 event riêng. Nên cluster thành 1 event group trong ngày.

Input trong ngày:

```text
1. CoinDesk: BTC ETF inflows hit record
2. The Block: BlackRock ETF attracts $900M
3. Decrypt: Institutional BTC demand grows
4. CNBC: Bitcoin funds see record volume
```

Daily event state:

```json
{
  "date": "2024-03-04",
  "symbol": "BTC",
  "article_count": 4,
  "source_count": 4,
  "source_diversity": 1.0,
  "novelty_7d": 0.78,
  "novelty_30d": 0.54,
  "events": [
    {
      "event_group": "institutional_adoption",
      "event_type": "etf_inflow",
      "polarity": 0.82,
      "article_count": 4,
      "source_count": 4,
      "novelty_score": 0.78,
      "confidence": 0.88
    }
  ]
}
```

Cách review bằng domain knowledge:

- Event group có đúng không?
- Polarity có đúng theo tài sản không?
- "Fed hawkish" có thể bearish với BTC nhưng tác động khác với stablecoin.
- "Exchange outflow" có thể bullish nếu là rút về cold wallet, nhưng bearish nếu liên quan panic withdrawal.

### Bước 5: Tạo label cho các horizon

Return:

```text
future_return_7d = (close_{T+7} - close_T) / close_T * 100
```

Label ví dụ với hold band 0.5%:

| future_return_7d | Label |
|---:|---|
| +4.6% | UP |
| -3.2% | DOWN |
| +0.2% | HOLD/NO_TRADE |

Ví dụ:

```text
BTC close ngày T = 60,000
BTC close ngày T+7 = 63,000
future_return_7d = (63,000 - 60,000) / 60,000 * 100 = +5.0%
label_7d = UP
```

Cần tạo label cho:

```text
1d, 3d, 7d, 15d, 30d
```

Nhưng paper nên lấy 7d làm primary task để tránh bị loãng.

### Bước 6: Chia train/validation/test theo thời gian

Split:

```text
Train:      2018-01-01 -> 2023-12-31
Validation: 2024-01-01 -> 2024-12-31
Test:       2025-01-01 -> 2026-05-17
```

Lý do:

- Không shuffle vì time-series sẽ rò rỉ thông tin.
- Validation dùng để tune `tau`, hyperparameters, calibration.
- Test chỉ dùng một lần để báo cáo final.

Thêm walk-forward:

```text
Train 2018-2023, validate 2024-01, test 2024-02
Train 2018-2024-01, validate 2024-02, test 2024-03
...
```

Giá trị:

- Chứng minh model không chỉ may mắn trong một split.
- Gần với cách system sẽ vận hành thật.

### Bước 7: Train learned retriever

Script hiện có:

```bash
python stockmem/scripts/train_learned_retriever.py \
  --data stockmem/data/real_optimizer.json \
  --output stockmem/config/learned_retriever.json \
  --horizon 7d
```

Input cần có:

```json
{
  "date": "2024-03-04",
  "symbol": "BTC",
  "factor_vec": [0, 1, 0, "..."],
  "indicator_vec": [0.55, 0.22, "..."],
  "price_vec": [0.01, 0.02, "..."],
  "event_features": [1, 28, 11, 0.78, 0.54, 0.82, 0.88, "..."],
  "future_return_7d": 6.4
}
```

Happy path metric cho retriever:

| Retriever | Hit same direction@5 | Avg retrieved future return sign accuracy |
|---|---:|---:|
| Random retrieval | 50% | 50% |
| BM25/text embedding | 54% | 53% |
| Numeric Euclidean | 56% | 55% |
| MarketLens-kNN | 58% | 57% |
| Learned CEM retriever | 63%+ | 62%+ |

Ví dụ hard negative cần có:

```text
Query: ETF optimism + RSI 58 + volume up -> future +6%
Hard negative: ETF approval day + RSI 85 + sell-the-news -> future -7%
```

Nếu retriever không học được hard negative, paper sẽ yếu vì nó chỉ tìm "keyword giống nhau".

### Bước 8: Train forecasting model

Paper-grade model nên có:

1. Price encoder:
   - Input: 30 ngày OHLCV, RSI, MACD, volume, Fear & Greed, returns.
   - Kiến trúc: PatchTST-style hoặc temporal Transformer.

2. Event encoder:
   - Input: FinBERT/CryptoBERT embedding của event text.
   - Thêm event metadata: polarity, novelty, source_count, article_count.

3. Retrieval encoder:
   - Input: top-k historical windows.
   - Mỗi case gồm features tại ngày lịch sử và future outcome để model học analogical evidence.

4. Fusion:
   - Cross-attention giữa price tokens, event tokens, retrieval tokens.

5. Output:
   - `p_up`, `p_down`, `p_hold`.
   - Horizon: 1d, 3d, 7d, 15d, 30d.

Minh họa:

```text
Price tokens:
  [day -29], [day -28], ..., [day 0]

Event tokens:
  [ETF inflow], [Fed hawkish], [exchange outflow]

Retrieved tokens:
  [case 2024-03-04: +6.4%],
  [case 2023-10-24: +8.7%],
  [case 2024-01-11: -7.2%]

Fusion output:
  p_up=0.61, p_down=0.21, p_hold=0.18
```

Ghi chú về code hiện tại:

- Repo đã có CEM-RAG code path nhẹ trong `aihub/src/predict/cem_rag.py`.
- Bản hiện tại là scaffold/probability policy để wire schema, endpoint, trace và evaluation.
- Để paper mạnh, cần thay hoặc mở rộng thành neural model đúng như mô tả trên.

### Bước 9: Calibration

Vì sao cần calibration?

Trong trading, `p_up=0.70` mà thực tế chỉ đúng 52% thì confidence vô nghĩa.

Metrics:

- Brier score.
- ECE.
- Reliability diagram.

Ví dụ calibration:

| Bucket confidence | Số mẫu | Accuracy thực |
|---:|---:|---:|
| 0.50-0.60 | 300 | 55% |
| 0.60-0.70 | 220 | 64% |
| 0.70-0.80 | 120 | 71% |

Đây là happy path tốt: confidence tăng thì accuracy thực cũng tăng.

Bad case:

| Bucket confidence | Số mẫu | Accuracy thực |
|---:|---:|---:|
| 0.70-0.80 | 120 | 51% |

Nếu gặp bad case, paper cần calibration lại bằng temperature scaling/isotonic trên validation.

### Bước 10: Tune trading policy trên validation

Dùng script:

```bash
python scripts/cem_rag_evaluate.py \
  --predictions artifacts/cem_rag_predictions_val.jsonl \
  --horizon 7d \
  --bootstrap 1000
```

Policy:

```text
BUY  if p_up - p_down >= tau
SELL if p_down - p_up >= tau
HOLD otherwise
```

Grid:

```text
tau = 0.05, 0.06, ..., 0.50
```

Constraint:

```text
directional coverage between 25% and 60%
```

Ví dụ validation:

| tau | Coverage | Directional precision | Net Sharpe |
|---:|---:|---:|---:|
| 0.05 | 72% | 51% | 0.31 |
| 0.10 | 51% | 56% | 0.74 |
| 0.14 | 42% | 59% | 0.91 |
| 0.22 | 24% | 63% | 0.62 |

Chọn `tau=0.14` vì:

- Coverage 42% nằm trong 25-60%.
- Sharpe cao nhất.

Sau đó dùng `tau=0.14` cho test.

### Bước 11: Evaluate trên test

Chạy:

```bash
python scripts/cem_rag_evaluate.py \
  --predictions artifacts/cem_rag_predictions_test.jsonl \
  --horizon 7d \
  --tau 0.14 \
  --bootstrap 1000
```

Cần báo cáo 3 nhóm metric:

Prediction:

- Directional accuracy.
- Balanced accuracy.
- Macro-F1.
- MCC.
- Brier.
- ECE.

Signal quality:

- BUY precision.
- SELL precision.
- Coverage.
- Coverage-precision curve.

Trading:

- Net cumulative return.
- Annualized Sharpe.
- Sortino.
- Max drawdown.
- Turnover.
- Transaction-cost-adjusted return.

Ví dụ bảng kết quả happy path:

| Model | MCC | Macro-F1 | Brier | ECE | Coverage | Net Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Buy-and-hold | n/a | n/a | n/a | n/a | 100% | 0.42 |
| RSI/MACD rules | 0.03 | 0.34 | n/a | n/a | 38% | 0.31 |
| XGBoost technical | 0.08 | 0.39 | 0.64 | 0.11 | 44% | 0.55 |
| PatchTST price-only | 0.11 | 0.41 | 0.61 | 0.09 | 47% | 0.68 |
| FinBERT text-only | 0.07 | 0.37 | 0.66 | 0.13 | 35% | 0.44 |
| Current MarketLens | 0.05 | 0.35 | n/a | n/a | 36% | 0.38 |
| MarketLens-kNN | 0.09 | 0.40 | 0.63 | 0.10 | 41% | 0.59 |
| CEM-RAG | 0.17 | 0.47 | 0.56 | 0.06 | 43% | 0.92 |

Điều kiện gate:

```text
MCC improvement >= 0.05
Net Sharpe improvement >= 0.25
```

Nếu baseline mạnh nhất là PatchTST price-only:

```text
CEM-RAG MCC 0.17 - PatchTST MCC 0.11 = +0.06 -> pass
CEM-RAG Sharpe 0.92 - PatchTST Sharpe 0.68 = +0.24 -> fail sát ngưỡng
```

Trong ví dụ này paper chưa nên nộp, vì Sharpe thiếu 0.01. Cần cải thiện hoặc báo cáo trung thực.

Happy path tốt hơn:

```text
CEM-RAG Sharpe 0.98 - PatchTST Sharpe 0.68 = +0.30 -> pass
```

### Bước 12: Kiểm định thống kê

Classification:

- McNemar test cho paired predictions.
- Ví dụ so sánh CEM-RAG vs PatchTST trên từng ngày test.

Trading:

- Block bootstrap cho Sharpe/return vì time-series có autocorrelation.
- Block size có thể là 7 ngày hoặc 14 ngày.

Ví dụ báo cáo:

```text
Sharpe improvement over PatchTST:
  mean diff = +0.30
  95% block-bootstrap CI = [+0.08, +0.53]
```

Đây là happy path vì CI không cắt 0.

Bad case:

```text
95% CI = [-0.04, +0.58]
```

Kết quả này không đủ mạnh để nói model trading tốt hơn một cách chắc chắn.

### Bước 13: Ablation bắt buộc

Cần chạy các biến thể:

| Ablation | Câu hỏi cần trả lời |
|---|---|
| No news/events | Tin tức có thêm giá trị ngoài price không? |
| No price encoder | Event-only có đủ không? |
| No novelty/dissemination | Source breadth và novelty có cần không? |
| No learned retriever | Learned retrieval có hơn fixed kNN không? |
| Fixed kNN vs learned retriever | Đóng góp của retriever nằm ở đâu? |
| No cross-attention fusion | Fusion có cần, hay concat đủ? |
| LLM explanation only vs LLM final decision | Giảm final LLM decision có tốt hơn không? |
| BTC-only vs multi-asset | Multi-asset training có generalize tốt hơn không? |

Ví dụ bảng ablation happy path:

| Variant | MCC | Net Sharpe | Diễn giải |
|---|---:|---:|---|
| Full CEM-RAG | 0.17 | 0.98 | Tốt nhất |
| No events | 0.11 | 0.68 | Mất event signal |
| No learned retriever | 0.13 | 0.76 | kNN kém hơn |
| No novelty/source breadth | 0.14 | 0.81 | Tin lặp lại gây nhiễu |
| No price encoder | 0.09 | 0.52 | Text-only không đủ |
| No cross-attention | 0.14 | 0.79 | Fusion đơn giản kém hơn |
| LLM final decision | 0.06 | 0.41 | Khó calibration |

Thông điệp:

> Full model tốt không phải vì thêm nhiều feature tùy tiện, mà vì event memory và learned retrieval thật sự đóng góp.

### Bước 14: Case study cho paper

Nên chọn 4 case:

1. CEM-RAG đúng BUY, baseline bỏ lỡ.
2. CEM-RAG đúng SELL, baseline false BUY/HOLD.
3. CEM-RAG đúng HOLD vì conflict.
4. CEM-RAG sai, phân tích nguyên nhân.

Ví dụ case study đúng BUY:

```text
Date: 2024-03-04
Symbol: BTC
Top event: ETF inflow, source_count=12, novelty_30d=0.54
Retrieved cases: 2023-10-24, 2024-02-12, 2024-02-26
CEM-RAG: BUY, p_up=0.61
Actual 7d return: +6.4%
Baseline RSI/MACD: HOLD
```

Ví dụ case study sai:

```text
Date: 2024-01-11
Symbol: BTC
Top event: ETF approval, polarity positive
CEM-RAG: BUY
Actual 7d return: -7.2%
Reason: sell-the-news effect, RSI overbought, event polarity positive but market already priced in.
Fix/Ablation insight: novelty alone insufficient; need expectation/surprise relative to pre-event run-up.
```

Một paper có cả phần sai sẽ đáng tin hơn.

---

## 8. Baseline cần có

Baselines tối thiểu:

1. Buy-and-hold.
2. Momentum rule.
3. RSI/MACD rule.
4. Logistic Regression technical.
5. Random Forest technical.
6. XGBoost technical.
7. LSTM/GRU price-only.
8. TFT/PatchTST price-only.
9. FinBERT/CryptoBERT text-only.
10. Current MarketLens LLM + StockMem + guardrails.
11. MarketLens-kNN fixed weighted cosine.
12. kNN-Returns deterministic strategy.
13. Random retrieval.
14. BM25/text embedding retrieval.
15. Numeric Euclidean retrieval.
16. Learned CEM retriever.

Lý do phải nhiều baseline:

- đồ án tốt nghiệp reviewer sẽ không chấp nhận so sánh với baseline yếu.
- Nếu CEM-RAG chỉ vượt current MarketLens nhưng thua XGBoost/PatchTST thì contribution không đủ.

---

## 9. Data leakage checklist

Những lỗi có thể làm paper bị reject:

- Dùng news sau ngày T để predict ngày T.
- Normalize feature bằng mean/std của cả train+test.
- Retrieval tìm cả record sau ngày T.
- Tune `tau` trên test.
- Chọn best checkpoint theo test.
- Dùng future return trong feature.
- Duplicate article giữa train/test theo URL hoặc syndicated content.
- Label crypto ngày T+7 nhưng dùng candle close sau cutoff.

Checklist cần pass:

```text
[ ] Mỗi prediction row có cutoff_time
[ ] Mỗi article có published_at và crawl_time
[ ] Feature normalization fit trên train only
[ ] Validation chỉ dùng để tune tau/calibration/hyperparameters
[ ] Test locked
[ ] Retrieval API bắt buộc có before_date
[ ] Duplicate URL/content được dedup trước split
[ ] Transaction cost được trừ vào return
```

---

## 10. Artifact nên tạo để reviewer tin

Nên có các file/artifact:

```text
artifacts/datasets/daily_records_train.parquet
artifacts/datasets/daily_records_val.parquet
artifacts/datasets/daily_records_test.parquet
artifacts/predictions/cem_rag_test.jsonl
artifacts/predictions/baselines/*.jsonl
artifacts/metrics/main_table.csv
artifacts/metrics/ablation_table.csv
artifacts/figures/reliability_diagram.png
artifacts/figures/coverage_precision_curve.png
artifacts/figures/equity_curve.png
artifacts/figures/retrieval_case_study.png
```

Ví dụ 1 row prediction dump:

```json
{
  "date": "2025-03-18",
  "symbol": "BTC",
  "horizon": "7d",
  "p_up": 0.61,
  "p_down": 0.21,
  "p_hold": 0.18,
  "signal": "BUY",
  "actual_return_7d": 4.6,
  "top_events": [
    {
      "event_group": "institutional_adoption",
      "event_type": "etf_inflow",
      "source_count": 11,
      "novelty_score": 0.74
    }
  ],
  "retrieval_trace": [
    {
      "date": "2024-03-04",
      "similarity": 0.91,
      "future_return": 6.4,
      "retrieval_model": "CEM-RAG-learned-retriever"
    }
  ]
}
```

---

## 11. Cách viết abstract theo happy path

Draft ý tưởng:

```text
Cryptocurrency prices are highly event-sensitive, yet existing news-based forecasting systems often rely on isolated sentiment scores or unverified LLM reasoning. We propose CEM-RAG, a point-in-time Crypto Event Memory Retrieval-Augmented Forecasting framework that transforms daily news into structured event memory, learns an event-aware historical retriever with supervised contrastive signals, and fuses price, event, and retrieved-case evidence for calibrated multi-horizon movement prediction. Experiments on ten major cryptocurrencies from 2018 to 2026 show that CEM-RAG outperforms technical, text-only, price-only, fixed kNN, and current MarketLens baselines on directional, calibration, and transaction-cost-adjusted trading metrics. Ablation studies demonstrate that both event novelty/dissemination features and learned retrieval contribute significantly. Case studies further show that CEM-RAG provides auditable event and retrieval traces for financial decision support.
```

Thông điệp chính:

- Crypto là event-sensitive.
- Sentiment đơn lẻ/LLM final decision chưa đủ.
- Event memory + learned retrieval + calibrated policy là đóng góp.
- Có implementation và test theo đúng scope đồ án tốt nghiệp.

---

## 12. Cấu trúc paper đề xuất

1. Introduction
   - Crypto market event sensitivity.
   - Limit của sentiment-only, price-only, LLM-only.
   - Giới thiệu CEM-RAG.

2. Related Work
   - News-based financial prediction.
   - Financial sentiment models: FinBERT/CryptoBERT.
   - Time-series Transformers/PatchTST.
   - Financial RAG/retrieval.
   - Event memory/StockMem-style approaches.

3. System Overview
   - Microservice architecture.
   - Point-in-time pipeline.
   - Data contracts.

4. Methodology
   - Event extraction.
   - Daily event memory.
   - Learned retriever.
   - Forecasting model.
   - Calibrated trading policy.

5. Experimental Design
   - Dataset.
   - Split.
   - Baselines.
   - Metrics.
   - Significance tests.

6. Results
   - Main table.
   - Trading table.
   - Calibration.
   - Walk-forward.

7. Ablation and Analysis
   - No events.
   - No learned retriever.
   - No cross-attention.
   - BTC-only vs multi-asset.

8. Case Studies
   - BUY, SELL, HOLD, failure case.

9. Limitations
   - Market non-stationarity.
   - News source bias.
   - Causal interpretation risk.
   - LLM extraction errors.
   - Crypto-specific liquidity/regime changes.

10. Conclusion

---

## 13. Những điểm domain expert nên review

Hãy review các điểm sau bằng kinh nghiệm finance/crypto:

1. Event taxonomy:
   - 13 group/62 type đã đủ crypto chưa?
   - Có thiếu event như ETF flow, stablecoin depeg, bridge exploit, exchange proof-of-reserves, liquidation cascade?

2. Polarity:
   - Một event có polarity khác nhau theo asset không?
   - Ví dụ "Binance enforcement" bearish cho BNB, nhưng có thể neutral/positive cho BTC nếu flight-to-quality.

3. Label:
   - Hold band 0.5% có hợp lý với crypto 7d không?
   - Có nên dùng volatility-adjusted label thay vì fixed threshold?

4. Trading policy:
   - Coverage 25-60% có hợp lý không?
   - Transaction cost 10 bps có hợp lý với spot crypto không?

5. Horizon:
   - 7d có phù hợp với news event không?
   - ETF/macro có thể 7-30d, exploit/liquidation có thể 1-3d.

6. Retrieval:
   - Similarity nên cùng symbol hay cho cross-asset?
   - Event "SOL outage" có nên retrieve từ "ETH congestion" không?

7. Evaluation:
   - Có cần tách bull/bear/sideways regime?
   - Có cần evaluate riêng high-volatility days?

8. Case study:
   - Retrieved cases có thật sự tương tự về finance narrative không, hay chỉ giống keyword?

---

## 14. Ví dụ các bảng/figure nên có trong paper

### Bảng dataset

| Symbol | Start | End | Daily rows | News articles | Event clusters |
|---|---|---|---:|---:|---:|
| BTC | 2018-01-01 | 2026-05-17 | 3059 | 120000 | 18000 |
| ETH | 2018-01-01 | 2026-05-17 | 3059 | 85000 | 14500 |
| SOL | 2020-01-01 | 2026-05-17 | 2329 | 40000 | 7200 |

### Bảng main results

| Model | Balanced Acc | Macro-F1 | MCC | Brier | ECE | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.39 | 0.37 | 0.08 | 0.64 | 0.11 | 0.55 |
| PatchTST | 0.42 | 0.41 | 0.11 | 0.61 | 0.09 | 0.68 |
| MarketLens-kNN | 0.40 | 0.40 | 0.09 | 0.63 | 0.10 | 0.59 |
| CEM-RAG | 0.49 | 0.47 | 0.17 | 0.56 | 0.06 | 0.98 |

### Figure nên có

- Architecture diagram.
- Event memory construction diagram.
- Learned retrieval triplet diagram.
- Reliability diagram.
- Coverage-precision curve.
- Equity curve after transaction cost.
- Case study retrieval trace.

---

## 15. Nguồn tham khảo nên đưa vào paper

Các nguồn nên đọc và cite:

- đồ án tốt nghiệp scope, Elsevier/ScienceDirect: https://www.sciencedirect.com/journal/expert-systems-with-applications
- News-based intelligent prediction SLR, đồ án tốt nghiệp: https://www.sciencedirect.com/science/article/pii/S0957417423000106
- StockMem event-reflection memory: https://arxiv.org/abs/2512.02720
- Financial time-series RAG/FinSeer: https://arxiv.org/abs/2502.05878
- FinBERT: https://arxiv.org/abs/1908.10063
- PatchTST: https://arxiv.org/abs/2211.14730
- FinGPT dissemination-aware forecasting: https://openreview.net/pdf?id=l2nHuTk6nc
- CryptoLin corpus: https://link.springer.com/article/10.1007/s10579-024-09743-x
- DLT-Sentiment-News dataset: https://huggingface.co/datasets/ExponentialScience/DLT-Sentiment-News
- CrypTop12 dataset: https://colab.ws/articles/10.1109%2Ficmla52953.2021.00065

---

## 16. Kết luận ngắn gọn

MarketLens hiện tại có kiến trúc tốt nhưng research core cũ chưa đủ mạnh để nộp đồ án tốt nghiệp. Hướng CEM-RAG biến system thành một intelligent financial forecasting system có đóng góp rõ:

- Tin tức được cấu trúc hóa thành event memory.
- Historical retrieval được học từ outcome, không chỉ cosine heuristic.
- Forecasting kết hợp price, event, và retrieved cases.
- Trading decision dựa trên probability calibration và validation-tuned policy.
- Evaluation nghiêm ngặt theo temporal split, walk-forward, ablation và statistical tests.

Nếu happy path thành công, paper có thể được position như một hệ thống thông minh có implementation đầy đủ cho finance/text mining/information retrieval/trading, đúng scope của đồ án tốt nghiệp. Điều kiện quan trọng nhất vẫn là kết quả thực nghiệm: CEM-RAG phải vượt baseline mạnh nhất trên held-out test, không chỉ vượt version MarketLens cũ.
