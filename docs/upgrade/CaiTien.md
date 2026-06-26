# CaiTien.md - Roadmap nâng cấp MarketLens thành CEM-RAG hướng ESWA

> Tài liệu này mô tả cách nâng cấp project MarketLens 

---

## 1. Cơ sở đọc tài liệu và hiện trạng repo

Tài liệu này được tổng hợp từ:

- `MoTa.md`: định hướng CEM-RAG, event memory, learned retrieval, calibrated trading policy.
- `README.md`: kiến trúc microservice tổng quan.
- `architecture.md`: hợp đồng module, data models, inter-module communication, testing strategy.
- `docs/api-guide.md`: API đang triển khai trên VPS, workflow run/backfill/result.
- `docs/backtest_report_2026-05.md`: backtest BTC hiện tại giai đoạn 2022-01-01 đến 2026-05-17.
- Code hiện tại trong `stockmem`, `aihub`, `factor_ledge`, `main_controller`, `market_data`, `crawler`, `shared`.

Các nguồn học thuật và guideline dùng để định vị hướng nâng cấp:

- ESWA Aims & Scope: https://www.sciencedirect.com/journal/expert-systems-with-applications
- ESWA Guide for Authors: https://www.sciencedirect.com/journal/expert-systems-with-applications/publish/guide-for-authors
- FinBERT: https://arxiv.org/abs/1908.10063
- PatchTST: https://arxiv.org/abs/2211.14730
- StockMem event-reflection memory: https://arxiv.org/abs/2512.02720
- FinSeer financial time-series RAG: https://arxiv.org/abs/2502.05878
- Dissemination-aware FinGPT: https://openreview.net/forum?id=l2nHuTk6nc
- CryptoLin corpus: https://link.springer.com/article/10.1007/s10579-024-09743-x

ESWA phù hợp với hướng này vì journal tập trung vào expert/intelligent systems, finance, information retrieval, stock trading, neural networks, knowledge discovery, data mining và text mining. Vì vậy MarketLens không nên được trình bày như một chatbot hoặc pipeline LLM đơn thuần, mà nên được nâng cấp và viết thành một **intelligent financial forecasting system** có dữ liệu point-in-time, event memory, retrieval được học từ outcome, mô hình dự báo xác suất, kiểm định thống kê và phần mềm có thể tái lập.

---

## 2. Hiện trạng MarketLens

MarketLens hiện là pipeline crypto news intelligence dạng microservice:

| Service | Vai trò hiện tại |
|---|---|
| `Crawler` | Lấy tin tức từ Supabase/RSS, trả article có sentiment/factors/raw text. |
| `AIHub` | Sentiment, factor extraction, RAG prediction qua LLM. |
| `MarketData` | OHLCV, history, indicators từ Binance/TradingView. |
| `StockMem` | Lưu daily record và tìm similar cases bằng weighted similarity. |
| `FactorLedge` | Chuẩn hóa factor, classify factor thành vector taxonomy 75d. |
| `MainController` | Điều phối collect -> score -> StockMem -> predict -> result. |
| `LLMGateway` | Final BUY/HOLD/SELL decision và guardrail policy. |

Luồng hiện tại:

```text
Crawler + MarketData
        |
        v
MainController step_ai_score
        |
        v
FactorLedge factor vector + StockMem daily record
        |
        v
StockMem weighted kNN search
        |
        v
AIHub/LLMGateway RAG predict
        |
        v
PredictionResult { signal, confidence, explanation, similar_cases }
```

StockMem hiện dùng split-vector embedding:

- `factor_vec`: 75 chiều, gồm 62 event type bits + 13 group bits.
- `indicator_vec`: 5 chiều, gồm MSI, RSI, sentiment, Fear & Greed, price_change.
- `price_vec`: 60 chiều, gồm return/range/volume change của 20 candles.

Similarity hiện tại:

```text
score = w_factor * cosine(factor_vec)
      + w_indicator * cosine(indicator_vec)
      + w_price * cosine(price_vec)
```

Repo đã có các điểm tốt:

- `StockMem.search` đã có `before_date` để tránh retrieve record cùng ngày hoặc tương lai trong backtest.
- `stockmem/scripts/optimize_weights.py` đã có Bayesian/Optuna weight optimization.
- `main_controller/src/orchestrator/steps.py` đã có historical mode `as_of_date`.
- `scripts/backtest_runner.py` đã có backtest theo ngày và forward returns 1d/3d/7d/15d/30d.
- `docs/backtest_report_2026-05.md` đã có báo cáo thực nghiệm BTC tương đối rõ.

Tuy nhiên, repo hiện **chưa có** implementation CEM-RAG đầy đủ. Cụ thể:

- Chưa có `aihub/src/predict/cem_rag.py`.
- Chưa có schema `EventRecord`, `DailyEventState`, `RetrievalTrace`.
- Chưa có event memory point-in-time theo ngày/symbol.
- Chưa có learned retriever được train bằng supervised contrastive/triplet learning.
- Chưa có `p_up`, `p_down`, `p_hold` được calibration.
- Chưa có evaluation suite đầy đủ cho MCC, Macro-F1, Brier, ECE, Sharpe, Sortino, max drawdown, McNemar, block bootstrap.

---

## 3. Kết quả backtest hiện tại và ý nghĩa

Theo `docs/backtest_report_2026-05.md`:

- Scope: 1,568 backtests, BTC, 2022-01-01 -> 2026-05-17.
- Pipeline: MainController -> LLMGateway -> StockMem kNN -> Regime Guardrails.
- Signal distribution:
  - BUY: 399 ngày, 25.4%.
  - SELL: 170 ngày, 10.8%.
  - HOLD: 999 ngày, 63.7%.
  - Coverage BUY+SELL: 36.3%.

Kết quả D+7:

| Signal | Accuracy D+7 | Nhận xét |
|---|---:|---|
| BUY | 51.9% | Chỉ nhỉnh hơn random, tốt hơn ở bull market 2023-2024. |
| SELL | 39.4% | Yếu nhất; false SELL nhiều trong bull run. |
| HOLD | 7.1% | HOLD chỉ đúng khi abs(return) nhỏ, nên metric thấp. |
| Directional precision BUY+SELL | 48.2% | Chưa đủ mạnh cho paper. |

Điểm yếu chính:

- Final decision phụ thuộc nhiều vào LLM + guardrails hardcoded, khó chứng minh tối ưu.
- Confidence chưa calibration tốt; SELL đúng và SELL sai có confidence gần nhau.
- HOLD quá nhiều, coverage thấp.
- SELL overall dưới 50%, không đủ thuyết phục.
- Backtest hiện tại chưa đủ baseline học thuật như XGBoost, PatchTST, FinBERT text-only, fixed kNN, learned retriever.

Điểm sáng quan trọng từ `MoTa.md`:

- Kết quả kNN-Returns nội bộ cho thấy historical memory có tín hiệu dự báo tốt hơn LLM decision hiện tại.
- Cấu hình New Bayesian threshold +/-2% trong `MoTa.md` đạt BUY DA khoảng 59.7%, SELL DA khoảng 57.5%, coverage khoảng 58.2%.
- Các số liệu này nên được xem là **milestone nội bộ cần tái lập bằng artifact chính thức**, chưa nên tuyên bố là final paper result nếu chưa có split train/validation/test và leakage audit.

Kết luận hiện trạng:

> MarketLens có kiến trúc tốt để phát triển thành một expert/intelligent system, nhưng research core hiện tại chưa đủ mạnh để nộp ESWA. Hướng nâng cấp nên chuyển từ "LLM + fixed kNN + guardrails" sang "point-in-time event memory + learned retrieval + calibrated probabilistic forecasting + rigorous evaluation".

---

## 4. Định vị paper đề xuất

Tên ngắn:

```text
CEM-RAG: Crypto Event Memory Retrieval-Augmented Forecasting
```

Tên đầy đủ:

```text
CEM-RAG: A Point-in-Time Crypto Event Memory and Learned Retrieval Framework
for Multi-Horizon Cryptocurrency Movement Forecasting
```

Nhiệm vụ nghiên cứu:

- Primary task: dự báo hướng biến động crypto 7 ngày.
- Secondary horizons: 1d, 3d, 15d, 30d.
- Output model: xác suất `UP`, `DOWN`, `HOLD/NO_TRADE`.
- Output trading: BUY/SELL/HOLD bằng policy tune trên validation.
- Output explainability: top events, retrieved historical cases, evidence trace.

Research questions:

| RQ | Câu hỏi |
|---|---|
| RQ1 | Event memory có cải thiện crypto movement forecasting so với price-only và sentiment-only không? |
| RQ2 | Learned retriever có lấy historical cases tốt hơn fixed weighted kNN không? |
| RQ3 | Fusion price-event-retrieval có tốt hơn từng modality riêng lẻ không? |
| RQ4 | Calibration + validation-tuned trading policy có cải thiện Sharpe sau phí không? |

Hypotheses:

- H1: CEM-RAG có MCC cao hơn baseline mạnh nhất ít nhất 0.05 trên held-out test.
- H2: CEM-RAG có net Sharpe cao hơn baseline mạnh nhất ít nhất 0.25 sau transaction cost.
- H3: Bỏ event memory làm giảm MCC/Macro-F1 rõ rệt.
- H4: Learned retriever vượt random, BM25/text embedding, numeric Euclidean và fixed weighted kNN.

---

## 5. Roadmap nâng cấp chi tiết

### Bước 1: Chuẩn hóa protocol dữ liệu point-in-time

Mục tiêu:

- Mọi prediction row phải biết chính xác hệ thống được phép nhìn thấy gì tại thời điểm dự báo.
- Loại bỏ data leakage trước khi train model.

Thay đổi cần làm:

1. Thêm `cutoff_time` vào mọi artifact prediction/backtest.
2. Chuẩn hóa article fields:
   - `published_at`
   - `crawled_at`
   - `source`
   - `url`
   - `canonical_url`
   - `content_hash`
   - `symbol_mentions`
3. Với ngày T, chỉ dùng article thỏa:

```text
published_at <= cutoff_time
crawled_at <= cutoff_time
```

4. Với market candle, chỉ dùng candle đã đóng trước hoặc tại cutoff hợp lệ.
5. Với retrieval, bắt buộc:

```text
candidate.date < query.date
candidate.date + horizon <= query.date
```

Điểm cần sửa trong code sau này:

- `shared.models.memory.StockMemRecord`: thêm optional `cutoff_time`.
- `scripts/backtest_runner.py`: ghi `cutoff_time` vào output/backtest_results.
- `StockMem.search`: giữ `before_date`; thêm kiểm tra maturity theo horizon trong evaluation layer.
- Dataset builder mới không được fit scaler trên toàn bộ dữ liệu; scaler chỉ fit trên train.

Leakage checklist bắt buộc:

```text
[ ] Mỗi prediction row có cutoff_time.
[ ] Mỗi article có published_at và crawled_at.
[ ] Article sau cutoff không xuất hiện trong features.
[ ] Feature normalization fit trên train only.
[ ] Validation chỉ dùng tune hyperparameter, tau, calibration.
[ ] Test locked và chỉ evaluate một lần cho result chính.
[ ] Retrieval không lấy record cùng ngày hoặc tương lai.
[ ] Retrieval không dùng future_return chưa matured tại prediction date.
[ ] Duplicate URL/content được dedup trước split.
[ ] Transaction cost được trừ trong trading metrics.
```

Temporal split đề xuất:

```text
Train:      2018-01-01 -> 2023-12-31
Validation: 2024-01-01 -> 2024-12-31
Test:       2025-01-01 -> 2026-05-17
```

Nếu dữ liệu trước 2022 chưa đủ:

```text
Train:      2022-01-01 -> 2023-12-31
Validation: 2024-01-01 -> 2024-12-31
Test:       2025-01-01 -> 2026-05-17
```

Không shuffle dữ liệu time-series.

---

### Bước 2: Xây event extraction từ article/factor

Mục tiêu:

- Biến news/factor rời rạc thành event có cấu trúc, có thể kiểm định.
- Giảm phụ thuộc vào LLM final reasoning.

Schema đề xuất:

```python
class EventRecord(BaseModel):
    event_group: str
    event_type: str
    entities: list[str]
    polarity: float
    confidence: float
    source_article_id: str
    observed_at: datetime
    evidence_text: str | None = None
```

Nguồn input:

- `IngestionRecord.article_name`
- `IngestionRecord.summary`
- `IngestionRecord.raw_text`
- `IngestionRecord.factors`
- `FactorLedge` taxonomy 13 group/62 type hiện có.

Event taxonomy cần mở rộng cho crypto:

| Group | Event type ví dụ |
|---|---|
| Regulation & Legal | ETF approval, ETF delay, enforcement action, sanctions, staking regulation |
| Market Performance | ETF flow, volume surge, BTC dominance shift, liquidation cascade |
| Whale & On-chain | exchange inflow, exchange outflow, whale accumulation, miner selling |
| Risk & Warning | bridge exploit, stablecoin depeg, exchange insolvency, proof-of-reserves concern |
| Protocol & Product | protocol upgrade, mainnet launch, fee/gas change, supply dynamics |
| Macroeconomic | CPI surprise, Fed rate decision, DXY movement, Treasury yield shock |

Triển khai đề xuất:

- Tạo module mới trong `aihub/src/events/`:
  - `schema.py`: `EventRecord`, `EventExtractionRequest`, `EventExtractionResponse`.
  - `extractor.py`: map article/factor -> event.
  - `taxonomy.py`: taxonomy crypto event.
  - `prompts.py`: prompt event extraction nếu dùng LLM.
- API mới trong `aihub/src/api.py`:

```text
POST /events/extract
```

Payload:

```json
{
  "symbol": "BTC",
  "article": {
    "id": "66214",
    "title": "SEC delays Ethereum ETF decision",
    "published_at": "2026-05-04T06:17:30Z",
    "source": "cointelegraph.com",
    "summary": "...",
    "raw_text": "..."
  }
}
```

Response:

```json
{
  "events": [
    {
      "event_group": "regulation_legal",
      "event_type": "etf_delay",
      "entities": ["SEC", "Ethereum ETF", "ETH"],
      "polarity": -0.72,
      "confidence": 0.84,
      "source_article_id": "66214",
      "observed_at": "2026-05-04T06:17:30Z",
      "evidence_text": "SEC delays Ethereum ETF decision"
    }
  ]
}
```

Testing:

- Unit test deterministic mapping cho known headlines.
- Test polarity theo asset:
  - `SEC sues Binance` bearish cho BNB, có thể neutral hoặc mixed cho BTC.
  - `Stablecoin exchange inflow rising` có thể bullish nếu liquidity inflow, bearish nếu panic context.
- Test confidence nằm trong `[0, 1]`, polarity trong `[-1, 1]`.

---

### Bước 3: Cluster same-day news thành DailyEventState

Mục tiêu:

- Một ngày có 30 bài cùng nói ETF inflow không nên là 30 tín hiệu độc lập.
- Cần đo breadth, source diversity và novelty để phân biệt event mới với tin lặp lại.

Schema đề xuất:

```python
class DailyEventState(BaseModel):
    date: date
    symbol: str
    events: list[EventRecord]
    article_count: int
    source_count: int
    source_diversity: float
    novelty_7d: float
    novelty_30d: float
    dominant_event_groups: list[str] = []
```

Aggregation logic:

1. Lấy tất cả article của symbol trong ngày T trước cutoff.
2. Extract event cho từng article.
3. Group event theo:

```text
(event_group, event_type, normalized_entities)
```

4. Với mỗi cluster, tính:
   - `article_count`
   - `source_count`
   - `source_diversity`
   - weighted polarity trung bình
   - confidence trung bình hoặc max-confidence weighted average
5. Tính novelty:

```text
novelty_7d  = 1 - max_similarity(current_event_cluster, clusters in T-7..T-1)
novelty_30d = 1 - max_similarity(current_event_cluster, clusters in T-30..T-1)
```

6. Persist `DailyEventState` để dùng train/evaluate.

Feature vector event đề xuất:

```text
event_type multi-hot
event_group multi-hot
mean_polarity
max_abs_polarity
article_count_log
source_count_log
source_diversity
novelty_7d
novelty_30d
confidence_mean
top_event_breadth
```

Ví dụ:

```json
{
  "date": "2025-03-18",
  "symbol": "BTC",
  "article_count": 28,
  "source_count": 11,
  "source_diversity": 0.86,
  "novelty_7d": 0.74,
  "novelty_30d": 0.54,
  "events": [
    {
      "event_group": "market_performance",
      "event_type": "etf_flow",
      "entities": ["BlackRock", "Bitcoin ETF", "BTC"],
      "polarity": 0.82,
      "confidence": 0.88,
      "source_article_id": "cluster:2025-03-18:etf_flow",
      "observed_at": "2025-03-18T23:59:59Z"
    }
  ]
}
```

Testing:

- 20 bài cùng event -> 1 cluster chính, không 20 event độc lập.
- 20 bài từ 1 source có source diversity thấp hơn 20 bài từ nhiều source.
- Event lặp lại 30 ngày có novelty thấp.
- Event shock mới có novelty cao.

---

### Bước 4: Mở rộng StockMem từ factor memory sang event memory

Mục tiêu:

- StockMem hiện đang lưu factor/indicator/price. Cần thêm event-memory features để phục vụ CEM-RAG.

Thay đổi schema:

```python
class StockMemRecord(BaseModel):
    ...
    event_state: DailyEventState | None = None
    event_vector: list[float] = []
    cutoff_time: datetime | None = None
```

Thay đổi embedder:

Hiện tại:

```text
factor_vec 75d
indicator_vec 5d
price_vec 60d
```

Đề xuất:

```text
event_vec       d_event
factor_vec      75d
indicator_vec   5d
price_vec       60d
```

Search score v1:

```text
score = w_event * sim(event_vec)
      + w_factor * sim(factor_vec)
      + w_indicator * sim(indicator_vec)
      + w_price * sim(price_vec)
```

Yêu cầu giữ compatibility:

- Nếu `event_vector` rỗng thì fallback về scoring cũ.
- Không phá endpoint `/record` và `/search` hiện tại.
- Cần version hóa retriever để backtest biết đang dùng method nào.

API search đề xuất:

```python
class SearchRequest(BaseModel):
    query: StockMemRecord
    k: int = 5
    before_date: date | None = None
    horizon: str = "7d"
    retriever_type: str = "fixed_knn"  # fixed_knn | learned_cem
```

Response mở rộng:

```python
class RetrievalTrace(BaseModel):
    retrieved_date: date
    similarity: float
    event_match: dict[str, float]
    future_return_7d: float | None
    retriever_version: str
```

Testing:

- Search cùng symbol vẫn được giữ.
- `before_date` loại record cùng ngày/tương lai.
- `retriever_type=fixed_knn` trả kết quả như cũ khi chưa bật learned retriever.
- Event vector rỗng không làm crash search.
- Search response có retriever version để audit.

---

### Bước 5: Huấn luyện learned event-memory retriever

Mục tiêu:

- Thay weighted cosine thủ công bằng retriever học từ outcome.
- Retriever phải phân biệt được trường hợp keyword giống nhau nhưng outcome trái chiều.

Training data:

Mỗi row:

```json
{
  "date": "2024-03-04",
  "symbol": "BTC",
  "event_vector": [0.1, 0.0, 1.0],
  "factor_vec": [0, 1, 0],
  "indicator_vec": [0.55, 0.22],
  "price_vec": [0.01, -0.02],
  "future_return_7d": 6.4,
  "regime": "bull",
  "cutoff_time": "2024-03-04T23:59:59Z"
}
```

Positive pair:

- Cùng hướng future return 7d.
- Regime volatility gần nhau.
- Event group/type tương đồng hoặc financial narrative tương đồng.

Hard negative:

- Event text/taxonomy rất giống query.
- Nhưng future return 7d trái dấu.

Ví dụ:

```text
Query:
ETF optimism + RSI 58 + volume up -> future_return_7d +6%

Positive:
ETF inflow + volume breakout -> future_return_7d +8%

Hard negative:
ETF approval + RSI 85 + sell-the-news -> future_return_7d -7%
```

Loss đề xuất:

```text
Triplet loss:
max(0, margin + dist(query, positive) - dist(query, negative))

hoặc supervised contrastive loss:
pull same-direction/same-regime cases together,
push opposite-outcome hard negatives apart.
```

Files/scripts đề xuất:

- `stockmem/src/search/learned_retriever.py`
- `stockmem/scripts/build_cem_dataset.py`
- `stockmem/scripts/train_learned_retriever.py`
- `stockmem/scripts/evaluate_retriever.py`
- output: `stockmem/config/learned_retriever.json` hoặc model checkpoint trong `artifacts/models/`.

Metrics retriever:

| Metric | Ý nghĩa |
|---|---|
| Hit same direction@5 | Top-5 có cùng direction với query bao nhiêu lần. |
| Avg retrieved sign accuracy | Mean sign của retrieved future returns đúng hướng không. |
| Hard-negative separation | Query gần positive hơn hard negative bao nhiêu. |
| Retrieval trace quality | Case lấy ra có narrative tài chính hợp lý không. |

Acceptance gate retriever:

- Learned retriever phải vượt fixed weighted kNN trên validation và test.
- Nếu chỉ tốt trên train, không dùng làm contribution chính.
- Nếu learned retriever không vượt kNN-Returns, paper nên trung thực gọi kNN-Returns là baseline mạnh nhất.

---

### Bước 6: Xây forecasting model price-event-retrieval fusion

Mục tiêu:

- Model dự báo xác suất thay vì để LLM quyết định BUY/SELL/HOLD.
- LLM chỉ nên hỗ trợ extraction/explanation, không là final decision không kiểm chứng.

Input:

1. Price encoder:
   - 30 ngày OHLCV.
   - Return, range, volume change.
   - RSI, MACD, Bollinger, Fear & Greed, MSI.

2. Event encoder:
   - Event taxonomy vector.
   - Event text embedding từ FinBERT/CryptoBERT/LedgerBERT nếu có.
   - `polarity`, `confidence`, `article_count`, `source_count`, `novelty_7d`, `novelty_30d`.

3. Retrieval encoder:
   - Top-k historical cases.
   - Similarity.
   - Event match score.
   - Historical future returns chỉ dùng cho retrieved past cases đã matured.

Fusion:

```text
price tokens       -> temporal encoder, ví dụ PatchTST-style
event tokens       -> event encoder
retrieval tokens   -> case encoder
        |
        v
cross-attention / gated fusion
        |
        v
p_up, p_down, p_hold
```

Output schema đề xuất:

```python
class CEMRAGPredictResponse(BaseModel):
    horizon: str
    p_up: float
    p_down: float
    p_hold: float
    signal: SignalType
    policy_tau: float
    confidence: float
    explanation: str
    top_events: list[EventRecord]
    retrieval_trace: list[RetrievalTrace]
```

Policy:

```text
BUY  nếu p_up - p_down >= tau
SELL nếu p_down - p_up >= tau
HOLD ngược lại
```

Quy tắc:

- `tau` chỉ tune trên validation.
- Không tune `tau` trên test.
- Horizon primary: 7d.
- 1d/3d/15d/30d là auxiliary hoặc secondary analysis.

Triển khai theo mức độ:

| Mức | Mô tả | Khi nào dùng |
|---|---|---|
| v0 | kNN-Returns deterministic + calibrated policy | Baseline mạnh, dễ tái lập. |
| v1 | Gradient boosting/logistic trên event+price+retrieval features | Nhanh, ít rủi ro. |
| v2 | Neural fusion với temporal encoder + event encoder + retrieval encoder | Contribution chính nếu đủ data. |
| v3 | Cross-attention multimodal model | Chỉ làm nếu v2 đủ ổn và có thời gian. |

Khuyến nghị pragmatic:

1. Làm v0/v1 trước để có baseline mạnh và artifact.
2. Chỉ làm v2/v3 khi data đủ và v1 chưa đạt gate.
3. Không nên nhảy thẳng vào neural model nếu dataset nhỏ hoặc news thiếu năm 2018-2021.

---

### Bước 7: Calibration và trading policy

Mục tiêu:

- Confidence phải có ý nghĩa xác suất.
- Trading signal phải được tune trên validation, không hardcode tùy cảm tính.

Calibration metrics:

- Brier score.
- Expected Calibration Error (ECE).
- Reliability diagram.

Calibration methods:

- Temperature scaling cho neural logits.
- Isotonic regression nếu validation đủ lớn.
- Platt scaling cho binary one-vs-rest nếu cần.

Policy tuning:

Grid search trên validation:

```text
tau = 0.05, 0.06, ..., 0.50
```

Constraint:

```text
directional coverage between 25% and 60%
```

Objective:

```text
maximize validation net Sharpe
subject to:
  coverage in [25%, 60%]
  SELL precision not below baseline
  max drawdown not worse than buy-and-hold by unacceptable margin
```

Trading return:

```text
BUY return  = future_return_h - transaction_cost
SELL return = -future_return_h - transaction_cost
HOLD return = 0
```

Default transaction cost:

```text
10 bps per round trip
```

Cần báo cáo sensitivity:

- 5 bps.
- 10 bps.
- 20 bps.

Không nên chỉ báo cáo directional accuracy, vì DA tốt chưa chắc có Sharpe tốt.

---

### Bước 8: Evaluation ESWA-grade

Mục tiêu:

- Chứng minh contribution bằng thực nghiệm nghiêm ngặt, không chỉ demo.

Baselines bắt buộc:

| Nhóm | Baseline |
|---|---|
| Naive | Buy-and-hold, always HOLD, random direction theo class prior |
| Technical rule | Momentum, RSI/MACD rule |
| ML technical | Logistic Regression, Random Forest, XGBoost/LightGBM |
| Deep price-only | LSTM/GRU, PatchTST-style price-only |
| Text-only | FinBERT/CryptoBERT sentiment-only, event-only |
| Retrieval | random retrieval, BM25/text embedding, numeric Euclidean, fixed MarketLens-kNN, kNN-Returns |
| Existing system | Current MarketLens LLM + StockMem + guardrails |
| Proposed | CEM-RAG full |

Prediction metrics:

- Directional accuracy.
- Balanced accuracy.
- Macro-F1.
- MCC.
- AUROC one-vs-rest nếu phù hợp.
- Brier score.
- ECE.

Trading metrics:

- Net cumulative return.
- Annualized Sharpe.
- Sortino.
- Max drawdown.
- Turnover.
- Coverage.
- BUY precision.
- SELL precision.
- Coverage-precision curve.

Statistical tests:

- McNemar test cho paired classification predictions.
- Block bootstrap cho Sharpe/return vì time-series có autocorrelation.
- Block size mặc định: 7 hoặc 14 ngày.

Ablation bắt buộc:

| Variant | Câu hỏi |
|---|---|
| Full CEM-RAG | Model chính. |
| No events | News/event có thêm giá trị ngoài price không? |
| No price encoder | Event-only có đủ không? |
| No novelty/source breadth | Dissemination và novelty có cần không? |
| No learned retriever | Learned retrieval có hơn fixed kNN không? |
| Fixed kNN vs learned retriever | Contribution của retriever nằm ở đâu? |
| No retrieval | Memory có đóng góp không? |
| No calibration | Calibration có cải thiện decision quality không? |
| LLM final decision | LLM final decision có kém hơn probabilistic model không? |

Acceptance gates nội bộ:

```text
[ ] CEM-RAG vượt baseline mạnh nhất trên held-out test.
[ ] MCC improvement >= 0.05 so với baseline mạnh nhất.
[ ] Net Sharpe improvement >= 0.25 sau transaction cost.
[ ] Brier/ECE tốt hơn hoặc không tệ hơn baseline calibration.
[ ] Ablation bỏ event memory làm metric giảm rõ.
[ ] Ablation bỏ learned retriever làm metric giảm rõ.
[ ] McNemar hoặc bootstrap 95% CI ủng hộ cải tiến.
[ ] Case studies có cả đúng BUY, đúng SELL, đúng HOLD và failure analysis.
```

Nếu không pass gate:

- Không viết "CEM-RAG outperforms all baselines".
- Viết trung thực theo kết quả:
  - "event memory improves interpretability but not trading Sharpe"
  - hoặc "fixed kNN-Returns remains stronger than learned retriever"
  - hoặc "benefit appears only in high-news/high-volatility regimes"

---

## 6. Artifact cần tạo để reviewer tin

Dataset:

```text
artifacts/datasets/daily_records_train.parquet
artifacts/datasets/daily_records_val.parquet
artifacts/datasets/daily_records_test.parquet
artifacts/datasets/event_states_train.parquet
artifacts/datasets/event_states_val.parquet
artifacts/datasets/event_states_test.parquet
```

Predictions:

```text
artifacts/predictions/current_marketlens_test.jsonl
artifacts/predictions/fixed_knn_test.jsonl
artifacts/predictions/knn_returns_test.jsonl
artifacts/predictions/learned_retriever_test.jsonl
artifacts/predictions/cem_rag_test.jsonl
artifacts/predictions/baselines/*.jsonl
```

Metrics:

```text
artifacts/metrics/main_table.csv
artifacts/metrics/trading_table.csv
artifacts/metrics/ablation_table.csv
artifacts/metrics/calibration_table.csv
artifacts/metrics/stat_tests.json
```

Figures:

```text
artifacts/figures/architecture.png
artifacts/figures/event_memory_construction.png
artifacts/figures/retriever_triplet.png
artifacts/figures/reliability_diagram.png
artifacts/figures/coverage_precision_curve.png
artifacts/figures/equity_curve_after_cost.png
artifacts/figures/retrieval_case_study.png
```

Một row prediction chuẩn:

```json
{
  "date": "2025-03-18",
  "cutoff_time": "2025-03-18T23:59:59Z",
  "symbol": "BTC",
  "horizon": "7d",
  "p_up": 0.61,
  "p_down": 0.21,
  "p_hold": 0.18,
  "signal": "BUY",
  "policy_tau": 0.14,
  "actual_return_7d": 4.6,
  "top_events": [
    {
      "event_group": "market_performance",
      "event_type": "etf_flow",
      "source_count": 11,
      "novelty_30d": 0.54
    }
  ],
  "retrieval_trace": [
    {
      "retrieved_date": "2024-03-04",
      "similarity": 0.91,
      "future_return_7d": 6.4,
      "retriever_version": "learned_cem_v1"
    }
  ]
}
```

---

## 7. Mapping với kiến trúc hiện tại

### `Crawler`

Hiện tại:

- Lấy article latest/historical.
- Article đã có sentiment/factors từ Supabase hoặc enrichment.

Nâng cấp:

- Đảm bảo article có `published_at`, `crawled_at`, `source`, `canonical_url`, `content_hash`.
- Thêm dedup content-level, không chỉ URL.
- Xuất article theo cutoff trong historical mode.

### `AIHub`

Hiện tại:

- `/sentiment`
- `/factors`
- `/predict`

Nâng cấp:

- Thêm `/events/extract`.
- Thêm `aihub/src/events/`.
- Thêm CEM-RAG predictor ở `aihub/src/predict/cem_rag.py`.
- `/predict` nên hỗ trợ provider:

```text
provider=llm_rag       # hiện tại
provider=cem_rag_v0    # deterministic kNN-Returns
provider=cem_rag_v1    # calibrated ML
provider=cem_rag_v2    # neural fusion
```

### `FactorLedge`

Hiện tại:

- Taxonomy 13 group/62 type.
- Classify factor vector 75d.

Nâng cấp:

- Dùng taxonomy này làm seed cho event taxonomy.
- Thêm endpoint hoặc service để build event vector.
- Lưu event group/type mapping có version.

### `StockMem`

Hiện tại:

- `factor_vec`, `indicator_vec`, `price_vec`.
- Weighted cosine.
- Bayesian weight optimization.
- `before_date` hỗ trợ chống look-ahead.

Nâng cấp:

- Thêm `event_state`, `event_vector`, `cutoff_time`.
- Thêm `retriever_type`, `horizon`.
- Thêm learned retriever.
- Search response thêm `RetrievalTrace`.

### `MainController`

Hiện tại:

- `step_collect`
- `step_ai_score`
- `step_stockmem`
- `step_predict`

Nâng cấp:

```text
step_collect
step_extract_events
step_build_daily_event_state
step_ai_score
step_stockmem
step_retrieve
step_cem_predict
step_calibrated_policy
```

`step_predict` cần cho phép chọn provider bằng config/env:

```text
PREDICT_PROVIDER=llm_gateway | aihub | cem_rag
```

### `scripts`

Hiện tại:

- `backtest_runner.py`
- `backtest_rolling_eval.py`
- `model_backtest_compare.py`
- StockMem optimizer scripts.

Nâng cấp:

- `scripts/build_event_states.py`
- `scripts/build_cem_dataset.py`
- `scripts/train_cem_retriever.py`
- `scripts/train_cem_forecaster.py`
- `scripts/calibrate_cem.py`
- `scripts/evaluate_cem_rag.py`
- `scripts/run_ablation_suite.py`
- `scripts/statistical_tests.py`

---

## 8. Lộ trình thực thi theo phase

### Phase 0: Reproducibility baseline

Mục tiêu:

- Khóa lại hiện trạng và tạo baseline có thể tái lập.

Việc cần làm:

1. Xuất toàn bộ backtest hiện tại ra JSONL/CSV.
2. Tái lập số liệu trong `docs/backtest_report_2026-05.md`.
3. Tái lập kNN-Returns trong `MoTa.md` bằng script chính thức.
4. Ghi rõ split train/val/test.

Done khi:

- Có `artifacts/predictions/current_marketlens_test.jsonl`.
- Có `artifacts/metrics/current_marketlens_report.csv`.
- Có script chạy lại ra cùng số liệu trong sai số cho phép.

### Phase 1: Data protocol và event states

Mục tiêu:

- Có dataset event memory point-in-time.

Việc cần làm:

1. Thêm cutoff fields.
2. Build event extraction.
3. Build DailyEventState.
4. Persist event states.
5. Test leakage.

Done khi:

- Có event states train/val/test.
- Có unit tests cho extraction/cluster/novelty.
- Không có article sau cutoff trong dataset.

### Phase 2: Strong deterministic baselines

Mục tiêu:

- Có baseline mạnh trước khi làm neural.

Việc cần làm:

1. Fixed MarketLens-kNN.
2. kNN-Returns deterministic.
3. Logistic/XGBoost technical.
4. Event-only and price-only ML.
5. PatchTST-style price-only nếu đủ thời gian.

Done khi:

- Có main baseline table.
- Biết baseline mạnh nhất là gì.

### Phase 3: Learned retriever

Mục tiêu:

- Chứng minh retrieval học từ outcome tốt hơn fixed kNN.

Việc cần làm:

1. Build pair/triplet dataset.
2. Train learned retriever.
3. Evaluate hit direction@5.
4. Compare random/BM25/numeric/fixed kNN.
5. Inspect retrieval traces bằng domain knowledge.

Done khi:

- Learned retriever vượt fixed kNN trên validation và test.
- Có hard negative analysis.

### Phase 4: CEM-RAG forecaster

Mục tiêu:

- Model trả xác suất calibrated.

Việc cần làm:

1. Implement v1 ML forecaster trước.
2. Thêm event/retrieval features.
3. Calibration trên validation.
4. Policy tau tuning trên validation.
5. Evaluate trên test.

Done khi:

- Có `p_up`, `p_down`, `p_hold`.
- Có reliability diagram.
- Có policy tau cố định từ validation.

### Phase 5: ESWA-grade evaluation

Mục tiêu:

- Tạo result package đủ tin cậy cho paper.

Việc cần làm:

1. Main results table.
2. Trading table sau phí.
3. Ablation table.
4. Statistical tests.
5. Case studies.
6. Limitations.

Done khi:

- Pass acceptance gates hoặc có kết luận trung thực nếu fail.

---

## 9. Case studies cần chuẩn bị

Chọn tối thiểu 4 case:

1. Đúng BUY:
   - Event breadth cao.
   - Price momentum ủng hộ.
   - Retrieved cases cùng direction.

2. Đúng SELL:
   - Regulatory/security/systemic event xấu.
   - Bear regime hoặc negative momentum.
   - Retrieved cases cho negative future return.

3. Đúng HOLD:
   - Event tích cực nhưng RSI quá nóng.
   - Hoặc price bullish nhưng event xấu.
   - Model tránh ép BUY/SELL khi evidence mâu thuẫn.

4. Sai:
   - Phân tích failure trung thực.
   - Ví dụ sell-the-news, priced-in event, event polarity đúng nhưng expectation sai.

Template:

```text
Date:
Symbol:
Actual 7d return:
Prediction:
p_up/p_down/p_hold:
Top events:
Retrieved cases:
Baseline decision:
Why CEM-RAG succeeded/failed:
```

---

## 10. Cấu trúc paper đề xuất

1. Introduction
   - Crypto market event sensitivity.
   - Hạn chế của sentiment-only, price-only, LLM final decision.
   - Giới thiệu CEM-RAG.

2. Related Work
   - Financial sentiment analysis: FinBERT, CryptoBERT.
   - Time-series forecasting: PatchTST, LSTM/GRU, XGBoost.
   - Financial RAG/retrieval: FinSeer.
   - Event memory: StockMem.
   - Dissemination-aware news forecasting: FinGPT.

3. System Overview
   - MarketLens microservice architecture.
   - Point-in-time pipeline.
   - Data contracts and event memory.

4. Methodology
   - Event extraction.
   - Daily event memory.
   - Learned retriever.
   - Forecasting model.
   - Calibration and trading policy.

5. Experimental Design
   - Dataset.
   - Temporal split.
   - Baselines.
   - Metrics.
   - Leakage prevention.
   - Statistical tests.

6. Results
   - Main table.
   - Trading table.
   - Calibration.
   - Walk-forward robustness.

7. Ablation and Analysis
   - No events.
   - No novelty/source breadth.
   - No learned retriever.
   - No retrieval.
   - No calibration.

8. Case Studies
   - BUY, SELL, HOLD, failure case.

9. Limitations
   - Non-stationarity.
   - News source bias.
   - LLM extraction error.
   - Causal interpretation risk.
   - Crypto liquidity and regime shifts.

10. Conclusion

---

## 11. Cảnh báo khi viết claim

Không nên viết:

```text
CEM-RAG guarantees strong accept at ESWA.
```

Nên viết:

```text
CEM-RAG is designed to meet the methodological expectations of an ESWA-style
intelligent financial forecasting system, provided that held-out experiments
confirm improvements over strong baselines.
```

Không nên viết:

```text
LLM reasoning proves the signal is correct.
```

Nên viết:

```text
LLM-generated explanations are used only as auxiliary narratives; quantitative
claims are evaluated through held-out predictions, ablations, calibration, and
statistical tests.
```

Không nên viết:

```text
Accuracy is 60%, so trading is profitable.
```

Nên viết:

```text
Directional metrics are reported together with transaction-cost-adjusted
trading metrics including Sharpe, Sortino, max drawdown, turnover, and coverage.
```

---

## 12. Kết luận

MarketLens hiện có nền tảng kỹ thuật tốt: microservice rõ, StockMem có weighted retrieval, MainController có historical mode, FactorLedge có taxonomy, và backtest hiện tại đã chỉ ra điểm yếu cụ thể. Để đạt mức bài nghiên cứu cạnh tranh tại ESWA, nâng cấp cần tập trung vào core research thay vì chỉ thêm guardrails:

1. Biến tin tức thành event memory point-in-time.
2. Học retriever từ outcome thay vì chỉ dùng cosine cố định.
3. Dự báo xác suất `p_up`, `p_down`, `p_hold` thay vì LLM quyết định trực tiếp.
4. Calibration và trading policy phải tune trên validation.
5. Evaluation phải so với baseline mạnh, có ablation, significance tests và transaction cost.

Nếu CEM-RAG vượt baseline mạnh nhất trên held-out test, đạt gate MCC/Sharpe, và ablation chứng minh event memory + learned retrieval thật sự đóng góp, MarketLens có thể được định vị như một **expert/intelligent system for cryptocurrency event-driven forecasting** phù hợp scope ESWA.

