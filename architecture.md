# Crypto News Intelligence Pipeline — System Architecture

> **Purpose of this document**: Provide a complete, self-contained reference for an AI model (or engineer) to understand the full system — every module's responsibilities, interfaces, data models, internal structure, and how modules compose into the end-to-end pipeline. Each module is independently testable without its peers.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Shared Layer](#3-shared-layer)
4. [Module: Crawler](#4-module-crawler)
5. [Module: AIHub](#5-module-aihub)
6. [Module: MarketData](#6-module-marketdata)
7. [Module: StockMem](#7-module-stockmem)
8. [Module: FactorLedge](#8-module-factorledge)
9. [Module: MainController](#9-module-maincontroller)
10. [End-to-End Pipeline Flow](#10-end-to-end-pipeline-flow)
11. [Data Models (Canonical Schemas)](#11-data-models-canonical-schemas)
12. [Inter-Module Communication](#12-inter-module-communication)
13. [Testing Strategy](#13-testing-strategy)
14. [Configuration Reference](#14-configuration-reference)

---

## 1. System Overview

This system continuously monitors crypto-related news, scores it with AI models, correlates it with market data, stores and retrieves historical patterns, and uses retrieval-augmented generation (RAG) to produce trading signal explanations and predictions.

### High-Level Pipeline

```
RSS Feeds
   │
   ▼
┌──────────┐    factors list     ┌─────────────┐    processed factors    ┌──────────┐
│ Crawler  │ ──────────────────► │ FactorLedge │ ──────────────────────► │ StockMem │
│          │                     └─────────────┘                         │          │
│ (also    │                                                              │  daily   │
│  calls   │    news + metadata                                          │  record  │
│  AIHub   │ ──────────────────────────────────────────────────────────► │  store   │
│  inline) │                                                              └──────────┘
└──────────┘                                                                   │
     │                                                                   k-similar
     │                                                                   records
     │                                                                         │
┌────▼──────────────────────────────────────────────────────────────────────── ▼ ───┐
│                              MainController                                        │
│                                                                                    │
│  1. Trigger Crawler + MarketData in parallel                                       │
│  2. Call AIHub: sentiment score, factors, indicators                               │
│  3. Save record to StockMem                                                        │
│  4. Retrieve 5 most similar past records from StockMem                             │
│  5. Pass current record + 5 similar cases to AIHub (RAG predict/explain)           │
│  6. Emit final PredictionResult                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  PredictionResult
  { signal, confidence, explanation, similar_cases[] }
```

### Design Principles

- **Module Independence**: Every module exposes its own HTTP REST API and can be started, tested, and operated without other modules running.
- **No Shared State at Runtime**: Modules communicate through well-defined request/response contracts. No shared in-process state or shared database connections.
- **Fail-Safe Defaults**: Each module returns a partial result or an explicit error rather than crashing the pipeline.
- **Nullable Fields**: Fields like `summary` are explicitly nullable and downstream consumers must handle null.

---

## 2. Directory Structure

```
crypto-pipeline/
├── shared/
│   ├── models/
│   │   ├── article.py          # ArticleRecord, IngestionRecord
│   │   ├── market.py           # OHLCV, Indicator, MarketSnapshot
│   │   ├── factor.py           # Factor, FactorList, NormalizedFactor
│   │   ├── memory.py           # StockMemRecord, SimilarRecord
│   │   └── prediction.py       # PredictionResult, ExplainRequest
│   ├── messaging/
│   │   ├── base.py             # Abstract MessageBus interface
│   │   ├── kafka_bus.py        # Kafka implementation
│   │   └── redis_bus.py        # Redis Streams implementation
│   ├── config/
│   │   ├── base_config.py      # Base Pydantic settings
│   │   └── loader.py           # Config loader (env + yaml)
│   └── http_client.py          # Shared async HTTP client with retry
│
├── crawler/
│   ├── src/
│   │   ├── rss/
│   │   │   ├── fetcher.py      # RSS feed polling loop
│   │   │   ├── parser.py       # Feed entry → RawArticle
│   │   │   └── deduplicator.py # URL-based dedup (Redis or DB set)
│   │   ├── llm/
│   │   │   ├── client.py       # LLM API wrapper (calls AIHub or direct)
│   │   │   ├── prompts.py      # Prompt templates
│   │   │   └── parser.py       # LLM response → structured fields
│   │   ├── db/
│   │   │   ├── writer.py       # Write IngestionRecord to DB
│   │   │   └── models.py       # SQLAlchemy / ORM models
│   │   ├── config.py
│   │   └── main.py             # Entry point: starts polling loop
│   ├── tests/
│   │   ├── test_fetcher.py
│   │   ├── test_parser.py
│   │   ├── test_llm_client.py
│   │   └── fixtures/
│   │       └── sample_feed.xml
│   ├── Dockerfile
│   └── config.yaml
│
├── aihub/
│   ├── src/
│   │   ├── sentiment/
│   │   │   ├── model.py        # CryptoBert loader + inference
│   │   │   ├── preprocessor.py # Text cleaning for BERT input
│   │   │   └── schema.py       # SentimentRequest / SentimentResponse
│   │   ├── factors/
│   │   │   ├── skgp.py         # SKGP technique implementation
│   │   │   ├── extractor.py    # Text → factor list
│   │   │   └── schema.py       # FactorRequest / FactorResponse
│   │   ├── predict/
│   │   │   ├── client.py       # External model API (gpt-oss-120b)
│   │   │   ├── rag_builder.py  # Assembles RAG context from similar cases
│   │   │   ├── prompt.py       # Predict/explain prompt templates
│   │   │   └── schema.py       # PredictRequest / PredictResponse
│   │   ├── api.py              # FastAPI app: /sentiment /factors /predict
│   │   └── config.py
│   ├── tests/
│   │   ├── test_sentiment.py
│   │   ├── test_factors.py
│   │   ├── test_predict.py
│   │   └── fixtures/
│   │       └── sample_texts.json
│   ├── Dockerfile
│   └── config.yaml
│
├── market_data/
│   ├── src/
│   │   ├── sources/
│   │   │   ├── base.py         # Abstract MarketSource interface
│   │   │   ├── binance.py      # Binance REST + WebSocket adapter
│   │   │   └── tradingview.py  # TradingView scraper / API adapter
│   │   ├── indicators/
│   │   │   ├── macd.py         # MACD calculation
│   │   │   ├── rsi.py          # RSI calculation
│   │   │   ├── bollinger.py    # Bollinger Bands
│   │   │   └── registry.py     # Indicator registry (name → fn)
│   │   ├── db/
│   │   │   ├── writer.py       # Write OHLCV + indicators to DB
│   │   │   └── reader.py       # Query historical candles
│   │   ├── stream.py           # On-demand indicator calculation
│   │   ├── api.py              # FastAPI: /snapshot /history /indicators
│   │   └── config.py
│   ├── tests/
│   │   ├── test_binance.py
│   │   ├── test_indicators.py
│   │   └── fixtures/
│   │       └── sample_ohlcv.json
│   ├── Dockerfile
│   └── config.yaml
│
├── stockmem/
│   ├── src/
│   │   ├── store/
│   │   │   ├── writer.py       # Persist daily record
│   │   │   ├── reader.py       # Read by date / id
│   │   │   └── schema.py       # DB table models
│   │   ├── search/
│   │   │   ├── embedder.py     # Record → embedding vector
│   │   │   ├── index.py        # Vector index (FAISS or pgvector)
│   │   │   └── searcher.py     # k-NN search over index
│   │   ├── api.py              # FastAPI: /record /search
│   │   └── config.py
│   ├── tests/
│   │   ├── test_store.py
│   │   └── test_search.py
│   ├── Dockerfile
│   └── config.yaml
│
├── factor_ledge/
│   ├── src/
│   │   ├── processor/
│   │   │   ├── receiver.py     # Accept raw factor list from Crawler
│   │   │   └── pipeline.py     # Orchestrate clean → normalize → enrich
│   │   ├── normalizer/
│   │   │   ├── cleaner.py      # Dedup, lowercase, trim noise
│   │   │   ├── scorer.py       # Assign weights to factors
│   │   │   └── enricher.py     # Attach sector / asset metadata
│   │   ├── api.py              # FastAPI: /ingest /factors /summary
│   │   └── config.py
│   ├── tests/
│   │   ├── test_receiver.py
│   │   └── test_normalizer.py
│   ├── Dockerfile
│   └── config.yaml
│
├── main_controller/
│   ├── src/
│   │   ├── orchestrator/
│   │   │   ├── pipeline.py     # Full pipeline run() method
│   │   │   ├── steps.py        # Individual step functions
│   │   │   └── context.py      # PipelineContext dataclass
│   │   ├── scheduler/
│   │   │   ├── cron.py         # APScheduler jobs
│   │   │   └── trigger.py      # Manual / event-driven trigger
│   │   ├── clients/
│   │   │   ├── crawler_client.py
│   │   │   ├── aihub_client.py
│   │   │   ├── market_client.py
│   │   │   ├── stockmem_client.py
│   │   │   └── factorledge_client.py
│   │   ├── api.py              # FastAPI: /run /status /result
│   │   └── main.py
│   ├── tests/
│   │   ├── test_pipeline.py    # Integration test with mocked clients
│   │   └── test_steps.py
│   ├── Dockerfile
│   └── config.yaml
│
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 3. Shared Layer

The `shared/` package is imported by all modules. It contains no business logic — only data contracts, base classes, and utilities.

### 3.1 Data Models

All models are Pydantic v2 `BaseModel` subclasses with `model_config = ConfigDict(extra="ignore")` so unknown fields are silently dropped (forward compatibility).

### 3.2 `shared/http_client.py`

Wraps `httpx.AsyncClient` with:
- Configurable retry (exponential backoff, max 3 attempts)
- Default timeout: 10s connect, 30s read
- JSON serialization with datetime ISO 8601

Usage pattern in every module client:
```python
from shared.http_client import get_client

async with get_client() as client:
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()
```

### 3.3 `shared/messaging/`

Abstract `MessageBus` with `publish(topic, payload)` and `subscribe(topic, handler)`. Concrete implementations: `KafkaBus` and `RedisBus`. Used optionally by Crawler → FactorLedge async path.

---

## 4. Module: Crawler

### 4.1 Responsibility

Continuously polls RSS feeds, deduplicates articles, calls the LLM layer (via AIHub or directly) to enrich each article, and persists results to the Ingestion Database.

### 4.2 Internal Flow

```
RSSFetcher.poll()
    │
    ▼
Parser.parse(entry) → RawArticle
    │
    ▼
Deduplicator.is_seen(url)?  ──yes──► skip
    │ no
    ▼
LLMClient.enrich(raw_article)
    ├── sentiment_score   (float, -1 to 1)
    ├── summary           (str | None)  ← nullable, controlled by config
    └── factors           (list[str])
    │
    ▼
DBWriter.write(IngestionRecord)
    │
    ▼
(optional) MessageBus.publish("factors.raw", factors)
```

### 4.3 `src/rss/fetcher.py`

```python
class RSSFetcher:
    def __init__(self, sources: list[FeedSource], poll_interval_seconds: int): ...
    async def poll_forever(self) -> None: ...
    async def fetch_one(self, source: FeedSource) -> list[RawArticle]: ...
```

`FeedSource`: `{ name: str, url: str, category: str }`

### 4.4 `src/rss/deduplicator.py`

```python
class Deduplicator:
    # Backed by Redis SET or in-memory set for testing
    async def is_seen(self, url: str) -> bool: ...
    async def mark_seen(self, url: str) -> None: ...
```

### 4.5 `src/llm/client.py`

```python
class LLMClient:
    # Calls AIHub /sentiment and /factors, or direct LLM if AIHub is down
    async def enrich(self, article: RawArticle) -> EnrichedFields: ...
    async def summarize(self, text: str) -> str | None: ...
```

`EnrichedFields`: `{ sentiment_score: float, summary: str | None, factors: list[str] }`

### 4.6 `src/db/writer.py`

```python
class IngestionDBWriter:
    async def write(self, record: IngestionRecord) -> str:  # returns record_id
        ...
```

### 4.7 Configuration (`config.yaml`)

```yaml
crawler:
  poll_interval_seconds: 60
  enable_summary: false        # Controls whether summary is generated
  feeds:
    - name: "CoinTelegraph"
      url: "https://cointelegraph.com/rss"
      category: "crypto_news"
  aihub_url: "http://aihub:8001"
  db_url: "postgresql+asyncpg://..."
  dedup_backend: "redis"       # redis | memory
  redis_url: "redis://redis:6379"
  factor_publish: true         # Whether to emit to MessageBus
```

### 4.8 Standalone Test Mode

Set `AIHUB_URL=mock` — the LLMClient returns a fixed `EnrichedFields` stub. Set `DB_URL=sqlite+aiosqlite:///test.db` for ephemeral DB. The module has no other external dependency.

---

## 5. Module: AIHub

### 5.1 Responsibility

Provides three AI inference endpoints as a service. Each endpoint is independently usable.

| Endpoint | Model | Input | Output |
|---|---|---|---|
| `POST /sentiment` | CryptoBert | `{ text: str }` | `{ score: float, label: str }` |
| `POST /factors` | SKGP | `{ text: str }` | `{ factors: list[Factor] }` |
| `POST /predict` | GPT-oss-120b (external) | `{ current: Record, similar: list[Record] }` | `{ signal, confidence, explanation }` |

### 5.2 `src/sentiment/model.py`

```python
class CryptoBertModel:
    def __init__(self, model_path: str): ...
    def load(self) -> None: ...
    def predict(self, text: str) -> SentimentResult: ...
        # SentimentResult: { score: float[-1,1], label: "bullish"|"bearish"|"neutral" }
```

Model is loaded once at startup and kept in memory. Thread-safe inference via `asyncio.to_thread`.

### 5.3 `src/factors/skgp.py`

```python
class SKGPExtractor:
    """
    Structured Knowledge-Guided Parsing.
    Parses article text to extract named factors (entities, events, macro signals)
    relevant to crypto markets.
    """
    def extract(self, text: str) -> list[Factor]: ...
        # Factor: { name: str, type: FactorType, polarity: float, confidence: float }
```

`FactorType` enum: `MACRO | REGULATORY | TECHNICAL | SENTIMENT | ON_CHAIN | EXCHANGE`

### 5.4 `src/predict/rag_builder.py`

```python
class RAGContextBuilder:
    def build(self, current: StockMemRecord, similar: list[SimilarRecord]) -> str:
        """
        Assembles a prompt context string from the current record and k similar
        historical cases. Format:
          === Current Situation ===
          <current record fields>
          === Similar Historical Cases ===
          Case 1 (similarity=0.92, date=...):
            <fields>
            Outcome: <what happened>
          ...
        """
```

### 5.5 `src/api.py` (FastAPI)

```python
app = FastAPI(title="AIHub")

@app.post("/sentiment")   -> SentimentResponse
@app.post("/factors")     -> FactorResponse
@app.post("/predict")     -> PredictResponse
@app.get("/health")       -> { "status": "ok", "models_loaded": bool }
```

### 5.6 Standalone Test Mode

All three model backends accept an `AIHUB_MOCK=true` env var that returns deterministic fixture responses. External GPT-oss-120b call is mockable via `PREDICT_BACKEND=mock`.

---

## 6. Module: MarketData

### 6.1 Responsibility

Fetches OHLCV candles from Binance and TradingView, calculates technical indicators (MACD, RSI, Bollinger Bands, etc.), and provides both historical queries and on-demand streaming calculation.

### 6.2 Endpoints

```
GET  /snapshot?symbol=BTCUSDT&interval=1h        → MarketSnapshot
GET  /history?symbol=BTCUSDT&interval=1h&limit=200 → list[OHLCV]
POST /indicators  { ohlcv: list[OHLCV], indicators: ["macd","rsi"] }
                   → { macd: MACDResult, rsi: RSIResult }
GET  /health
```

### 6.3 `src/sources/base.py`

```python
class MarketSource(ABC):
    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int
    ) -> list[OHLCV]: ...

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker: ...
```

`OHLCV`: `{ timestamp: datetime, open, high, low, close, volume: float }`

### 6.4 `src/indicators/registry.py`

```python
INDICATOR_REGISTRY: dict[str, Callable[[list[OHLCV]], Any]] = {
    "macd": calculate_macd,
    "rsi":  calculate_rsi,
    "bb":   calculate_bollinger,
    "ema":  calculate_ema,
    "vwap": calculate_vwap,
}

def calculate_indicators(ohlcv: list[OHLCV], names: list[str]) -> dict[str, Any]:
    return {name: INDICATOR_REGISTRY[name](ohlcv) for name in names}
```

### 6.5 `MarketSnapshot` schema

```python
class MarketSnapshot(BaseModel):
    symbol: str
    timestamp: datetime
    ohlcv: OHLCV               # Latest candle
    indicators: dict[str, Any] # Computed indicators at latest candle
    source: str                # "binance" | "tradingview"
```

### 6.6 Standalone Test Mode

Set `MARKET_SOURCE=mock` — returns deterministic fixture data from `tests/fixtures/sample_ohlcv.json`. No external network calls needed.

---

## 7. Module: StockMem

### 7.1 Responsibility

Stores one `StockMemRecord` per day (or per pipeline run). Provides vector-similarity search to retrieve the k most similar past records given a query record.

### 7.2 Endpoints

```
POST /record              { record: StockMemRecord } → { id: str }
GET  /record/{id}                                    → StockMemRecord
POST /search              { query: StockMemRecord, k: int = 5 }
                           → { results: list[SimilarRecord] }
GET  /health
```

### 7.3 `src/store/writer.py`

```python
class RecordWriter:
    async def save(self, record: StockMemRecord) -> str:
        """
        Persists record to relational DB.
        Also triggers embedder to compute and store vector in vector index.
        Returns record id (UUID).
        """
```

### 7.4 `src/search/embedder.py`

```python
class RecordEmbedder:
    """
    Converts a StockMemRecord to a dense vector for similarity search.
    Embedding strategy:
      - Concatenate: [sentiment_score, rsi, macd_hist, ...factors_tfidf...]
      - Normalize to unit sphere
    """
    def embed(self, record: StockMemRecord) -> np.ndarray: ...
```

### 7.5 `StockMemRecord` schema

```python
class StockMemRecord(BaseModel):
    id: str | None = None          # Assigned on write
    date: date
    symbol: str
    sentiment_score: float
    factors: list[str]
    market_snapshot: MarketSnapshot
    summary: str | None = None
    article_ids: list[str] = []    # IDs from Ingestion DB
```

### 7.6 `SimilarRecord` schema

```python
class SimilarRecord(BaseModel):
    record: StockMemRecord
    similarity: float              # Cosine similarity [0, 1]
    outcome: str | None = None     # What happened after this date, if known
```

### 7.7 Standalone Test Mode

Set `VECTOR_BACKEND=memory` — uses in-memory FAISS index. Set `DB_URL=sqlite+aiosqlite:///test.db`.

---

## 8. Module: FactorLedge

### 8.1 Responsibility

Receives raw factor lists from the Crawler (via HTTP push or message bus), applies a cleaning and normalization pipeline, optionally enriches with asset/sector metadata, and forwards processed factors to StockMem.

### 8.2 Endpoints

```
POST /ingest    { article_id: str, factors: list[str], source: str }
                 → { processed: list[NormalizedFactor] }
GET  /factors   ?symbol=BTC&limit=50  → list[NormalizedFactor]
GET  /summary   ?symbol=BTC           → FactorSummary
GET  /health
```

### 8.3 `src/processor/pipeline.py`

```python
class FactorPipeline:
    """
    Steps:
    1. Cleaner: lowercase, dedup, remove stopwords, trim noise tokens
    2. Scorer:  assign a weight [0,1] to each factor based on frequency + recency
    3. Enricher: attach { sector, asset_class, related_symbols } from a lookup table
    """
    def run(self, raw: list[str], article_id: str) -> list[NormalizedFactor]: ...
```

### 8.4 `NormalizedFactor` schema

```python
class NormalizedFactor(BaseModel):
    name: str
    type: FactorType
    weight: float              # 0–1
    polarity: float            # -1 to 1
    sector: str | None
    related_symbols: list[str]
    source_article_id: str
    observed_at: datetime
```

### 8.5 Standalone Test Mode

`/ingest` can be called independently with any factor list. The Enricher uses a local static lookup table — no network calls needed.

---

## 9. Module: MainController

### 9.1 Responsibility

Owns the master pipeline logic. Calls all other modules in the correct order, assembles the final `PredictionResult`, handles partial failures gracefully, and exposes run control via its own API.

### 9.2 Endpoints

```
POST /run       { symbol: str, trigger: "manual"|"scheduled" }
                 → { run_id: str, status: "started" }
GET  /status/{run_id}  → RunStatus
GET  /result/{run_id}  → PredictionResult
GET  /health
```

### 9.3 `src/orchestrator/pipeline.py`

```python
class Pipeline:
    def __init__(self, clients: ModuleClients, config: PipelineConfig): ...

    async def run(self, symbol: str) -> PredictionResult:
        ctx = PipelineContext(symbol=symbol, run_id=uuid4())
        await self._step_collect(ctx)      # Crawler + MarketData in parallel
        await self._step_ai_score(ctx)     # AIHub sentiment + factors
        await self._step_stockmem(ctx)     # Save record + retrieve similar
        await self._step_predict(ctx)      # AIHub RAG predict/explain
        return ctx.build_result()
```

### 9.4 `src/orchestrator/steps.py`

```python
async def step_collect(ctx: PipelineContext, clients: ModuleClients) -> None:
    """
    Runs Crawler trigger and MarketData snapshot concurrently.
    Stores results in ctx.latest_articles and ctx.market_snapshot.
    On individual failure: logs warning, stores None, continues.
    """
    results = await asyncio.gather(
        clients.crawler.get_latest(ctx.symbol),
        clients.market.get_snapshot(ctx.symbol),
        return_exceptions=True
    )
    ctx.latest_articles = results[0] if not isinstance(results[0], Exception) else []
    ctx.market_snapshot = results[1] if not isinstance(results[1], Exception) else None

async def step_ai_score(ctx: PipelineContext, clients: ModuleClients) -> None:
    """Calls AIHub /sentiment and /factors, attaches to ctx."""

async def step_stockmem(ctx: PipelineContext, clients: ModuleClients) -> None:
    """Saves current record, retrieves k=5 most similar past records."""

async def step_predict(ctx: PipelineContext, clients: ModuleClients) -> None:
    """Calls AIHub /predict with current context + similar cases (RAG)."""
```

### 9.5 `PipelineContext` dataclass

```python
@dataclass
class PipelineContext:
    symbol: str
    run_id: UUID
    latest_articles: list[IngestionRecord] = field(default_factory=list)
    market_snapshot: MarketSnapshot | None = None
    sentiment_score: float | None = None
    factors: list[NormalizedFactor] = field(default_factory=list)
    current_record_id: str | None = None
    similar_records: list[SimilarRecord] = field(default_factory=list)
    prediction: PredictResponse | None = None
    errors: list[str] = field(default_factory=list)

    def build_result(self) -> PredictionResult: ...
```

### 9.6 `src/clients/` — Module Clients

Each client in `src/clients/` is a thin async HTTP wrapper with:
- Base URL from config
- `health_check() -> bool`
- All request/response types imported from `shared/models/`
- Mockable via dependency injection in tests

### 9.7 Standalone Test Mode

All module clients accept a `MockClient` implementation injected via `Pipeline(clients=MockModuleClients(...))`. `test_pipeline.py` runs the full pipeline with all clients mocked — no real modules needed.

---

## 10. End-to-End Pipeline Flow

### Triggered Run (detailed steps)

```
MainController.POST /run { symbol: "BTCUSDT" }
│
├── STEP 1: Collect  [parallel]
│   ├── CrawlerClient.get_latest("BTCUSDT")
│   │     → list[IngestionRecord] (latest enriched articles)
│   └── MarketClient.get_snapshot("BTCUSDT")
│         → MarketSnapshot { ohlcv, indicators: { rsi, macd, bb } }
│
├── STEP 2: AI Score
│   ├── AIHubClient.sentiment(combined_text)
│   │     → { score: 0.42, label: "bullish" }
│   └── AIHubClient.factors(combined_text)
│         → list[Factor]
│         → FactorLedgeClient.ingest(factors)
│               → list[NormalizedFactor]
│
├── STEP 3: StockMem
│   ├── StockMemClient.save(StockMemRecord { ... })
│   │     → record_id
│   └── StockMemClient.search(query=current_record, k=5)
│         → list[SimilarRecord] (5 most similar past situations)
│
└── STEP 4: RAG Predict
    └── AIHubClient.predict({
            current: current_record,
            similar: similar_records   # The 5 retrieved cases
        })
        → PredictResponse {
              signal: "BUY" | "SELL" | "HOLD",
              confidence: float,
              explanation: str,
              reasoning_steps: list[str]
          }

RESULT: PredictionResult assembled from all ctx fields
```

---

## 11. Data Models (Canonical Schemas)

### `IngestionRecord`

```python
class IngestionRecord(BaseModel):
    id: str
    article_name: str
    source: str
    url: str
    date_published: datetime
    date_crawled: datetime
    summary: str | None = None         # Nullable — generated only if enabled
    sentiment_score: float             # -1.0 (bearish) to 1.0 (bullish)
    sentiment_label: str               # "bullish" | "bearish" | "neutral"
    factors: list[str]                 # Raw factor strings from LLM
    raw_text: str | None = None        # Original article body (may be truncated)
    metadata: dict[str, Any] = {}      # Source-specific extra fields
```

### `OHLCV`

```python
class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str   # "1m" | "5m" | "1h" | "4h" | "1d"
```

### `MarketSnapshot`

```python
class MarketSnapshot(BaseModel):
    symbol: str
    timestamp: datetime
    ohlcv: OHLCV
    indicators: dict[str, Any]  # { "rsi": 58.4, "macd": { "macd": ..., "signal": ..., "hist": ... } }
    source: str
```

### `Factor`

```python
class FactorType(str, Enum):
    MACRO       = "macro"
    REGULATORY  = "regulatory"
    TECHNICAL   = "technical"
    SENTIMENT   = "sentiment"
    ON_CHAIN    = "on_chain"
    EXCHANGE    = "exchange"

class Factor(BaseModel):
    name: str
    type: FactorType
    polarity: float      # -1 to 1: negative = bearish signal for this factor
    confidence: float    # 0 to 1
```

### `StockMemRecord`

```python
class StockMemRecord(BaseModel):
    id: str | None = None
    date: date
    symbol: str
    sentiment_score: float
    sentiment_label: str
    factors: list[str]
    normalized_factors: list[NormalizedFactor] = []
    market_snapshot: MarketSnapshot
    summary: str | None = None
    article_ids: list[str] = []
    run_id: str | None = None
```

### `PredictionResult`

```python
class SignalType(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class PredictionResult(BaseModel):
    run_id: str
    symbol: str
    timestamp: datetime
    signal: SignalType
    confidence: float              # 0 to 1
    explanation: str               # Human-readable narrative
    reasoning_steps: list[str]     # Step-by-step reasoning from LLM
    similar_cases: list[SimilarRecord]  # The 5 retrieved cases
    sentiment_score: float
    key_factors: list[NormalizedFactor]
    market_snapshot: MarketSnapshot | None
    errors: list[str] = []         # Non-fatal errors encountered during run
```

---

## 12. Inter-Module Communication

### Primary: HTTP REST

All modules expose FastAPI applications. The default port assignments are:

| Module | Default Port |
|---|---|
| Crawler | 8000 |
| AIHub | 8001 |
| MarketData | 8002 |
| StockMem | 8003 |
| FactorLedge | 8004 |
| MainController | 8005 |

### Optional: Async Messaging (Crawler → FactorLedge)

If `factor_publish: true` is set in Crawler config, raw factor lists are published to the `factors.raw` topic via `MessageBus`. FactorLedge subscribes and processes them asynchronously. This path is optional — the synchronous HTTP path is the default and is always available.

### Service Discovery

URLs are passed as environment variables and loaded via `shared/config/loader.py`. No service registry is required. For Docker Compose, service names act as hostnames.

---

## 13. Testing Strategy

### Unit Tests (per module)

Each module's `tests/` directory contains unit tests that mock all external dependencies (other modules, DBs, external APIs). Every module can run `pytest tests/` in isolation.

Key fixtures:
- `tests/fixtures/` — static JSON/XML files for RSS feeds, OHLCV data, LLM responses
- `conftest.py` — shared pytest fixtures (mock DB session, mock HTTP clients)

### Module Integration Tests

Each module has an integration test mode using:
- SQLite (in-memory) instead of PostgreSQL
- `MOCK=true` env var to disable external HTTP calls
- Real internal logic (parsers, indicators, normalizers) tested against fixtures

Running: `pytest tests/ -m integration`

### Pipeline Integration Test (MainController)

`main_controller/tests/test_pipeline.py` injects `MockModuleClients` into `Pipeline` and exercises the full `run()` flow end-to-end. This tests step sequencing, error handling (one module times out), and `PredictionResult` assembly.

### Contract Tests

`shared/models/` acts as the contract. Any breaking change to a model (removing a field, changing a type) will break deserialization in all consuming modules. CI runs `pytest shared/` first as a gate.

---

## 14. Configuration Reference

### Environment Variables (all modules)

| Variable | Description | Default |
|---|---|---|
| `ENV` | `development` / `production` | `development` |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` | `INFO` |
| `DB_URL` | SQLAlchemy DB URL | module-specific |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `MOCK` | `true` to disable all external calls | `false` |

### Module-Specific URL Variables (consumed by MainController clients)

| Variable | Module |
|---|---|
| `CRAWLER_URL` | Crawler |
| `AIHUB_URL` | AIHub |
| `MARKET_DATA_URL` | MarketData |
| `STOCKMEM_URL` | StockMem |
| `FACTOR_LEDGE_URL` | FactorLedge |

### `docker-compose.yml` (abbreviated)

```yaml
services:
  crawler:
    build: ./crawler
    ports: ["8000:8000"]
    environment:
      AIHUB_URL: http://aihub:8001
      DB_URL: postgresql+asyncpg://postgres:pass@db:5432/ingestion
      REDIS_URL: redis://redis:6379

  aihub:
    build: ./aihub
    ports: ["8001:8001"]
    environment:
      GPT_API_KEY: ${GPT_API_KEY}
      GPT_MODEL: gpt-oss-120b

  market_data:
    build: ./market_data
    ports: ["8002:8002"]
    environment:
      BINANCE_API_KEY: ${BINANCE_API_KEY}
      DB_URL: postgresql+asyncpg://postgres:pass@db:5432/market

  stockmem:
    build: ./stockmem
    ports: ["8003:8003"]
    environment:
      DB_URL: postgresql+asyncpg://postgres:pass@db:5432/stockmem
      VECTOR_BACKEND: pgvector     # pgvector | faiss | memory

  factor_ledge:
    build: ./factor_ledge
    ports: ["8004:8004"]
    environment:
      STOCKMEM_URL: http://stockmem:8003

  main_controller:
    build: ./main_controller
    ports: ["8005:8005"]
    environment:
      CRAWLER_URL: http://crawler:8000
      AIHUB_URL: http://aihub:8001
      MARKET_DATA_URL: http://market_data:8002
      STOCKMEM_URL: http://stockmem:8003
      FACTOR_LEDGE_URL: http://factor_ledge:8004

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: pass

  redis:
    image: redis:7-alpine
```

---

*Document version: 1.0 — generated for AI model consumption. All module contracts are defined by the schemas in `shared/models/`. Breaking changes to shared models must be versioned.*