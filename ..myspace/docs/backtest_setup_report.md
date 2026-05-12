# Backtest System — Setup & Developer Report

## 1. Changes Summary

### 1.1 Bug Fix: `factor_ledge/start.sh` — Docker crash on Windows

**Symptom:** Container `factor_ledge` exited immediately:
```
exec /app/start.sh: no such file or directory
```

**Root cause:** Git on Windows checked out the file with CRLF (`\r\n`) line endings.
The shebang `#!/bin/bash\r` caused the Linux kernel to search for `/bin/bash\r`
(doesn't exist) instead of `/bin/bash`.

**Fix:** Convert line endings from CRLF to LF. Add `.gitattributes` entry to
prevent recurrence:
```
factor_ledge/start.sh text eol=lf
```

### 1.2 Bug Fix: `aihub/src/predict/rag_builder.py` — Serialization errors

**Symptom 1:** `'dict' object has no attribute 'name'`

**Root cause:** `StockMemRecord.normalized_factors` is typed as `list[Any]`.
After JSON serialization (model_dump → HTTP → model_validate), nested
`NormalizedFactor` objects become plain `dict`. Code at line 158 accessed
`nf.name` directly, failing on dicts.

**Symptom 2:** `Unknown format code 'f' for object of type 'str'`

**Root cause:** Same dict deserialization issue. After fix #1, `_nf_val`
returned string `"0.5"` from dict, but `:.2f` format requires float.

**Fix:** Added two helper functions (lines 32-46):
```python
def _nf_val(nf, key, default="?"):   # dict-compatible field access
def _nf_float(nf, key, default=0.0): # dict-compatible float cast
```
Updated `record_to_text()` line 173 to use these helpers for `name`, `type`,
`weight`, `polarity`.

### 1.3 New File: `scripts/backtest_runner.py` — Bulk backtest engine

Walk-forward evaluation script. Features:

| Feature | Detail |
|---|---|
| **Data source** | Supabase `daily_factor_snapshots` → discover testable dates |
| **Prediction** | Calls `POST /run?symbol=BTC&date=...` (full 4-step pipeline + LLM) |
| **Ground truth** | Binance OHLCV via `market_data:8002` → actual D+1/D+7/D+30 returns |
| **Correctness** | BUY ➜ correct if ret>0; SELL ➜ ret<0; HOLD ➜ abs(ret)<1% |
| **Storage** | Upserts to Supabase `backtest_results` table |
| **Resumable** | Skips already-completed dates automatically |
| **Warmup** | `--warmup N` — first N days saved but excluded from accuracy stats |
| **Reverse** | `--reverse` — newest-first order (leverages existing StockMem data) |
| **Dry-run** | `--dry-run` — show pending dates without executing |
| **Stats** | `--stats` — print cumulative accuracy from Supabase (no LLM calls) |

### 1.4 New Supabase Table: `backtest_results`

SQL: `..myspace/docs/backtest_results_schema.sql`

```sql
backtest_results (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    symbol TEXT, backtest_date DATE (UNIQUE pair),
    run_id TEXT,
    signal TEXT CHECK (BUY/SELL/HOLD),
    confidence FLOAT, sentiment_score FLOAT,
    actual_return_1d/7d/30d FLOAT,
    correct_1d/7d/30d BOOLEAN,
    n_similar_cases INTEGER,
    top_similar_date DATE, top_similar_similarity FLOAT,
    top_similar_ret7d FLOAT,
    explanation TEXT, errors JSONB,
    run_duration_ms INTEGER,
    created_at/updated_at TIMESTAMPTZ
)
```

---

## 2. Data Flow — How a Single Backtest Works

```
POST /run?symbol=BTC&date=2024-03-01
│
├─ Step 1 COLLECT (parallel, REQUIRED)
│   ├─ Supabase news_articles → IngestionRecord[] (publish_at ≤ cutoff)
│   └─ Binance (market_data:8002) → MarketSnapshot (last candle ≤ date)
│       └─ /history?symbol=BTC&interval=1d&limit=50&end_time={cutoff_ts}
│
├─ Step 2 AI SCORE
│   ├─ Sentiment: mean(article.sentiment_score) → ctx.sentiment_score
│   ├─ Factors: Supabase daily_factor_snapshots → Factor[]
│   │   └─ Query: snapshot_date=eq.{date}, symbol=eq.BTC
│   │   └─ Fallback: AIHub LLM extraction if Supabase has no snapshot
│   ├─ FactorLedge (8004): POST /classify/vector → 75d binary vector
│   └─ FactorLedge (8004): POST /ledger/update → rolling window factors
│
├─ Step 3 STOCKMEM (REQUIRED)
│   ├─ POST /record → save StockMemRecord (date, factors, market_snapshot, ...)
│   └─ POST /search (k=5, before_date=date) → SimilarRecord[]
│       ├─ Embedding: factor_vector (75d) + indicator_vec (5d) + price
│       └─ Similarity: weighted cosine (w_factor=0.35, w_indicator=0.20, w_price=0.45)
│
└─ Step 4 PREDICT (AIHub:8001, REQUIRED)
    └─ POST /predict {current: StockMemRecord, similar: SimilarRecord[]}
        ├─ RAG Builder: current + 5 similar cases → structured prompt
        └─ Groq LLM (llama-3.3-70b) → signal + confidence + explanation
```

### Key constraints (no look-ahead bias):
- Articles filtered by `publish_at ≤ cutoff` (midnight of target date)
- OHLCV uses `end_time` so future candles are excluded
- StockMem search uses `before_date` to exclude records on/after target date
- Forward returns computed from actual Binance data D+1/D+7/D+30

---

## 3. Environment Setup

### 3.1 `.env` (root directory)

```env
# Required
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
AIHUB_GROQ_API_KEY=gsk_...       # or GEMINI_API_KEY / OPENAI_API_KEY
AIHUB_LLM_BACKEND=groq           # groq | gemini | openai

# Optional overrides
AIHUB_GROQ_MODEL=llama-3.3-70b-versatile
```

### 3.2 Start services

```bash
cd marketlens
docker compose up -d
```

Verification:
```bash
# All 6 services + db + redis
docker compose ps
# Ports: 8000(crawler) 8001(aihub) 8002(market_data) 8003(stockmem) 8004(factor_ledge) 8005(main_controller)
```

### 3.3 Create Supabase table

Run `..myspace/docs/backtest_results_schema.sql` in Supabase SQL Editor.

---

## 4. Usage

### 4.1 Initial data population (one-time)

```bash
# Populate StockMem with historical records
curl -X POST "http://localhost:8005/backfill?symbol=BTC&days=90&offset=0"
# Repeat with offset+=90 until full range covered

# Compute forward returns for all records
curl -X POST "http://localhost:8005/fill-returns?symbol=BTC"
```

### 4.2 Single backtest

```bash
curl -X POST "http://localhost:8005/run?symbol=BTC&date=2024-03-01"
curl "http://localhost:8005/status/{run_id}"
curl "http://localhost:8005/result/{run_id}"
```

### 4.3 Bulk backtest

```bash
# Recommended: reverse order (2026→2023), 60-day warmup, limit 100
python scripts/backtest_runner.py --symbol BTC --reverse --warmup 60 --max 100

# Specific date range
python scripts/backtest_runner.py --symbol BTC --start 2024-01-01 --end 2024-12-31

# All dates (1000+ days, high LLM token cost)
python scripts/backtest_runner.py --symbol BTC --reverse --warmup 60

# View accuracy stats (no API calls)
python scripts/backtest_runner.py --symbol BTC --stats

# Dry run: see what would be executed
python scripts/backtest_runner.py --symbol BTC --dry-run
```

### 4.4 Data validation

```bash
python ..myspace/scripts/check_backtest_data.py --symbol BTC --start 2023-01-01 --end 2026-05-11
```

---

## 5. Data Sources Summary

| Source | Contains | Size |
|---|---|---|
| Supabase `news_articles` | Articles (2023-2026) with pre-computed `sentiment_score` (CryptoBERT/FinBERT) | 54,930 rows |
| Supabase `daily_factor_snapshots` | Per-day factors (name, type, polarity, weight) + 75d binary vector | 1,190 BTC rows |
| Supabase `backtest_results` | Prediction results + ground truth (NEW) | — |
| StockMem PostgreSQL | Historical records with OHLCV, factors, forward returns | 1,225 records |
| Binance (via market_data) | Real-time OHLCV candles | Unlimited history |
| Groq (via AIHub) | LLM inference (llama-3.3-70b) | ~500 tokens/run |

---

## 6. Known Issues & Notes

| Item | Detail |
|---|---|
| **Groq rate limit** | Free tier ~30 req/min, ~1000 req/day. `HOLD(0.00)` = Groq returned error. Use `--reverse` to test recent dates first. |
| **Warmup critical** | StockMem has 1,225 records now, but first N days of backtest have minimal history. Recommend `--warmup 60`. |
| **`key_factors` empty** | By design — `update_ledger()` returns `[]` (side-effect store). Factors still used in prediction via `record.factors` + `factor_vector`. |
| **`sentiment_label` missing** | Field not present in `PredictionResult` model. |
| **HOLD in strong trends** | Model occasionally outputs HOLD when market is clearly trending — needs confidence calibration. |
| **Windows CRLF** | Files edited on Windows may reintroduce CRLF. Check `.gitattributes` for: `*.sh text eol=lf`. |
