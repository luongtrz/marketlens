# Backtest Readiness Report — StockMem / POST /run

## 1. How the Backtest Pipeline Works

`POST /run?symbol=BTC&date=2026-02-12` triggers a 4-step pipeline that runs a
**single historical date** through the full prediction flow (no look-ahead):

### Step 1 — COLLECT (parallel, REQUIRED)
| Data source               | What it fetches                                     |
|---------------------------|-----------------------------------------------------|
| Supabase `news_articles`  | Articles from that date (via `shared.supabase_news`) |
| Binance (market_data:8002)| OHLCV candles & technical indicators                |

The `date` param ensures `publish_at <= cutoff` so no future articles leak in.
The last OHLCV candle ≤ target date is selected; indicators are computed from
the preceding 21 candles only.

### Step 2 — AI SCORE
| Source                                  | What                                          |
|-----------------------------------------|-----------------------------------------------|
| Article `sentiment_score` (pre-computed)| Averaged across all articles for that day     |
| Supabase `daily_factor_snapshots`       | Pre-computed factors JSONB (preferred)        |
| AIHub fallback (if Supabase missing)    | LLM-extracted factors from article text       |
| factor_ledge:8004 (`classify/vector`)   | 75d binary factor vector                      |

### Step 3 — STOCKMEM (REQUIRED)
| Action                                     | Why                                            |
|--------------------------------------------|------------------------------------------------|
| `stockmem.save(current_record)`            | Persist today's record                         |
| `stockmem.search(query, k, before_date)`   | Find k most similar past records (look-back only) |

### Step 4 — PREDICT (REQUIRED)
Calls `aihub.predict(current, similar)` — LLM generates signal with RAG context.

---

## 2. Data You Already Have (per `..myspace/docs/`)

| Table                      | Status   | Notes                                          |
|----------------------------|----------|------------------------------------------------|
| `news_articles`            | EXISTS   | Sample shows `id`, `header`, `content`, `publish_at`, `source_url`, `sentiment_score`, `summary` |
| `daily_factor_snapshots`   | EXISTS   | Has `symbol`, `snapshot_date`, `factors_json`, `factor_vector`, `type_vector`, `group_vector` |

Sample data:
- `news_articles`: 5 rows from 2026-04-10..11 with sentiment scores -0.97..0.94
- `daily_factor_snapshots`: 1 row for BTC on 2026-02-12 with 8 factors

---

## 3. Data Gaps — What's Missing to Run a Full Backtest

### 3.1 `news_articles` — coverage gap
The pipeline needs articles for **every single date** you want to backtest, plus
enough articles per date to produce a meaningful sentiment average.

**Checklist:**
- [ ] Articles exist for the full date range (e.g. 2023-01-01 → 2026-05-10)
- [ ] At least 3-5 articles per day with non-zero `sentiment_score`
- [ ] Articles are correctly matched by symbol (the pipeline uses token-level
  matching: "BTC" → word-boundary regex on header+content)
- [ ] `publish_at` is a proper `timestamp with time zone`

### 3.2 `daily_factor_snapshots` — coverage gap
The pipeline queries this table with `snapshot_date=eq.{date}&symbol=eq.{SYMBOL}`.
Only dates that have a row will be used; otherwise it falls back to AIHub (which
might not be available or reliable in batch).

**Checklist:**
- [ ] Snapshot exists for every date in the backtest range
- [ ] `factors_json` is a valid JSONB array of `{name, type, polarity, weight/confidence}` objects
- [ ] `factor_vector` is optionally populated (75-element smallint array) — avoids
  needing the classify-service for every date

### 3.3 StockMem `stockmem_records` — EMPTY (critical gap!)
The similarity search in Step 3 returns **nothing** if no past records exist.
The `before_date` filter also needs prior records to have been saved.

**To populate:** Use `POST /backfill?symbol=BTC&days=30&offset=0` in batches
to build the StockMem database. This endpoint:
1. Fetches articles from Supabase for each date in the window
2. Fetches OHLCV from Binance
3. Loads factors from `daily_factor_snapshots` (or AIHub fallback)
4. Computes indicators (RSI, MACD, price change, MSI)
5. Calls `classify/vector` on factor_ledge
6. Saves `StockMemRecord` to StockMem's DB

### 3.4 `future_return_1d/7d/30d` — missing from StockMem records
After populating StockMem, you need to compute actual forward returns so the
similarity search can evaluate which past patterns predicted good outcomes.

**To populate:** `POST /fill-returns?symbol=BTC` — fetches close prices from
Binance for D+1, D+7, D+30 and patches each record.

### 3.5 Services that must be running
```
stockmem (8003)     — record storage + similarity search
market_data (8002)  — Binance OHLCV + indicators
factor_ledge (8004) — classify/vector (75d) + ledger
aihub (8001)        — final prediction + factor fallback
main_controller (8005) — orchestra
```

The crawler service (8000) is **not strictly needed** for backtest because the
`CrawlerClient` reads directly from Supabase via `shared/supabase_news.py`.

---

## 4. Step-by-Step: How to Get a Backtest Running

### Phase A: Validate data quality
```bash
python ..myspace/scripts/check_backtest_data.py --symbol BTC
```

### Phase B: Backfill StockMem
```bash
# Start all services via docker-compose or individually
# Then batch-backfill StockMem (e.g. 90 days at a time)
curl -X POST "http://localhost:8005/backfill?symbol=BTC&days=90&offset=0"
curl -X POST "http://localhost:8005/backfill?symbol=BTC&days=90&offset=90"
# ... repeat until reaching your target date range
```

### Phase C: Compute future returns
```bash
curl -X POST "http://localhost:8005/fill-returns?symbol=BTC"
```

### Phase D: Run individual backtest predictions
```bash
curl -X POST "http://localhost:8005/run?symbol=BTC&date=2025-03-15"
curl -s "http://localhost:8005/status/{run_id}"
curl -s "http://localhost:8005/result/{run_id}"
```

### Phase E: Bulk walk-forward evaluation
Use `stockmem/scripts/backtest_api.py` for a full walk-forward backtest.
Requires a `mock_3y_records.json`-format dataset or similar.

---

## 5. Key Architecture Notes

1. **Supabase is the source of truth** for articles and daily factors — StockMem
   is a derived, vec-similarity-search-optimized store.

2. **No look-ahead**: The pipeline uses `before_date` in StockMem search and
   `publish_lte` in crawler reads to prevent future data leakage.

3. **Backfill reuses the same logic** as `/run` but batch-processes multiple dates,
   pre-computes indicators locally (RSI, MACD, MSI), and uses the same
   `classify/vector` endpoint.

4. **StockMem DB** can be either SQLite (local dev) or PostgreSQL (docker/prod).
   The table schema is `stockmem_records(id, record_date, symbol, payload)`.
