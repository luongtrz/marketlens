# MarketLens System Architecture

MarketLens is a modular crypto-market pipeline. The app code is kept in this
branch so the system can still run end to end, while the StockMem evaluation
docs focus on the research contribution.

## Services

| Service | Port | Role |
| --- | ---: | --- |
| Crawler | 8000 | Poll RSS/news sources, deduplicate articles, persist raw/enriched news. |
| AIHub | 8001 | Sentiment, factor extraction, LLM clients, prediction/evaluation helpers. |
| MarketData | 8002 | OHLCV snapshots, historical candles, technical indicators. |
| StockMem | 8003 | Daily memory records, vector embeddings, fixed/learned retrieval, and trend-aware evidence retrieval audits. |
| FactorLedge | 8004 | Factor normalization and factor summary APIs. |
| MainController | 8005 | Orchestrates pipeline runs and result assembly. |
| LLMGateway | 8006 | Optional external LLM decision gateway. |
| Frontend | 3000 | Dashboard and forecast UI. |

## Data Flow

```text
news/rss + market candles
        |
        v
Crawler + MarketData
        |
        v
AIHub sentiment/factors + FactorLedge normalization
        |
        v
StockMem daily record and historical retrieval
        |
        v
MainController forecast/explanation API
        |
        v
Frontend dashboard
```

The research path evaluated in this branch is narrower: it uses the frozen
StockMem NDJSON export, compares naive current-context LLM prompting against
structured StockMem baselines, and reports compact tables.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `aihub/` | LLM clients, sentiment/factor modules, naive LLM baseline evaluator. |
| `stockmem/` | Retrieval code, learned metrics, strict evaluators, reproducibility runner. |
| `docs/stockmem/` | Maintained StockMem methodology, experiments, and reproduction docs. |
| `docs/archive/` | Historical reports retained for audit, not primary reading material. |
| `docs/upgrade/` | Local copies of reference papers and external notes. |
| `scripts/archive/` | Legacy one-off research/backfill scripts. |

## Operational Notes

- Generated outputs are ignored: `artifacts/`, `results_tables/`, `submission/`.
- The official StockMem test window is `2025-07-01` to `2026-05-01`.
- The official D7 label threshold is `±2%` on `future_return_7d`.
- The maintained evidence retriever is `learned_recency_50_50`, documented in
  `docs/stockmem/trend_aware_retrieval.md`.
- The maintained offline decision head is `count_vote_buy3_sell4` over that
  retriever's top-10 evidence set.
- Multi-asset StockMem profiles are prepared in
  `stockmem/config/model_profiles.json`; BTC uses the maintained artifact,
  while ETH has reserved paths for ETH-specific fine-tuning.
- The main reproducibility command lives in
  `stockmem/scripts/run_submission_reproduction.py`.
