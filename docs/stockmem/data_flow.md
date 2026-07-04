# StockMem Data Flow

This document explains the StockMem data flow from raw records to retrieval,
decision heads, evaluation tables, and endpoint profiles. It is written as an
implementation map: each section lists the relevant files and what each file
does.

## 1. Big Picture

StockMem is a structured historical-memory pipeline:

```text
raw market/news/factor data
  -> StockMem records
  -> vector blocks
  -> historical candidate pool
  -> retriever score
  -> top-k evidence
  -> decision head
  -> BUY/HOLD/SELL prediction
  -> evaluation artifacts and report tables
```

The current maintained production-style flow is:

```text
BTC:
  data/exports/stockmem_records.ndjson
  -> learned_recency_50_50 retriever
  -> top-10 historical evidence
  -> count_vote_buy3_sell4 head

ETH:
  data/exports/stockmem_records_eth.ndjson
  -> eth_learned_recency_50_50_h30 retriever
  -> top-10 historical evidence
  -> mean_learned_weights_buy0.50_sell0.75 head
```

The profile router is:

```text
stockmem/config/model_profiles.json
```

## 2. Data Sources

### Supabase / Database Source

| File | Role |
| --- | --- |
| `stockmem/src/store/schema.py` | Defines the StockMem storage schema. |
| `stockmem/src/store/base.py` | Repository interface used by the service. |
| `stockmem/src/store/pg_repository.py` | PostgreSQL/Supabase-backed repository implementation. |
| `stockmem/src/store/reader.py` | Read helpers for stored records. |
| `stockmem/src/store/writer.py` | Persists records and updates the in-memory cache/index. |
| `scripts/archive/pull_stockmem_records_from_supabase.py` | Pulls StockMem records from Supabase into local NDJSON for reproducible evaluation. |

The clean submission branch expects local exported datasets rather than live
database access:

```text
data/exports/stockmem_records.ndjson
data/exports/stockmem_records_eth.ndjson
```

These files are not committed. They are runtime/reproduction inputs.

### StockMem Record Shape

| File | Role |
| --- | --- |
| `stockmem/src/models.py` | Defines `StockMemRecord`, `SimilarRecord`, market snapshot fields, and typed model structure. |

Each record contains:

```text
date, symbol, market snapshot, event/factor fields,
event_vec, factor_vec, indicator_vec, price_vec,
future_return_1d, future_return_3d, future_return_7d,
future_return_15d, future_return_30d
```

The target label for current experiments is:

```text
BUY  if future_return_7d > +2%
SELL if future_return_7d < -2%
HOLD otherwise
```

## 3. Vector Construction

StockMem uses structured vector blocks rather than a single free-text prompt.

| File | Role |
| --- | --- |
| `stockmem/src/search/embedder.py` | Converts a `StockMemRecord` into normalized vector blocks and a joint vector. |
| `stockmem/src/search/taxonomy.py` | Defines taxonomy/factor conventions used by event/factor features. |
| `stockmem/src/search/event_memory.py` | Builds daily event state from current record plus historical context when missing. |
| `stockmem/scripts/regen_optimizer_data.py` | Regenerates optimizer-ready vectorized data from StockMem records. |
| `stockmem/scripts/build_cem_dataset.py` | Builds older CEM/RAG experiment datasets. Kept for audit and legacy comparison. |
| `stockmem/scripts/cem_dataset.py` | Shared dataset helpers for older CEM/RAG experiments. |

The main vector blocks are:

| Block | Meaning | Typical role |
| --- | --- | --- |
| `event_vec` | Encoded event/news memory. | Used by learned retriever artifacts when available. |
| `factor_vec` | Factor/event taxonomy and market context. | Main fixed-kNN semantic/context block. |
| `indicator_vec` | Compact indicators/sentiment-style features. | Lightweight technical/context signal. |
| `price_vec` | Recent price, range, and volume dynamics. | Market-state similarity and trend behavior. |

## 4. Runtime Service Flow

Runtime service entry points:

| File | Role |
| --- | --- |
| `stockmem/src/service.py` | Main `StockMemService`: loads records, builds cache/index, saves records, searches evidence, updates returns, retrains fixed weights. |
| `stockmem/src/api.py` | HTTP/API wrapper around the StockMem service. |
| `stockmem/src/config.py` | Loads weights and learned retriever config. |
| `stockmem/config.yaml` | Runtime service configuration. |
| `stockmem/.env.example` | Environment variable example for service/database setup. |

Runtime startup:

```text
StockMemService.startup()
  -> repository.init()
  -> load learned retriever artifact if configured
  -> repository.list_all()
  -> cache records by id
  -> RecordEmbedder.rebuild_corpus()
  -> MemoryVectorIndex.rebuild()
  -> RecordSearcher ready
```

Search request:

```text
StockMemService.search(query, k, before_date, retriever_type)
  -> ensure query has event_state
  -> RecordSearcher.search()
  -> return SimilarRecord list
```

Important leakage control:

```text
before_date should exclude records on/after the query date.
offline evaluators additionally require candidate_date + 7 days <= query_date
so candidate D7 labels are matured before use.
```

## 5. Vector Index And Search

| File | Role |
| --- | --- |
| `stockmem/src/search/index.py` | In-memory vector index for preselecting candidates. |
| `stockmem/src/search/searcher.py` | Fixed and learned search logic, same-symbol filtering, date filtering, regime adjustment, output formatting. |
| `stockmem/src/search/learned_metric.py` | Loads and applies learned diagonal metric artifacts. |

### Fixed kNN

Config:

```text
stockmem/config/weights.auto.json
stockmem/config/weights.eth.auto.json
```

Fixed kNN computes:

```text
score(q,c) =
  w_factor    * cos(factor_q, factor_c)
+ w_indicator * cos(indicator_q, indicator_c)
+ w_price     * cos(price_q, price_c)
```

BTC current fixed weights:

```text
w_factor    = 0.5443920554
w_indicator = 0.3090805325
w_price     = 0.1415662727
```

ETH tuned fixed weights:

```text
w_factor    = 0.3488
w_indicator = 0.2885
w_price     = 0.3627
```

### Learned Retriever

Config/artifacts:

```text
stockmem/config/learned_retriever_finbert.json
stockmem/config/learned_retriever_finbert.eth.json
```

The learned retriever applies a learned diagonal metric:

```text
score(q,c) = sum_b alpha_b * cos(D_b q_b, D_b c_b)
```

where:

```text
D_b      = learned per-dimension weights for block b
alpha_b  = learned block scale
```

The learned retriever is deterministic after the artifact is written. Repeating
the same query against the same candidate pool and artifact returns the same
ranking.

### Regime Adjustment

`stockmem/src/search/searcher.py` also has a lightweight regime bonus/penalty:

```text
same regime:     +0.15
opposite regime: -0.15
```

This is used in runtime search. The maintained offline majority-consensus
retriever uses explicit fusion configs described below.

## 6. Maintained Evidence Retriever

The maintained evidence retriever is not pure fixed kNN and not pure learned
retrieval. It fuses learned memory with recency:

```text
score(q,c) =
  w_learned * learned_similarity(q,c)
+ w_recency * exp(-age_days(q,c) / half_life_days)
```

BTC config:

```text
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

BTC formula:

```text
0.5 * learned_similarity + 0.5 * recency(half_life=21d)
```

ETH config:

```text
stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json
```

ETH formula:

```text
0.5 * ETH learned_similarity + 0.5 * recency(half_life=30d)
```

Diagnostic ETH tuned fusion:

```text
stockmem/config/majority_consensus_retriever.eth.tuned_eth_weights.json
```

This diagnostic config includes fixed and regime terms:

```text
w_fixed=0.2, w_learned=0.3, w_recency=0.4, w_regime=0.1
```

It has strong evidence `Majority@10`, but the maintained ETH endpoint keeps the
cleaner learned+recency design.

## 7. Decision Heads

The retriever returns top-k historical evidence. The decision head converts
that evidence into a current D7 prediction.

### BTC Maintained Head

Config is selected by:

```text
stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py
```

Maintained BTC head:

```text
count_vote_buy3_sell4
```

Rule:

```text
SELL if sell_count >= 4 and sell_count >= buy_count
BUY  if buy_count  >= 3 and buy_count  >  sell_count
HOLD otherwise
```

Input evidence:

```text
top-10 records from learned_recency_50_50
```

### ETH Maintained Head

Maintained ETH head:

```text
mean_learned_weights_buy0.50_sell0.75
```

Input evidence:

```text
top-10 records from eth_learned_recency_50_50_h30
```

The ETH head was validation-selected after ETH-specific learned retriever
fine-tuning.

### Older Strict Heads

| File | Role |
| --- | --- |
| `stockmem/config/knn_head.fixed_knn_rolling_stable.json` | Older fixed-kNN rolling-stable head. |
| `stockmem/config/knn_head.learned_finbert_rolling_stable.json` | Older learned FinBERT rolling-stable head. |

These are still useful as baselines, but they are not the maintained endpoint
heads.

## 8. Offline Evaluation Flow

Most reported results come from NDJSON offline evaluation rather than the live
service. This keeps evaluation reproducible and avoids database drift.

Shared evaluator helpers:

| File | Role |
| --- | --- |
| `stockmem/scripts/ndjson_eval_common.py` | Shared split dates, NDJSON loading, matured-pool logic, D7 labeling, fixed-kNN retrieval, metrics, and prediction helpers. |

Official split:

```text
train:      through 2024-12-24
validation: 2025-01-01 to 2025-06-23
test:       2025-07-01 to 2026-05-01
```

Matured-pool rule:

```text
candidate.date < query.date
candidate.date + 7 days <= query.date
```

This prevents leakage from using a historical record whose D7 result would not
yet be known at query time.

## 9. Training And Tuning Scripts

| Script | Purpose |
| --- | --- |
| `stockmem/scripts/optimize_weights.py` | Tunes fixed-kNN block weights. |
| `stockmem/scripts/train_learned_retriever.py` | Trains learned diagonal retriever on optimizer data. |
| `stockmem/scripts/retrain_finbert_retriever.py` | Retrains/fine-tunes learned retriever artifacts from NDJSON records; used for ETH fine-tuning. |
| `stockmem/scripts/experimental/train_majority_consensus_retriever.py` | Tunes fusion weights for majority-consensus retrieval. |
| `stockmem/scripts/experimental/train_downside_sensitive_head.py` | Experimental downside-sensitive head training. |
| `stockmem/scripts/experimental/train_head_aligned_retriever.py` | Experimental head-aligned retriever training; kept as negative/audit result. |
| `stockmem/src/weights_retrainer.py` | Runtime helper for auto-retraining fixed weights from stored records. |

Training outputs:

| Output | Meaning |
| --- | --- |
| `stockmem/config/weights.auto.json` | BTC fixed-kNN weight artifact. |
| `stockmem/config/weights.eth.auto.json` | ETH fixed-kNN diagnostic weight artifact. |
| `stockmem/config/learned_retriever_finbert.json` | BTC learned retriever artifact. |
| `stockmem/config/learned_retriever_finbert.eth.json` | ETH learned retriever artifact. |
| `stockmem/config/majority_consensus_retriever*.json` | Fusion/retriever profile artifacts. |

## 10. Evaluation Scripts

| Script | Purpose | Main output |
| --- | --- | --- |
| `aihub/scripts/evaluate_naive_llm_baseline.py` | Naive current-context LLM baseline; no historical retrieval evidence. | `artifacts/current_context_ai_eval/summary.json` |
| `stockmem/scripts/evaluate_stockmem_strict_models.py` | Strict fixed/learned retriever and head comparison. | `artifacts/learned_strict_test_v3/summary.json` |
| `stockmem/scripts/evaluate_stockmem_feature_ablation.py` | Fixed-kNN feature-block ablation. | `artifacts/fixed_knn_component_ablation/summary.json` |
| `stockmem/scripts/experimental/evaluate_majority_consensus_retrievers.py` | Evaluates `Majority@10` on val/test/full history. | `artifacts/majority_consensus_retriever_eval_*/summary.json` |
| `stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py` | Searches decision heads over top-10 evidence. | `artifacts/*consensus_heads*/summary.json` |
| `stockmem/scripts/experimental/evaluate_hybrid_retrieval.py` | Two-stage hybrid reranking audit. | `artifacts/hybrid_retrieval_frozen_v2/d7_consistency_eval.md` |
| `stockmem/scripts/experimental/evaluate_structured_models_full_history.py` | Exploratory full-history structured model audit. | `artifacts/audit_runs/*/full_history_structured_models/summary.json` |
| `stockmem/scripts/experimental/run_stockmem_audit.py` | Runs grouped metric/HOLD policy audits. | `artifacts/audit_runs/*/metric_audit.md` |
| `stockmem/scripts/export_stockmem_report_tables.py` | Exports compact report tables. | `submission/*/tables/*.md`, `*.csv` |
| `stockmem/scripts/run_submission_reproduction.py` | End-to-end reproduction orchestrator. | `submission/stockmem_*/manifest.json` |

## 11. Report And Artifact Flow

Evaluation outputs are written under:

```text
artifacts/
submission/
results_tables/
```

These folders are audit/reproduction outputs. Most are not committed by
default.

Maintained report files:

| File | Purpose |
| --- | --- |
| `docs/stockmem/methodology.md` | Explains current StockMem method and model design. |
| `docs/stockmem/experiments.md` | Main experiment narrative and maintained result interpretation. |
| `docs/stockmem/trend_aware_retrieval.md` | Focused retrieval audit for learned+recency evidence. |
| `docs/stockmem/experiment_metrics_catalog.md` | Numeric catalog of experiments and artifacts. |
| `docs/stockmem/multi_asset_stockmem_report.md` | BTC/ETH multi-asset report. |
| `docs/stockmem/eth_zero_shot.md` | ETH zero-shot transfer report. |
| `docs/stockmem/eth_finetune.md` | ETH fine-tune report. |
| `docs/stockmem/reproducibility.md` | Docker commands and expected outputs. |
| `docs/stockmem/academic_paper.md` | Long academic-style paper draft. |

## 12. Endpoint Profile Flow

The endpoint should not hard-code one global model for all assets. It should
read:

```text
stockmem/config/model_profiles.json
```

Conceptually:

```text
request symbol
  -> load asset profile from model_profiles.json
  -> load dataset/source for that asset
  -> load learned retriever artifact
  -> load consensus retriever config
  -> retrieve top-10 evidence
  -> apply selected decision head
  -> return signal + evidence rows + component scores
```

BTC profile:

```text
learned artifact:
  stockmem/config/learned_retriever_finbert.json
retriever config:
  stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
decision head:
  count_vote_buy3_sell4
```

ETH profile:

```text
learned artifact:
  stockmem/config/learned_retriever_finbert.eth.json
retriever config:
  stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json
decision head:
  mean_learned_weights_buy0.50_sell0.75
```

## 13. How To Follow A Single Prediction

For one query date:

```text
1. Load the query StockMemRecord.
2. Assign D7 label only for evaluation; do not use it for prediction.
3. Build candidate pool from same-symbol historical records.
4. Remove candidates whose D7 return is not matured at query time.
5. Compute learned_similarity(q,c).
6. Compute recency(q,c).
7. Fuse scores using the asset retriever config.
8. Sort candidates by fused score.
9. Keep top-10 as evidence.
10. Read each evidence record's matured future_return_7d.
11. Apply the asset decision head.
12. Compare predicted class with query's actual D7 class for evaluation.
13. Write prediction row and aggregate summary metrics.
```

This is the core StockMem mechanism: the model does not ask an LLM to reason
from raw current context. It uses structured historical evidence and a
validation-selected head.

## 14. What To Cite In Reports

Use these files as source of truth:

```text
docs/stockmem/methodology.md
docs/stockmem/experiments.md
docs/stockmem/experiment_metrics_catalog.md
docs/stockmem/multi_asset_stockmem_report.md
docs/stockmem/reproducibility.md
docs/references.md
```

Use these only as audit trail unless the report explicitly discusses older
experiments:

```text
docs/archive/
scripts/archive/
artifacts/metrics/
artifacts/audit_runs/
```
