# Learned Retriever FinBERT vs kNN Baselines — Report

**Date:** 2026-06-27  
**Project:** MarketLens / StockMem  
**Scope:** FinBERT-based retriever retraining, metric fixes, and comparison against current kNN baselines

---

## 1. Objective

Mục tiêu của đợt làm này là:

1. Chuyển pipeline retriever sang dùng **FinBERT sentiment** thay cho `sentiment_score` cũ trong phần indicator embedding.
2. Retrain learned retriever trên dữ liệu `stockmem_records` mới nhất.
3. Bổ sung metric đánh giá tốt hơn cho retriever:
   - `val_ndcg@k`
   - `val_hybrid`
   - configurable checkpoint selection
4. Sửa lỗi metric `NDCG > 1.0`.
5. So sánh learned retriever với:
   - fixed-kNN retriever baseline
   - `knn_returns` production baseline trong docs

---

## 2. What Was Done

### 2.1 Data and embedding pipeline

Đã cập nhật model và embedder để hỗ trợ trường:

- `finbert_sentiment_score`

Các file liên quan:

- `shared/models/memory.py`
- `stockmem/src/models.py`
- `stockmem/src/search/embedder.py`
- `stockmem/tests/test_vectorize.py`

Thay đổi chính:

- `RecordEmbedder` giờ hỗ trợ `sentiment_source`:
  - `sentiment_score`
  - `finbert`
  - `auto`
- `regen_optimizer_data.py` có thể build dataset từ:
  - PostgreSQL
  - NDJSON export local
- indicator vector có thể dùng FinBERT score thay cho sentiment cũ

### 2.2 Retraining pipeline

Đã thêm wrapper retrain:

- `stockmem/scripts/retrain_finbert_retriever.py`

Mục đích:

- rebuild optimizer dataset từ NDJSON
- train learned retriever bằng FinBERT sentiment
- hỗ trợ:
  - `--selection-metric`
  - `--init-artifact`
  - `--skip-optuna`
  - teacher relevance weights

### 2.3 Retriever training logic

Đã mở rộng trainer:

- `stockmem/scripts/train_learned_retriever.py`
- `stockmem/scripts/cem_dataset.py`
- `stockmem/tests/test_learned_retriever.py`

Các thay đổi chính:

1. Teacher relevance có weight config:
   - outcome
   - regime
   - surface

2. Bổ sung metric:
   - `ndcg_at_k`
   - `hybrid_selection_score`

3. Trainer hỗ trợ:
   - `selection_metric = hit | combined | ndcg | hybrid`
   - per-epoch CSV history
   - warm start từ artifact cũ
   - Optuna objective đồng bộ với metric selection

4. Sửa lỗi NDCG:
   - trước đó `val_ndcg@k` có thể > `1.0`
   - sau khi sửa, metric đã về đúng range `[0, 1]`

### 2.4 Evaluation tooling

Đã thêm 2 script để so sánh rõ hơn:

- `stockmem/scripts/compare_signal_accuracy.py`
  - so sánh strict `UP/HOLD/DOWN` trên test split canonical của retriever

- `scripts/compare_knn_returns_retrievers.py`
  - so sánh **apples-to-apples** giữa:
    - fixed-kNN retriever
    - learned retriever
  - nhưng dùng **cùng head `knn_returns`** như production/docs

---

## 3. Training Runs and Results

### 3.1 Full FinBERT retrain with hybrid selection

Run config:

- `trials = 10`
- `epochs = 40`
- `seeds = 5`
- `selection_metric = hybrid`

Artifact produced:

- `stockmem/config/learned_retriever_finbert.json`

Initial full-run result:

- `val_combined = 0.3617`
- `val_hit_at_k = 0.9855`
- `val_ndcg_at_k = 1.1909`  ← invalid
- `seed_std = 0.0170`

Nhận xét:

- `combined` và `hit@k` tốt
- nhưng `ndcg` sai vì vượt `1.0`
- do đó run này usable để xem xu hướng, nhưng **không đủ sạch để dùng làm final selection**

### 3.2 Warm-start rerun after NDCG fix

Run config:

- `epochs = 20`
- `seeds = 5`
- `skip_optuna = true`
- `init_artifact = learned_retriever_finbert.json`

Result:

- `val_combined = 0.3696`
- `val_hit_at_k = 0.9442`
- `val_ndcg_at_k = 0.3232`
- `seed_std = 0.0315`

Nhận xét:

- `ndcg` đã đúng range
- objective `hybrid` giờ có nghĩa thật
- nhưng vì run này dùng default params + warm start, đây là:
  - **validation run sạch**
  - không phải full best-search run

---

## 4. Comparison Results

## 4.1 Strict retriever-level test: `UP/HOLD/DOWN`

Script:

- `stockmem/scripts/compare_signal_accuracy.py`

Setup:

- dataset: `stockmem/data/real_optimizer_finbert.json`
- artifact: `stockmem/config/learned_retriever_finbert.json`
- split: `test`
- thresholds: `±2%`

Result:

| Model | Overall acc | Action acc (`UP/DOWN` only) | Coverage |
|---|---:|---:|---:|
| fixed-kNN baseline | 28.2% | 33.2% | 80.0% |
| learned FinBERT retriever | **33.8%** | **37.8%** | 78.0% |

Conclusion:

- learned retriever **beats fixed-kNN retriever**
- strongest gains:
  - better `DOWN`
  - better `HOLD`
- `UP` performance roughly similar

Confusion summary:

### fixed-kNN

```text
actual\pred   UP   HOLD   DOWN
UP            57    24      17
HOLD          39     5      14
DOWN          93    32      24
```

### learned FinBERT

```text
actual\pred   UP   HOLD   DOWN
UP            57    17      24
HOLD          37    13       8
DOWN          79    37      33
```

---

## 4.2 Apples-to-apples test with `knn_returns` decision head

Script:

- `scripts/compare_knn_returns_retrievers.py`

This is the most important comparison, because production/docs are not using
raw retriever averaging. They use:

1. top-k retrieval
2. multi-horizon weighted return head
3. threshold mapping to BUY/HOLD/SELL

So here the head is fixed, and only the retriever changes.

### A. Using exact doc search weights

Search weights:

- factor = `0.544392055430515`
- indicator = `0.30908053253948164`
- price = `0.14156627274414413`

Return weights:

- `1d = 0.40`
- `3d = 0.30`
- `7d = 0.15`
- `15d = 0.10`
- `30d = 0.05`

Thresholds:

- buy = `+2%`
- sell = `-2%`

Results:

| Model | Coverage | BUY DA | SELL DA | HOLD DA | Overall DA | Active acc |
|---|---:|---:|---:|---:|---:|---:|
| baseline `knn_returns` head | **56.7%** | **56.7%** | 49.3% | 11.9% | **35.9%** | **54.3%** |
| learned retriever + same head | 54.7% | 55.1% | **51.2%** | **12.5%** | 34.9% | 53.5% |

Conclusion:

- learned retriever improves:
  - SELL DA
  - HOLD DA
- but loses more on BUY DA
- net result: **still below baseline**

### B. Using current `weights.auto.json` in repo

Search weights:

- factor = `0.6846920901629276`
- indicator = `0.19261347699748183`
- price = `0.12269443283959038`

Results:

| Model | Coverage | BUY DA | SELL DA | HOLD DA | Overall DA | Active acc |
|---|---:|---:|---:|---:|---:|---:|
| baseline `knn_returns` head | **57.5%** | **57.1%** | 51.2% | 11.6% | **36.7%** | **55.3%** |
| learned retriever + same head | 54.7% | 55.1% | 51.2% | **12.5%** | 34.9% | 53.5% |

Conclusion:

- learned retriever still **does not beat** the `knn_returns` baseline

---

## 5. Why It Does Not Beat the Production Baseline Yet

### 5.1 Current retriever objective is misaligned

Current learned retriever is trained to optimize retriever-level proxies:

- hit@k
- combined
- ndcg / hybrid
- teacher relevance from similarity/outcome/regime

But production is judged by:

- BUY DA
- SELL DA
- HOLD DA
- coverage
- overall decision quality after `knn_returns` downstream head

This means:

- retriever may improve semantic match
- but not necessarily improve final trading decision

### 5.2 Main observed pattern

In both apples-to-apples tests:

- learned retriever becomes less bullish
- catches more `DOWN` and `HOLD`
- but misses enough `UP` that total score drops

So right now it is learning a different balance than the production head needs.

---

## 6. Recommended Next Step

The correct next optimization target is **not just retriever quality**.
It is:

> optimize the retriever for the downstream `knn_returns` decision objective

### Proposed immediate next phase

1. Keep the learned artifact fixed.
2. Optimize the downstream head on top of it:
   - `k`
   - return weights `1d/3d/7d/15d/30d`
   - buy threshold
   - sell threshold
3. Score by:
   - `BUY_DA`
   - `SELL_DA`
   - `overall_DA`
   - `coverage`

This is the cheapest test of whether the learned retriever contains useful signal
that is currently being underused by the old decision head.

### After that

If learned + re-optimized head still loses to baseline:

- retraining objective itself must be changed
- trainer should select checkpoints by downstream `knn_returns` validation score
- positive/negative mining should be tied to downstream signal contribution, not only semantic similarity

---

## 7. Reproduction Commands

### Full FinBERT retrain

```bash
python3 stockmem/scripts/retrain_finbert_retriever.py \
  --input-ndjson data/exports/stockmem_records.ndjson \
  --dataset-output stockmem/data/real_optimizer_finbert.json \
  --artifact-output stockmem/config/learned_retriever_finbert.json \
  --trials 10 \
  --epochs 40 \
  --seeds 5 \
  --selection-metric hybrid
```

### Strict `UP/HOLD/DOWN` test

```bash
python stockmem/scripts/compare_signal_accuracy.py \
  --data stockmem/data/real_optimizer_finbert.json \
  --artifact stockmem/config/learned_retriever_finbert.json \
  --weights stockmem/config/weights.auto.json \
  --split test \
  --buy-threshold 2 \
  --sell-threshold 2
```

### Apples-to-apples `knn_returns` head comparison

```bash
python scripts/compare_knn_returns_retrievers.py \
  --input-ndjson data/exports/stockmem_records.ndjson \
  --artifact stockmem/config/learned_retriever_finbert.json \
  --weights stockmem/config/weights.auto.json \
  --start-date 2022-01-01 \
  --k 5 \
  --buy-thr 2 \
  --sell-thr 2
```

---

## 8. Final Summary

### What succeeded

- FinBERT sentiment was integrated into the StockMem retraining pipeline.
- Trainer now supports:
  - NDCG
  - hybrid checkpoint selection
  - teacher weighting
  - warm start
  - history logging
- NDCG bug was fixed.
- Learned retriever clearly beats fixed-kNN retriever on strict test-set classification.

### What did not succeed yet

- Learned retriever **does not yet beat** the current `knn_returns` production baseline.
- Therefore, replacing the production baseline now would be premature.

### Practical decision

- Keep current learned retriever work as a valid baseline and training infrastructure upgrade.
- Next work should focus on **optimizing the downstream head** before changing retriever training objective.
