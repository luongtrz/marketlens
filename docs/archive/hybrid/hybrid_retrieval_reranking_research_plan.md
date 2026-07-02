# Hybrid Retrieval and Reranking Research Plan for StockMem

**Date:** 2026-06-29  
**Project:** MarketLens / StockMem  
**Scope:** Reframe the next retrieval experiment as two-stage retrieval and reranking for D7-consistent historical evidence

---

## 1. Motivation

Current repo results show a real split between retrieval-level gains and trading-level robustness.

- In [docs/learned_retriever_finbert_vs_knn_report_2026-06-27.md](./learned_retriever_finbert_vs_knn_report_2026-06-27.md), the learned FinBERT retriever beats the fixed-kNN baseline on strict retriever metrics at `±2%` D7 thresholds:
  - overall accuracy: `33.8%` vs `28.2%`
  - action accuracy: `37.8%` vs `33.2%`
  - coverage: `78.0%` vs `80.0%`
- In [docs/learned_finbert_candidate_backtest_2026-06-27.md](./learned_finbert_candidate_backtest_2026-06-27.md), learned retrieval improves some downstream classification metrics and coverage, but fixed kNN remains strong on return-oriented proxy metrics. In the long-horizon `2022-01-01 -> 2026-06-21` comparison:
  - learned candidate has better `overall_DA` (`47.6%` vs `46.0%`)
  - fixed stable control still has better `avg_ret7d` (`2.43%` vs `1.90%`)
  - production baseline remains slightly better on `active_acc` (`56.0%` vs `55.4%`)

This creates an academic problem if the next step is framed as a manual choice between fixed kNN and learned retrieval. Fixed manual weights are still heuristic unless tuned on a proper validation protocol. That is difficult to defend as a research contribution.

The stronger framing is:

> fixed kNN is a strong market-state candidate generator, while learned FinBERT or event-aware retrieval is a useful reranker for evidence relevance.

The next experiment should therefore test a **two-stage hybrid retriever** whose goal is not immediate PnL maximization, but better retrieval of historical records whose `future_return_7d` direction is consistent with the query's D7 outcome.

This framing is better aligned with prior work:

- convex score fusion is a standard and sample-efficient way to combine complementary retrieval signals
- nearest-neighbor quality can improve when the distance function is learned rather than assumed
- reranking is naturally evaluated with ranking metrics such as nDCG and pairwise ranking objectives
- time-aware validation is required because random splits can leak future market structure

---

## 2. Research Question

### Main question

**Does hybrid reranking over kNN candidates improve D7 signal consistency of retrieved historical records?**

### Retrieval target

For each query day, the top historical evidence records should preferentially share the same D7 direction as the query.

### D7 class definition

Using the current default threshold of `±2%`, represented in the repo scripts as `±2.0` percentage points:

- `UP` if `future_return_7d > +2.0`
- `DOWN` if `future_return_7d < -2.0`
- `HOLD` otherwise

The primary target is therefore not "best trading return" first. The immediate target is **higher-quality historical evidence retrieval** under a fixed D7 labeling rule.

---

## 3. Proposed Method

### Stage 1: candidate generation

Retrieve the top-30 historical candidates using the current fixed kNN search over the existing market-state vectors:

- factor vector
- indicator vector
- price vector

This preserves the strongest property of the current baseline: stable retrieval of similar market states.

### Stage 2: reranking

Rerank only those 30 candidates with a convex fusion model:

```text
score(q,c) =
  w_knn * knn_market_score(q,c)
  + w_learned * learned_finbert_score(q,c)
  + w_regime * regime_score(q,c)
  + w_prior * signal_prior_score(q,c)
```

Where:

- `knn_market_score(q,c)` is the current fixed-kNN similarity score
- `learned_finbert_score(q,c)` is the learned retrieval score, initialized from `stockmem/config/learned_retriever_finbert.json`
- `regime_score(q,c)` measures whether query and candidate share similar regime conditions
- `signal_prior_score(q,c)` expresses whether the candidate historically belongs to a useful D7 class prior for the query context

### Constraints

All fusion weights must satisfy:

- `w_knn >= 0`
- `w_learned >= 0`
- `w_regime >= 0`
- `w_prior >= 0`
- `w_knn + w_learned + w_regime + w_prior = 1`

### Initial search grid

- `w_knn ∈ {0.3, 0.4, 0.5, 0.6}`
- `w_learned ∈ {0.1, 0.2, 0.3, 0.4}`
- `w_regime ∈ {0.0, 0.1, 0.2}`
- `w_prior ∈ {0.0, 0.1, 0.2}`

Keep only combinations whose weights sum to `1`.

### Final evidence set

After reranking, keep the final top-5 evidence records. These top-5 records are the object of the main evaluation.

---

## 4. Training Protocol

### Split design

Use rolling time-series validation. Do not use random train/validation/test splits.

Candidate protocol:

- train window: `36 months`
- validation window: `6 months`
- test window: `3 months`
- step size: `3 months`
- embargo: `7 days`

The embargo prevents near-overlap leakage between a query and adjacent future-labeled records.

### Fold procedure

For each rolling fold:

1. Freeze the candidate generator.
2. Tune fusion weights only on the validation window.
3. Select the fold-best reranker under the validation metric.
4. Do not touch the fold test window until the validation choice is frozen.

### Shared stable configuration

After evaluating all folds, choose one shared stable configuration by maximizing:

```text
mean_validation_score - variance_penalty
```

This avoids reporting a lucky weight vector that only works in one regime slice.

### Primary optimization metric

Use:

```text
selection_score = 0.5 * Hit@5_same_D7_sign + 0.3 * nDCG@5 + 0.2 * downstream_DA
```

Interpretation:

- `Hit@5_same_D7_sign`: measures whether the top-5 evidence set contains D7-consistent records
- `nDCG@5`: measures whether more relevant candidates are ranked near the top
- `downstream_DA`: checks that retrieval improvements still transfer to decision quality

### Final holdout rule

Keep the final test period completely untouched until the shared stable configuration is selected. The final test result is reported once, after model selection is done.

---

## 5. Expected Small Steps

### Step 1: add D7-consistency evaluator

Implement an evaluator for `top5_same_D7_sign_rate`.

Expected output:

- baseline numbers for fixed kNN
- baseline numbers for learned-only retrieval
- baseline numbers for the current production head

### Step 2: add hybrid reranker

Implement hybrid reranking over the kNN top-30 pool.

Expected output:

- per-query top-5 evidence
- component scores for `knn`, `learned`, `regime`, and `prior`

### Step 3: add rolling validation tuner

Tune fusion weights with rolling validation.

Expected output:

- one stable selected weight configuration
- fold-level validation table

### Step 4: run final test comparison

Run the untouched test comparison for:

- fixed kNN
- learned-only
- hybrid reranker

Expected output:

- one comparison table on the final test folds

### Step 5: write final research report

Write the final report as a methodology-first result, including:

- main comparison table
- ablations
- failure cases
- citations

---

## 6. Training Target

The training target is not "best trading PnL" first. The immediate academic target is better historical evidence retrieval.

### Minimum success target

- hybrid reranker improves `Hit@5_same_D7_sign` over fixed kNN by at least `2` percentage points on rolling test folds
- hybrid reranker improves or matches `nDCG@5`
- hybrid reranker does not reduce downstream `active_acc` by more than `1` percentage point
- evidence coverage remains at least `75%`

### Secondary target

- improve downstream `overall_DA` or `active_acc` versus learned-only
- keep return-based metrics as analysis, not as the main claim

---

## 7. Test Plan

### Focused tests

- unit test D7 class labeling around the threshold boundaries
- unit test fusion score normalization and validation of `sum(weights) = 1`
- unit test that the reranker returns candidates only from the kNN top-N pool
- regression test that fixed-kNN baseline results are unchanged
- integration test on a small fixture to verify:
  - `Hit@5_same_D7_sign`
  - `nDCG@5`
  - coverage
  - downstream DA

### Acceptance scenarios

- running the evaluator produces one JSON artifact and one Markdown summary
- the summary includes baseline, learned-only, and hybrid rows
- the selected hybrid weights are reported as validation-selected hyperparameters, not manual constants

---

## 8. Assumptions

- default candidate pool size is top-30 from fixed kNN
- default evidence size is top-5
- default D7 threshold is `±2%`, matching current comparison scripts and reports
- the first hybrid experiment uses `stockmem/config/learned_retriever_finbert.json` as the learned score source
- implementation should proceed in this order:
  1. create this document
  2. implement the evaluator
  3. implement the hybrid reranker
  4. implement rolling validation and final comparison

---

## 9. Why This Is Academically Defensible

This experiment is stronger than a manual "kNN vs learned retriever" comparison for four reasons.

1. It treats kNN and learned retrieval as complementary signals rather than mutually exclusive systems.
2. It moves the fusion choice from heuristic manual weights to validation-selected hyperparameters.
3. It uses ranking metrics that directly match the evidence-selection problem.
4. It respects temporal dependence through rolling validation and an embargo.

In short, the claim becomes:

> a hybrid reranker over market-state kNN candidates improves retrieval of D7-consistent historical evidence.

That is a narrower claim than "this model makes more money," but it is much easier to justify empirically and methodologically.

---

## 10. References

1. Sebastian Bruch, Siyu Gai, and Amir Ingber. *An Analysis of Fusion Functions for Hybrid Retrieval*. arXiv:2210.11934. https://arxiv.org/abs/2210.11934
2. Bharath K. Sriperumbudur and Gert R. G. Lanckriet. *Metric Embedding for Nearest Neighbor Classification*. arXiv:0706.3499. https://arxiv.org/abs/0706.3499
3. Marius Köppel, Alexander Segner, Martin Wagener, Lukas Pensel, Andreas Karwath, and Stefan Kramer. *Pairwise Learning to Rank by Neural Networks Revisited: Reconstruction, Theoretical Analysis and Practical Performance*. arXiv:1909.02768. https://arxiv.org/abs/1909.02768
4. Sylvain Arlot and Alain Celisse. *A survey of cross-validation procedures for model selection*. arXiv:0907.4728. https://arxiv.org/abs/0907.4728
