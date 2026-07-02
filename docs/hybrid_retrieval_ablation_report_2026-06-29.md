# Hybrid Retrieval Ablation Report

**Date:** 2026-06-29  
**Project:** MarketLens / StockMem  
**Scope:** Final retrieval-side decision after hybrid reranking ablation

---

## 1. Decision

For the current StockMem retrieval task, **fixed kNN remains the best retrieval method**.

This report freezes:

- `fixed_knn` as the **retrieval baseline**
- learned-only retrieval as a **comparison baseline**
- hybrid reranking as a **research candidate**, not a retrieval replacement

The current learned and reranked variants do **not** beat fixed kNN on the main retrieval target:

- `Hit@5_same_D7_sign`

They also do not produce a clear retrieval win on the joint retrieval view of:

- `Hit@5_same_D7_sign`
- `nDCG@5`

---

## 2. Authoritative Artifacts

Use these artifacts as the final reference for this conclusion:

- corrected rolling validation:
  - `artifacts/hybrid_retrieval_tuning_full_v2/rolling_validation.json`
  - `artifacts/hybrid_retrieval_tuning_full_v2/rolling_validation.md`
- frozen-weight final evaluation:
  - `artifacts/hybrid_retrieval_frozen_v2/d7_consistency_eval.json`
  - `artifacts/hybrid_retrieval_frozen_v2/d7_consistency_eval.md`
- two-score ablations:
  - `artifacts/hybrid_retrieval_ablation_7030/d7_consistency_eval.md`
  - `artifacts/hybrid_retrieval_ablation_8020/d7_consistency_eval.md`

Note:

- the first full tuner output in `artifacts/hybrid_retrieval_tuning_full/` is **not authoritative**
- its global selector aggregated only fold-winning configs
- the corrected `v2` run aggregates every candidate weight vector across all folds

---

## 3. Corrected Stable Hybrid Weights

The corrected rolling-validation run selected:

```text
w_knn = 0.6
w_learned = 0.4
w_regime = 0.0
w_prior = 0.0
```

This is the best **stable hybrid** under the current search space and validation protocol.

Even so, it still does not beat fixed kNN on the main retrieval target.

---

## 4. Final Retrieval Comparison

Full test split, `top_k = 5`, D7 threshold `±2.0`:

| Method | Hit@5 same D7 sign | nDCG@5 | Downstream DA | Active Acc | Coverage |
|---|---:|---:|---:|---:|---:|
| **fixed_knn** | **0.9312** | 0.3011 | 0.2820 | 0.3320 | 0.8000 |
| learned_only | 0.9109 | 0.2932 | **0.3377** | **0.3782** | 0.7803 |
| hybrid_reranker `0.6/0.4` | 0.9069 | 0.3004 | **0.3377** | 0.3775 | 0.8164 |
| fixed_knn_production_head | **0.9312** | 0.3011 | 0.2852 | 0.3681 | 0.5344 |

Readout:

- fixed kNN is best on `Hit@5_same_D7_sign`
- fixed kNN is also slightly ahead of the tuned hybrid on `nDCG@5`
- learned-only and hybrid variants are better on downstream classification metrics
- therefore the learned and reranked variants are **not** better retrieval systems yet

---

## 5. Two-Score Ablation

To remove regime/prior effects, two narrower hybrid blends were tested:

| Method | Hit@5 same D7 sign | nDCG@5 | Downstream DA | Active Acc | Coverage |
|---|---:|---:|---:|---:|---:|
| fixed_knn | **0.9312** | 0.3011 | 0.2820 | 0.3320 | 0.8000 |
| learned_only | 0.9109 | 0.2932 | **0.3377** | **0.3782** | 0.7803 |
| hybrid `0.7 * knn + 0.3 * learned` | 0.9150 | **0.3038** | 0.3279 | 0.3699 | 0.8066 |
| hybrid `0.8 * knn + 0.2 * learned` | 0.9109 | 0.2998 | 0.2951 | 0.3500 | 0.7869 |

Readout:

- `0.7/0.3` is the closest challenger
- but it still trails fixed kNN on the main retrieval target:
  - `0.9150` vs `0.9312`
- `0.7/0.3` slightly improves `nDCG@5`
  - `0.3038` vs `0.3011`
- this is not enough to claim a retrieval win

So even after removing regime and prior terms, the conclusion does not change.

---

## 6. Conclusion

### Retrieval conclusion

For **historical evidence retrieval**, the current ranking is:

1. `fixed_knn`
2. `hybrid 0.7/0.3` as the nearest challenger
3. `learned_only`
4. other hybrid variants tested so far

### Modeling conclusion

The learned signal is useful, but not yet as a retrieval replacement.

Right now it helps more as:

- a downstream decision-quality signal
- a policy candidate
- a research direction for later reranking work

It does not yet justify replacing fixed kNN as the retrieval engine.

---

## 7. Operational Recommendation

Freeze the system this way for the next research step:

- **retrieval baseline:** `fixed_knn`
- **retrieval claim:** kNN remains strongest on D7-consistent evidence retrieval
- **learned claim:** learned or hybrid methods can improve downstream decision metrics, but not retrieval quality enough yet

This is the defensible statement to carry into the next report or paper draft.
