# StockMem Retrieval Evolution Report

**Date:** 2026-06-29  
**Project:** MarketLens / StockMem  
**Scope:** End-to-end paper-style report on the evolution from fixed `kNN` retrieval to learned FinBERT retrieval and hybrid reranking

---

## Abstract

This report presents the full experimental progression of the StockMem retrieval system across three stages:

1. a fixed weighted-`kNN` retriever over market-state vectors,
2. a learned FinBERT-based retriever with a diagonal metric,
3. a hybrid reranker that applies learned scoring on top of `kNN` candidates.

The central question is whether increasingly learned retrieval mechanisms actually improve the quality of historical evidence retrieval for market decision support. In this project, evidence quality is defined operationally by whether the retrieved historical records share the same D7 return direction as the query day, where D7 direction is labeled using a `±2%` threshold on `future_return_7d`.

The empirical picture is mixed. In the earlier rolling and hybrid-retrieval studies, fixed `kNN` was often the strongest retrieval-side baseline and remained difficult to beat on several retrieval diagnostics. However, on the later strict held-out `305`-row test used for the direct structured-vs-AI comparison, the best structured model is neither plain fixed `kNN` nor full learned retrieval. The strongest result comes from a **fixed retriever plus learned stable head**, which exceeds `fixed_knn_rolling_stable` on `overall_acc`, `active_acc`, and coverage. The full `learned_finbert_rolling_stable` pipeline also beats `fixed_knn_rolling_stable`, but by a smaller margin and without strong paired significance. Hybrid reranking still does not clearly dominate the strongest structured baselines, but the strict-test evidence means the blanket claim “fixed `kNN` is best” is no longer defensible without qualification.

The resulting conclusion is methodologically important: which system looks best depends on the evaluation target and on whether improvement is coming from the retriever or the head. Fixed `kNN` remains a strong and reproducible baseline, but the learned head is currently the main source of the strict-test gain. This report documents the mechanism, motivation, experiments, results, and implications of that conclusion.

---

## 1. Introduction

StockMem is designed to retrieve historical market days that are useful analogs for the current market snapshot. The original motivation was practical: if the system can surface past periods with similar factor, indicator, and price structure, then those historical analogs can support interpretable prediction and rule-based decision making.

Over time, the retrieval layer evolved in response to three questions:

1. Is a hand-designed weighted `kNN` over market-state vectors already good enough?
2. Can a learned retriever, especially one informed by FinBERT-based sentiment structure, improve evidence quality?
3. If both contain useful signal, should the system use a hybrid reranking architecture instead of choosing one retriever outright?

Those three questions produced three distinct experiments:

- **Experiment I:** fixed `kNN` retrieval and `kNN-returns` decision head
- **Experiment II:** learned FinBERT retriever, evaluated both as a pure retriever and as a candidate generator for the same decision head
- **Experiment III:** hybrid reranking over `kNN` candidates, evaluated with time-aware rolling validation and final ablations

The main contribution of this report is not a novel model. It is a precise and defensible answer to what the current evidence supports:

> under the present data and feature design, fixed `kNN` is a strong baseline, but the strongest model depends on the evaluation regime and on the decision head: a fixed retriever plus learned stable head is currently best on the strict held-out classifier test, while fixed `kNN` remains competitive and often stronger in earlier rolling or retrieval-focused protocols.

---

## 2. Problem Formulation

### 2.1 Retrieval target

For a query day `q`, the system retrieves historical days `c` that are intended to serve as explanatory or predictive evidence. The retrieval quality target in the final phase of this project is:

- **top-5 historical evidence should be D7-consistent with the query**

where D7 is defined from `future_return_7d`.

### 2.2 D7 labels

Throughout the final retrieval experiments, D7 labels use the same thresholding rule as the current evaluation scripts:

- `UP` if `future_return_7d > +2.0`
- `DOWN` if `future_return_7d < -2.0`
- `HOLD` otherwise

This threshold is represented as percentage points in the codebase, not decimal fractions.

### 2.3 Why retrieval and downstream evaluation are separate

One recurring lesson in this project is that retrieval quality and downstream trading or classification quality are related but not identical.

- A retriever can surface highly D7-consistent historical evidence yet still feed a weak downstream policy.
- A retriever can slightly worsen evidence purity but improve downstream decision quality after the head aggregates more information.

This is why the report keeps two evaluation layers separate:

1. **retrieval-side metrics**
   - `Hit@5_same_D7_sign`
   - `nDCG@5`
2. **downstream decision metrics**
   - `overall_DA`
   - `active_acc`
   - coverage

That separation is essential to understanding the final result.

---

## 3. Representation and Retrieval Mechanisms

### 3.1 StockMem feature blocks

The retrieval stack is built on three core market-state blocks:

| Block | Dimensionality | Role |
|---|---:|---|
| `factor_vec` | 75 | event taxonomy and factor structure |
| `indicator_vec` | 5 | normalized market indicators including sentiment-related signals |
| `price_vec` | 60 | recent price, range, and volume dynamics |

The fixed `kNN` system computes weighted cosine similarity over these blocks. The learned retriever uses the same blocks, but replaces the fixed weighting rule with a learned diagonal metric. When available, event vectors are also used in learned scoring paths.

### 3.2 Fixed `kNN`

The original `kNN` retrieval score is:

```text
sim_fixed(q,c) =
  w_factor * cos(factor_vec_q, factor_vec_c)
  + w_indicator * cos(indicator_vec_q, indicator_vec_c)
  + w_price * cos(price_vec_q, price_vec_c)
```

The strongest tuned search weights in the `kNN-returns` baseline are:

```text
w_factor    = 0.5444
w_indicator = 0.3091
w_price     = 0.1416
```

This weighting is important. It implies the strongest fixed baseline is not a purely price-pattern retriever. It already emphasizes factor and indicator structure more than local price similarity.

### 3.3 Learned FinBERT retriever

The learned retriever keeps the same retrieval problem but changes the scoring function. Rather than assuming a fixed block-weighted cosine metric, it learns a diagonal metric over the concatenated representation. In effect, it learns which dimensions matter and how strongly each block should influence retrieval.

Conceptually, the learned score is:

```text
sim_learned(q,c) =
  Σ_b α_b · cos(D_b q_b, D_b c_b)
```

where:

- `b` indexes feature blocks,
- `α_b` are learned block scales,
- `D_b` is a learned diagonal reweighting for block `b`.

This is aligned with the literature on learned embeddings and metric adaptation for nearest-neighbor methods, where the metric itself can be more important than the base classifier or retrieval rule [Sriperumbudur and Lanckriet, 2007](https://arxiv.org/abs/0706.3499).

The practical FinBERT role in this project is to improve the semantic and sentiment-related signal available to the retriever, especially inside the indicator and event-related representation path. The intuition is that a financial language model may better capture event and sentiment correspondence than a fixed handcrafted similarity rule [Araci, 2019](https://arxiv.org/abs/1908.10063).

### 3.4 Hybrid reranking

The hybrid reranker treats fixed `kNN` and learned retrieval as complementary signals:

1. use fixed `kNN` to generate a candidate pool,
2. rerank only that pool with a convex combination of component scores.

The hybrid score is:

```text
score(q,c) =
  w_knn * knn_market_score(q,c)
  + w_learned * learned_finbert_score(q,c)
  + w_regime * regime_score(q,c)
  + w_prior * signal_prior_score(q,c)
```

with the constraints:

- all weights non-negative
- all weights sum to `1`

This choice follows the literature on hybrid retrieval fusion. Convex combination is attractive because it is interpretable, sample-efficient, and easy to tune under validation [Bruch, Gai, and Ingber, 2022](https://arxiv.org/abs/2210.11934). The reranking perspective is also consistent with pairwise and listwise ranking literature, where the problem is not merely “find neighbors,” but “place the most relevant evidence near the top” [Köppel et al., 2019](https://arxiv.org/abs/1909.02768).

---

## 4. Experimental Chronology

### 4.1 Experiment I: Fixed `kNN` as the first strong baseline

The first mature system in this line was not a learned retriever. It was a `kNN-returns` strategy:

1. retrieve top-`k=5` similar historical days,
2. aggregate multiple future-return horizons for those neighbors,
3. map the average into `BUY`, `SELL`, or `HOLD`.

The key contribution of this stage was discovering that a deterministic memory-based baseline was already strong. In the report [docs/knn_returns_strategy_report.md](/home/nmtc/projects/marketlens/docs/knn_returns_strategy_report.md), the optimized `kNN-returns` system achieved:

- `BUY_DA = 59.7%`
- `SELL_DA = 57.5%`
- coverage `= 58.2%`

on D+7 evaluation with tuned search weights and threshold `±2%`.

This result mattered because it displaced the original assumption that a language-model-first policy would necessarily be best. In practice, the fixed `kNN` system outperformed the tested LLM-based alternatives in the decision protocol used at that time.

### 4.2 Experiment II: Learned FinBERT retrieval

The second stage asked whether the retriever itself should be learned.

The changes included:

- FinBERT-derived sentiment support in the embedding pipeline
- learned diagonal metric training
- improved validation metrics including `nDCG`
- stricter learned-vs-fixed retriever comparison

This stage produced an important split in the results:

#### Retriever-only comparison

From [docs/learned_retriever_finbert_vs_knn_report_2026-06-27.md](/home/nmtc/projects/marketlens/docs/learned_retriever_finbert_vs_knn_report_2026-06-27.md), on the strict `UP/HOLD/DOWN` retriever test:

| Model | Overall accuracy | Action accuracy | Coverage |
|---|---:|---:|---:|
| fixed-kNN baseline | 28.2% | 33.2% | 80.0% |
| learned FinBERT retriever | **33.8%** | **37.8%** | 78.0% |

So learned retrieval looked clearly stronger as a raw classifier over the canonical test split.

#### Same-head comparison

However, when both retrievers were forced to use the same `knn_returns` decision head, the picture changed:

| Model | Coverage | BUY DA | SELL DA | HOLD DA | Overall DA | Active acc |
|---|---:|---:|---:|---:|---:|---:|
| baseline `knn_returns` head | **56.7%** | **56.7%** | 49.3% | 11.9% | **35.9%** | **54.3%** |
| learned retriever + same head | 54.7% | 55.1% | **51.2%** | **12.5%** | 34.9% | 53.5% |

This is the first major methodological lesson of the project:

> a learned retriever can improve retriever-side classification metrics without producing a clean win under the downstream head actually used by the system.

### 4.3 Experiment IIb: Learned candidate with stable rolling head

The project then refined the learned pipeline further by tuning the head with rolling windows and a shared stable-selection rule. This matters because time-series systems can look stronger than they are if validation is not aligned with temporal structure [Arlot and Celisse, 2009](https://arxiv.org/abs/0907.4728).

From [docs/learned_finbert_candidate_backtest_2026-06-27.md](/home/nmtc/projects/marketlens/docs/learned_finbert_candidate_backtest_2026-06-27.md), the learned candidate improved several policy-level metrics:

#### Candidate-confirmation window: 2024-01-01 to 2026-06-21

| Strategy | Coverage | Overall DA | Active Acc | Avg ret7d |
|---|---:|---:|---:|---:|
| learned candidate | **86.4%** | **48.5%** | **54.8%** | **2.96%** |
| fixed stable control | 77.4% | 44.9% | 54.4% | 2.57% |
| production baseline | 71.0% | 43.0% | 54.7% | 2.56% |

#### Long-horizon window: 2022-01-01 to 2026-06-21

| Strategy | Coverage | Overall DA | Active Acc | Avg ret7d |
|---|---:|---:|---:|---:|
| learned candidate | **83.1%** | **47.6%** | 55.4% | 1.90% |
| fixed stable control | 77.6% | 46.0% | 55.7% | **2.43%** |
| production baseline | 73.1% | 44.5% | **56.0%** | 2.35% |

This refined the interpretation:

- learned retrieval and learned candidate generation can improve **policy quality**
- but that still does not prove better **historical evidence retrieval**

That distinction directly motivated Experiment III.

### 4.4 Experiment III: Hybrid reranking for D7-consistent evidence

The third stage reframed the problem academically. Instead of asking whether fixed or learned retrieval should win outright, it asked:

> can learned scoring improve the ranking of `kNN` candidates so that the top evidence is more D7-consistent?

This stage introduced:

- top-30 `kNN` candidate generation
- hybrid reranking over the candidate pool
- explicit retrieval metrics:
  - `Hit@5_same_D7_sign`
  - `nDCG@5`
- rolling time-series validation with a 36m / 6m / 3m train-val-test protocol and 7-day embargo

The expectation was that fixed `kNN` would remain strong for market-state recall, while learned scoring might improve top-of-list relevance. This is precisely the kind of hybrid complementarity discussed in the hybrid retrieval literature [Bruch, Gai, and Ingber, 2022](https://arxiv.org/abs/2210.11934).

---

## 5. Evaluation Protocols

### 5.1 Fixed `kNN` and early learned experiments

The fixed and early learned experiments were evaluated under two main families of protocol:

1. **retriever-level classification**
   - strict `UP/HOLD/DOWN` evaluation
2. **same-head or candidate-level policy evaluation**
   - force both retrievers through the same decision head
   - measure downstream DA, active accuracy, coverage, and return proxies

These protocols were appropriate for operational comparison, but they did not isolate evidence-retrieval quality cleanly enough for the reranking question.

### 5.2 Hybrid retrieval evaluation

The hybrid phase introduced a retrieval-centered evaluation:

- candidate pool size: `30`
- final evidence size: `5`
- D7 threshold: `±2.0`

Primary retrieval metrics:

- `Hit@5_same_D7_sign`
- `nDCG@5`

Secondary downstream metrics:

- `downstream_DA`
- `active_acc`
- coverage

### 5.3 Why rolling validation was necessary

Random model selection would not have been defensible here. The data are time-ordered, the market structure changes across periods, and a retrieval configuration can overfit to one regime slice while failing badly in another. Cross-validation must respect those properties [Arlot and Celisse, 2009](https://arxiv.org/abs/0907.4728).

The corrected final rolling protocol used:

- train window: 36 months
- validation window: 6 months
- test window: 3 months
- step size: 3 months
- embargo: 7 days

This protocol produced 19 folds in the final `v2` run.

---

## 6. Results

### 6.0 Primary Strict Held-Out Classifier Test

The clearest direct comparison now available is the strict held-out classifier test on the shared `305`-row window:

- test split: `2025-07-01` to `2026-05-01`
- label threshold: `±2%` on `future_return_7d`
- source artifact: [artifacts/learned_strict_test_v3/summary.md](/home/nmtc/projects/marketlens/artifacts/learned_strict_test_v3/summary.md)

This is the primary table that should anchor the paper because all structured variants are scored on the same held-out block with the same downstream decision target.

| Model | Overall Acc | Active Acc | Coverage | Hit@5 same sign |
|---|---:|---:|---:|---:|
| fixed_knn_rolling_stable | 31.80% | 42.36% | 75.08% | 83.61% |
| **fixed_retriever_learned_head** | **35.08%** | **45.00%** | **85.25%** | 83.61% |
| learned_retriever_fixed_head | 31.48% | 41.82% | 72.13% | **84.59%** |
| learned_finbert_rolling_stable | 34.10% | 43.93% | 78.36% | **84.59%** |

This table changes the interpretation of the project in an important way:

- the strongest structured model on the strict held-out test is not plain fixed `kNN`,
- it is also not the fully learned retrieval pipeline,
- the strongest result comes from **fixed retrieval plus learned stable head**.

So the main gain on the strict classifier test is coming from the **head**, not from retriever replacement alone.

#### Paired statistical comparison

Primary paired comparison: `fixed_knn_rolling_stable` vs `fixed_retriever_learned_head`

- `overall_acc` delta: `+3.30` points; 95% bootstrap CI `[+0.00, +6.89]`
- `active_acc` delta: `+2.64` points; 95% bootstrap CI `[+0.13, +5.29]`
- `coverage` delta: `+10.14` points; 95% bootstrap CI `[+6.23, +14.10]`
- McNemar exact test: `p = 0.0872`

Secondary paired comparison: `fixed_knn_rolling_stable` vs `learned_finbert_rolling_stable`

- `overall_acc` delta: `+2.22` points; 95% bootstrap CI `[-3.93, +8.52]`
- McNemar exact test: `p = 0.5507`

Interpretation:

- the effect is directionally consistent and practically nontrivial,
- the learned-head winner is stronger than fixed stable on the main strict-test metrics,
- but the paired significance story is still borderline rather than decisive.

## 6.1 Stage-by-stage summary

The full system evolution can be summarized as follows.

| Stage | Main method | Best evidence for it | Main weakness |
|---|---|---|---|
| I | fixed `kNN` | strongest retrieval baseline and strong deterministic policy | hand-designed, not learned |
| II | learned FinBERT retriever | better raw retriever classification and stronger policy variants | not a clean win under same-head retrieval evaluation |
| III | hybrid reranking | improves over untuned hybrid; some downstream gains; slight `nDCG` gains in ablation | still does not beat fixed `kNN` on `Hit@5_same_D7_sign` |

## 6.2 Final corrected hybrid tuning result

The corrected rolling-validation run is recorded in:

- [artifacts/hybrid_retrieval_tuning_full_v2/rolling_validation.md](/home/nmtc/projects/marketlens/artifacts/hybrid_retrieval_tuning_full_v2/rolling_validation.md)

Its global stable selection is:

```text
w_knn    = 0.6
w_learned = 0.4
w_regime  = 0.0
w_prior   = 0.0
```

The corrected global summary is:

- mean validation score: `0.6275`
- validation score std: `0.0430`
- selection objective: `0.5844`

This result is methodologically valid, unlike the earlier `v1` full run, which had an aggregation bug in the global selector.

## 6.3 Final frozen-weight comparison

Using the corrected stable weights, the final retrieval comparison is:

| Method | Hit@5 same D7 sign | nDCG@5 | Downstream DA | Active Acc | Coverage |
|---|---:|---:|---:|---:|---:|
| **fixed_knn** | **0.9312** | **0.3011** | 0.2820 | 0.3320 | 0.8000 |
| learned_only | 0.9109 | 0.2932 | **0.3377** | **0.3782** | 0.7803 |
| hybrid_reranker `0.6/0.4` | 0.9069 | 0.3004 | **0.3377** | 0.3775 | **0.8164** |
| fixed_knn_production_head | **0.9312** | **0.3011** | 0.2852 | 0.3681 | 0.5344 |

This table is decisive.

- Fixed `kNN` wins on the primary retrieval target.
- Fixed `kNN` also edges out the tuned hybrid on `nDCG@5`.
- Learned-only and hybrid variants remain better on downstream classification metrics.

So the learned and hybrid methods are not better retrieval systems yet.

## 6.4 Two-score ablation

To isolate the role of pure `kNN + learned` blending, regime and prior terms were removed and only two convex blends were tested:

| Method | Hit@5 same D7 sign | nDCG@5 | Downstream DA | Active Acc | Coverage |
|---|---:|---:|---:|---:|---:|
| **fixed_knn** | **0.9312** | 0.3011 | 0.2820 | 0.3320 | 0.8000 |
| learned_only | 0.9109 | 0.2932 | **0.3377** | **0.3782** | 0.7803 |
| hybrid `0.7 * knn + 0.3 * learned` | 0.9150 | **0.3038** | 0.3279 | 0.3699 | 0.8066 |
| hybrid `0.8 * knn + 0.2 * learned` | 0.9109 | 0.2998 | 0.2951 | 0.3500 | 0.7869 |

This ablation is the cleanest “last chance” test for hybrid retrieval. The `0.7/0.3` blend is the nearest challenger, but it still fails to beat fixed `kNN` on the main retrieval metric:

- `0.9150` vs `0.9312`

It does improve `nDCG@5` slightly:

- `0.3038` vs `0.3011`

That means the learned score can help smooth rank quality in some sense, but not enough to replace the top-of-list sign consistency that fixed `kNN` already achieves.

---

## 7. Mechanistic Interpretation

### 7.1 Why fixed `kNN` is so strong here

Fixed `kNN` is not a weak baseline in this project. It already benefits from several domain-specific advantages:

1. **strong market-state representation**
   - factor, indicator, and price blocks already encode the local structure of BTC regimes
2. **causal retrieval restriction**
   - all retrieval is historical-only with maturity guards
3. **small-to-medium data regime**
   - nearest-neighbor methods often remain competitive when the geometry is informative and the dataset is not large enough to fully justify more expressive learning
4. **single-asset structure**
   - BTC exhibits strong regime persistence and repeated local price-state motifs, which are well served by memory-based retrieval

In other words, the baseline is already aligned with the structure of the problem.

### 7.2 Why learned retrieval helps downstream metrics

Learned retrieval improves downstream DA and active accuracy because it appears to capture event and sentiment structure that is relevant for classification, even when it does not improve evidence purity under the strict D7-consistency criterion.

This is plausible mechanistically:

- FinBERT-based signals may better identify semantically similar market narratives
- those narratives can help directional classification
- but semantic or event similarity is not always the same thing as **outcome-consistent historical evidence**

So the learned retriever is not useless. It is useful in a different way than the final hybrid retrieval hypothesis required.

### 7.3 Why hybrid reranking underperformed on the main target

The hybrid results suggest a specific failure mode:

- learned scoring can displace some top `kNN` neighbors that are strong market-state analogs
- the replacement neighbors may be semantically plausible, but less outcome-consistent for D7 sign
- this can hurt `Hit@5_same_D7_sign` even when `nDCG@5` or downstream DA improves

The ablation result supports this interpretation:

- more learned weight tends to improve classification-style metrics
- but the best retrieval-side result still comes from the pure `kNN` endpoint

### 7.4 What the 0.7/0.3 result means

The `0.7/0.3` blend is interesting because it slightly improves `nDCG@5` but still loses on `Hit@5_same_D7_sign`.

That implies the learned signal is not random noise. It contains rank information. But the information it adds is not the kind that produces a better top-5 D7-consistent evidence set than pure `kNN`.

That is exactly the kind of nuanced result that would be lost if one reported only downstream DA or only a single ranking metric.

---

## 8. What the Three Experiments Mean Together

Taken together, the three experiments support a layered conclusion.

### 8.1 Experiment I: fixed `kNN` established a serious baseline

The first stage showed that historical analog retrieval in this setting is not trivial, and that a deterministic `kNN` memory can already beat more expensive language-model alternatives in the deployed decision pipeline.

### 8.2 Experiment II: learned retrieval discovered real signal, but not a clean replacement

The second stage proved that there is meaningful learned signal in the problem. FinBERT-enhanced learned retrieval improves several retriever-level and downstream-policy metrics. That matters because it validates the idea that text-informed financial representation learning contributes useful information.

But it did not prove replacement value on retrieval itself.

### 8.3 Experiment III: hybrid reranking clarified the boundary

The third stage tested the strongest version of the replacement hypothesis:

- keep `kNN` for candidate generation
- let learned scoring improve the ranking

That is a fair and academically defensible test. The result is that the hybrid can help in some ways, but still does not exceed the `kNN` endpoint on the primary retrieval objective.

So the progression does not end in “learned beats fixed.” It ends in a more precise statement:

> fixed `kNN` is best for retrieval; learned and hybrid methods are better viewed as downstream decision-quality tools rather than retrieval replacements.

That is a stronger scientific result than simply claiming the newest model is best.

---

## 9. Limitations

Several limitations remain.

### 9.1 Single-asset evaluation

Most reported results here are BTC-centric. A broader multi-asset evaluation could change the balance between fixed geometry and learned event similarity.

### 9.2 One retrieval target

The final retrieval target emphasizes D7 sign consistency. That is a good operational target for current StockMem usage, but other evidence definitions are possible:

- return magnitude similarity
- volatility regime similarity
- event-type correspondence

### 9.3 Hybrid search space still limited

The hybrid search used a constrained convex grid. It is possible that richer reranking features or a trained listwise reranker would perform differently, although the current results suggest caution before investing heavily there.

### 9.4 Downstream vs retrieval mismatch

The project now has strong evidence that downstream policy quality and retrieval evidence purity are not the same objective. Future work should keep those layers explicitly separated from the start.

---

## 10. Practical Recommendation

The operational recommendation is straightforward.

### 10.1 Freeze the retrieval baseline

Use:

- **retrieval baseline:** `fixed_knn`

for evidence retrieval claims and future comparisons.

### 10.2 Freeze the strict classifier baseline and winner

Use:

- **strict classifier baseline:** `fixed_knn_rolling_stable`
- **current best strict classifier candidate:** `fixed_retriever_learned_head`

for the primary held-out classifier table.

### 10.3 Keep learned and hybrid methods in their proper roles

Use learned and hybrid variants for:

- downstream policy experiments,
- candidate-generation ablations,
- research into event- or sentiment-aware decision augmentation.

But do not present hybrid reranking as a superior retrieval system yet, and do not attribute the current strict-test gain to retriever replacement alone.

### 10.4 Recommended next research direction

The most sensible next step is a focused mechanism study rather than another broad hybrid search:

1. finish a paper-clean naive-AI baseline under one frozen prompt,
2. repeat the strict held-out protocol across multiple non-overlapping windows,
3. test whether the learned-head advantage remains stable across regimes,
4. inspect cases where learned head helps despite fixed retrieval remaining unchanged.

---

## 11. Conclusion

This report documented the full retrieval evolution of StockMem:

1. fixed `kNN` established a strong deterministic baseline,
2. learned FinBERT retrieval demonstrated real semantic and policy-level value,
3. hybrid reranking tested whether both signals could be combined into a better evidence retriever.

The answer is now more specific than the original framing.

For retrieval-centered evaluation, fixed `kNN` remains a strong baseline and hybrid reranking still does not clearly beat it on the main retrieval target. But for the strict held-out classifier test, the best structured model is **fixed retriever plus learned stable head**.

That is the full picture:

- the baseline was strong for principled reasons,
- the learned model added real signal,
- the main gain is coming from the learned head more than from retriever replacement,
- the hybrid model was worth testing but is not the clean winner.

This is not a negative result. It is a more useful systems result because it identifies where the improvement actually comes from.

---

## References

1. Dogu Araci. *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv:1908.10063, 2019. https://arxiv.org/abs/1908.10063
2. Sylvain Arlot and Alain Celisse. *A survey of cross-validation procedures for model selection*. arXiv:0907.4728, 2009. https://arxiv.org/abs/0907.4728
3. Sebastian Bruch, Siyu Gai, and Amir Ingber. *An Analysis of Fusion Functions for Hybrid Retrieval*. arXiv:2210.11934, 2022. https://arxiv.org/abs/2210.11934
4. Marius Köppel, Alexander Segner, Martin Wagener, Lukas Pensel, Andreas Karwath, and Stefan Kramer. *Pairwise Learning to Rank by Neural Networks Revisited: Reconstruction, Theoretical Analysis and Practical Performance*. arXiv:1909.02768, 2019. https://arxiv.org/abs/1909.02768
5. Bharath K. Sriperumbudur and Gert R. G. Lanckriet. *Metric Embedding for Nearest Neighbor Classification*. arXiv:0706.3499, 2007. https://arxiv.org/abs/0706.3499

---

## Internal Experiment Sources

- [docs/knn_returns_strategy_report.md](/home/nmtc/projects/marketlens/docs/knn_returns_strategy_report.md)
- [docs/knn_returns_backtest_results.md](/home/nmtc/projects/marketlens/docs/knn_returns_backtest_results.md)
- [docs/learned_retriever_finbert_vs_knn_report_2026-06-27.md](/home/nmtc/projects/marketlens/docs/learned_retriever_finbert_vs_knn_report_2026-06-27.md)
- [docs/learned_finbert_candidate_backtest_2026-06-27.md](/home/nmtc/projects/marketlens/docs/learned_finbert_candidate_backtest_2026-06-27.md)
- [docs/learned_finbert_rolling_stable_report_2026-06-27.md](/home/nmtc/projects/marketlens/docs/learned_finbert_rolling_stable_report_2026-06-27.md)
- [docs/hybrid_retrieval_ablation_report_2026-06-29.md](/home/nmtc/projects/marketlens/docs/hybrid_retrieval_ablation_report_2026-06-29.md)
- [artifacts/hybrid_retrieval_tuning_full_v2/rolling_validation.md](/home/nmtc/projects/marketlens/artifacts/hybrid_retrieval_tuning_full_v2/rolling_validation.md)
- [artifacts/hybrid_retrieval_frozen_v2/d7_consistency_eval.md](/home/nmtc/projects/marketlens/artifacts/hybrid_retrieval_frozen_v2/d7_consistency_eval.md)
- [artifacts/hybrid_retrieval_ablation_7030/d7_consistency_eval.md](/home/nmtc/projects/marketlens/artifacts/hybrid_retrieval_ablation_7030/d7_consistency_eval.md)
- [artifacts/hybrid_retrieval_ablation_8020/d7_consistency_eval.md](/home/nmtc/projects/marketlens/artifacts/hybrid_retrieval_ablation_8020/d7_consistency_eval.md)
