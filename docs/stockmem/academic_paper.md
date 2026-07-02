# StockMem: Structured Historical Market Memory for Retrieval-Augmented Crypto Directional Decision Support

**Project:** MarketLens / StockMem
**Version:** production-ready submission draft
**Primary task:** BTC D7 directional decision support
**Official held-out test window:** `2025-07-01` to `2026-05-01`
**Official test size:** `305` rows
**Primary label rule:** `BUY` if `future_return_7d > +2%`, `SELL` if `< -2%`, otherwise `HOLD`

## Maintainability Note

This report is written as a long-form academic paper, but it is structured so
future revisions are easy to maintain. It uses plain Markdown plus small
machine-readable blocks so that future OpenAI-assisted or human edits can
update facts without rewriting the whole narrative. The report separates:

- **the system objective**: demonstrate a structured StockMem pipeline for
  historical evidence retrieval;
- **the mechanism**: fixed kNN retrieval, learned diagonal retrieval, learned
  stable head, hybrid reranking, and naive LLM baseline;
- **the audit trail**: exact test split, artifact sources, primary metrics,
  confidence intervals, and negative-result interpretations;
- **the citation layer**: references are centralized in the final section and
  mirrored in `docs/references.md`.

The report also uses a PRISMA-like evidence ledger in Section 3. It is not a
formal medical PRISMA review; it borrows the transparency principle from the
PRISMA reporting tradition [10] and adapts it into a screening table for
internal experiments, retained artifacts, and reported claims.

### Report Maintenance Schema

The report can be maintained by updating the following conceptual schema:

```yaml
report:
  objective: "Evaluate structured StockMem retrieval for D7 decision support"
  primary_dataset:
    path: "data/exports/stockmem_records.ndjson"
    test_window: ["2025-07-01", "2026-05-01"]
    label_threshold_pct: 2.0
  primary_claims:
    - structured_memory_beats_naive_llm
    - fixed_knn_is_robust_retriever
    - learned_head_is_strict_test_winner
    - learned_and_hybrid_are_diagnostic_not_replacements
  primary_tables:
    - artifacts/current_context_ai_eval/summary.json
    - artifacts/learned_strict_test_v3/summary.json
    - artifacts/fixed_knn_component_ablation/summary.json
  appendix_tables:
    - artifacts/hybrid_retrieval_frozen_v2/d7_consistency_eval.json
    - artifacts/learned_strict_test_head_aligned/summary.json
  citations:
    - FinBERT
    - cross_validation
    - hybrid_retrieval_fusion
    - pairwise_learning_to_rank
    - metric_embedding
```

When new results are produced, update this schema first, then update the
corresponding table and interpretation section. This keeps historical claims,
current evidence, and citation dependencies separate.

---

## Abstract

Large language models can summarize market context and produce plausible
financial narratives, but a practical forecasting system needs more than
unstructured reasoning over the current day. Market behavior is temporally
structured: a current snapshot should be compared against prior market states,
and any historical evidence used for prediction must be point-in-time,
leakage-controlled, and measurable. This paper presents StockMem, the
historical-memory component of MarketLens, as a structured retrieval pipeline
for crypto directional decision support. StockMem retrieves past BTC market
days from normalized factor, indicator, price, and event representations, then
uses retrieved outcomes to support a D7 `BUY`/`HOLD`/`SELL` decision.

The central research question is whether structured historical memory improves
over naive current-context LLM prompting, and whether more learned retrieval
mechanisms improve the StockMem pipeline beyond a strong fixed kNN baseline.
The final held-out evaluation uses 305 BTC daily records from `2025-07-01` to
`2026-05-01`, labels D7 direction with a `+/-2%` threshold on
`future_return_7d`, and compares naive LLM prompting, fixed kNN retrieval,
learned FinBERT-enhanced retrieval, a validation-selected learned stable head,
hybrid reranking, and feature-block ablations.

The results support a layered conclusion. First, the structured StockMem
pipeline outperforms the naive LLM baseline on the shared held-out split:
`fixed_knn_rolling_stable` reaches `0.3180` overall accuracy and `0.4236`
active accuracy, compared with `0.2787` and `0.4031` for the naive LLM
baseline. Second, the strongest strict structured variant is not a fully
learned retriever. It is fixed kNN retrieval combined with a learned stable
decision head, reaching `0.3508` overall accuracy, `0.4500` active accuracy,
and `0.8525` coverage. Third, learned retrieval improves some retrieval and
classification diagnostics, especially `Hit@5_same_sign`, but it does not
clearly replace fixed kNN as the robust retrieval engine. Hybrid reranking and
head-aligned retriever training were useful negative experiments: they
clarified that learned scores can help some ranking or downstream metrics, but
they do not yet dominate fixed kNN evidence retrieval.

The main contribution is therefore not a claim that the most complex model is
best. It is a defensible applied methodology: structured, leakage-controlled
historical memory is more reliable than naive current-context LLM prompting for
this task, and the current best StockMem architecture is a deterministic fixed
kNN retriever with a validation-selected learned stable head.

---

## 1. Introduction

Financial prediction systems often fail because their evidence path is unclear.
An LLM can produce a convincing explanation from recent headlines, but that
explanation may not be grounded in comparable historical cases. Conversely, a
purely numerical nearest-neighbor system can be deterministic and auditable,
but it may miss textual or event-level information that matters for market
state. StockMem sits between these approaches. It stores structured daily
market memory and retrieves historical analogs for the current snapshot, making
prediction support explicit and inspectable.

The broader MarketLens pipeline collects crypto news, extracts sentiment and
factor information, combines it with market data, stores daily records, and
retrieves historical neighbors from StockMem. The research focus of this paper
is narrower: given a frozen StockMem export, can structured historical memory
support D7 directional decisions better than a naive LLM prompt that sees only
current market/news context? If yes, which part of the structured pipeline is
responsible for the gain?

This question evolved through several experimental stages. The first stage
tested a fixed weighted kNN retriever over factor, indicator, and price vectors.
The second stage trained a learned diagonal metric informed by FinBERT-style
sentiment and event representations. The third stage tested hybrid reranking:
fixed kNN generated candidate evidence, then learned and regime-aware scores
reranked the candidate set. Finally, mechanism-focused ablations separated the
retriever from the decision head and compared the structured pipeline with a
naive current-context LLM baseline.

The result is nuanced but useful. Fixed kNN remains a strong retrieval method.
Learned retrieval contains real signal, but it is not a clean replacement for
fixed kNN on the final strict test. The largest strict-test gain comes from the
learned stable decision head applied to fixed-kNN candidates. This matters for
an applied graduation project because it shows not only that the pipeline works,
but also why it works: a stable historical-memory retriever provides the
evidence set, while the learned head improves how that evidence is converted
into a decision.

The paper makes five claims:

1. **Structured memory beats naive prompting** on the official held-out split.
2. **Fixed kNN is a robust evidence generator**, not a weak baseline.
3. **The learned stable head is the main source of the strict-test gain**.
4. **Learned retrieval and hybrid reranking are useful but not dominant** under
   the current objective.
5. **Negative results strengthen the methodology** because they separate
   retrieval quality from downstream decision quality.

---

## 2. Related Work And Positioning

### 2.1 Financial Language Models

FinBERT demonstrates the value of adapting BERT-style representations to
financial sentiment analysis [1]. In StockMem, FinBERT-style sentiment is not
used as a standalone forecasting model. It is used as part of the information
available to the learned retrieval path and event-aware representations. This
distinction is important: the project does not claim that FinBERT alone solves
market prediction. It uses financial language representation as one feature
source inside a larger historical-memory system.

### 2.2 Nearest-Neighbor Methods And Metric Learning

Nearest-neighbor classifiers are simple, but their behavior depends strongly
on the metric. Metric embedding work shows that adapting the distance or
similarity function can materially change nearest-neighbor performance [5].
This motivates the learned diagonal retriever:

```text
score_learned(q,c) = sum_b alpha_b * cos(D_b q_b, D_b c_b)
```

where `D_b` learns dimension weights and `alpha_b` learns block weights. The
StockMem learned retriever is therefore a metric-learning extension of the
fixed kNN system, not a black-box sequence model.

### 2.3 Hybrid Retrieval And Reranking

Hybrid retrieval combines different retrieval signals, often lexical and dense,
with a fusion function. Bruch et al. analyze fusion functions for hybrid
retrieval and motivate score-combination approaches when signals are
complementary [3]. StockMem adapts this idea to market evidence retrieval:
fixed kNN is treated as a market-state candidate generator, while learned,
regime, and prior scores are possible reranking components.

Pairwise learning-to-rank work also motivates separating candidate generation
from ordering [4]. In StockMem, the hybrid reranker is intentionally
interpretable and convex:

```text
score_hybrid(q,c) =
  w_knn     * s_knn(q,c)
  + w_learned * s_learned(q,c)
  + w_regime  * s_regime(q,c)
  + w_prior   * s_prior(q,c)
```

with non-negative weights summing to 1.

### 2.4 Time-Aware Validation

Market data are time ordered. Random splitting can leak regime information and
make a model appear more stable than it is. Arlot and Celisse survey
cross-validation procedures and motivate careful validation design [2]. The
StockMem experiments therefore use chronological splits, mature historical
pools, rolling validation for hybrid tuning, and a final untouched held-out
test period.

### 2.5 Retrieval-Augmented Financial Forecasting

Recent financial RAG systems, including FinSeer-style financial time-series
retrieval and StockMem-like event-reflection memory papers, motivate the
general idea that historical evidence can support financial forecasting [6,7].
FinSeer is especially relevant because it treats retrieval as a forecasting
component rather than a document-search add-on. The event-reflection StockMem
paper is relevant because it argues that historical event chains and their
subsequent price reactions can be stored as reusable memory.

This project differs by emphasizing an applied, audit-friendly pipeline rather
than an end-to-end learned model. It asks which component actually improves
performance under strict held-out evaluation.

### 2.6 Financial LLMs And Time-Series Forecasting Baselines

Financial LLM work such as FinGPT shows why domain-specific financial language
processing is useful, but also why data curation and controlled evaluation are
essential [8]. Dissemination-aware FinGPT variants further motivate the idea
that news breadth and contextual framing matter, not only the sentiment of a
single article. This supports the StockMem design choice to store structured
daily records rather than feed raw headlines directly into a final prompt.

Time-series transformer work such as PatchTST shows that market forecasting
can be framed through temporal representation learning over price sequences
[9]. StockMem does not compete as a deep time-series architecture in this
report. Instead, it contributes a memory and retrieval layer that can coexist
with future temporal encoders. A future MarketLens system could use PatchTST or
another temporal model as a price encoder, while StockMem remains responsible
for historical evidence retrieval and auditability.

The current paper therefore positions StockMem as a middle layer:

```text
raw market/news data -> structured daily memory -> historical evidence retrieval
                     -> decision support and audit trail
```

The contribution is not that kNN is theoretically novel. The contribution is
that a deterministic, leakage-controlled memory layer is shown to outperform a
naive current-context LLM baseline and remains competitive against learned and
hybrid retrieval variants.

---

## 3. Evidence Ledger And Audit Method

This section records how internal evidence was selected for the maintained
report. It is inspired by the transparency goals of PRISMA-style reviews, but
adapted to engineering artifacts and experiments.

The goal is claim traceability. Every numerical statement in the paper should
map to one of four objects:

```text
artifact -> table -> interpretation -> claim
```

If an artifact changes, the corresponding table and claim can be updated
without changing unrelated sections. This is the main maintenance benefit of
the PRISMA-like structure.

### 3.1 Evidence Sources

| Source group | Included material | Reason |
| --- | --- | --- |
| Maintained docs | `docs/stockmem/methodology.md`, `docs/stockmem/experiments.md`, `docs/aihub/llm_baseline.md` | Current cleaned source of truth. |
| Archived StockMem reports | learned retriever report, retrieval evolution report, hybrid ablation report, current-context AI ablation report | Contains mechanism, chronology, and original result interpretation. |
| Local artifacts | `summary.json` and `summary.md` outputs in local `artifacts/` | Source of official tables and paired statistics. |
| Code | evaluator scripts and common metric helpers | Confirms metric definitions, split logic, and model variants. |
| External literature | arXiv papers and local papers in `docs/upgrade/` | Supports methodology and citations. |

### 3.2 Evidence Inclusion Rules

An internal result is included if it satisfies all of the following:

1. It uses a known data split or clearly identifies its evaluation window.
2. It records the model variant and metric definitions.
3. It is relevant to one of the paper's mechanism questions:
   retrieval, decision head, naive LLM baseline, hybrid reranking, or
   feature-block ablation.
4. It does not contradict a later, cleaner artifact without being labeled as
   historical or superseded.

### 3.3 Evidence Exclusion Rules

An internal result is excluded from the primary claim if:

1. It was a smoke run or partial run.
2. It was later superseded by a corrected evaluation.
3. It measured a different objective without clear comparability.
4. It lacked a mature-pool or chronological validation guard.

For example, early hybrid tuning outputs are archived but not authoritative
because a later corrected `v2` run fixed the global selector aggregation.

### 3.4 Primary Evidence Kept

| Evidence item | Maintained use |
| --- | --- |
| Strict structured test, `learned_strict_test_v3` | Primary model comparison. |
| Naive LLM vs StockMem summary | Pipeline comparison against current-context prompting. |
| Feature-block ablation | Mechanism audit, with caveat that not every block is necessary. |
| Hybrid reranking final and ablation summaries | Negative result for learned reranking as retrieval replacement. |
| Head-aligned retriever test | Negative result showing overfit and event-block collapse. |

### 3.5 Claim Ledger

The following ledger links each major claim to the evidence used in this
paper:

| Claim | Evidence source | Status |
| --- | --- | --- |
| Structured StockMem beats naive current-context LLM prompting. | `artifacts/current_context_ai_eval/summary.json` | Supported on the 305-row held-out test. |
| Fixed kNN is a robust evidence generator. | strict comparison and hybrid retrieval results | Supported as the recommended retriever. |
| Learned retrieval improves some evidence diagnostics. | `Hit@5_same_sign` in strict comparison | Supported but not sufficient for replacement. |
| The learned stable head is the strongest strict-test decision layer. | `learned_strict_test_v3` | Supported by overall, active, and coverage metrics. |
| Hybrid reranking is not the current winner. | hybrid reranking final and two-score ablation | Supported as a negative result. |
| Every representation block is individually necessary. | feature-block ablation | Not supported; claim rejected. |

This ledger is intentionally conservative. It prevents the report from
presenting a broader claim than the artifacts justify.

---

## 4. System Objective

The main objective is to evaluate a structured historical-memory pipeline for
crypto market decision support. The task is not to maximize trading PnL first.
The immediate academic target is better, auditable historical evidence and a
more reliable D7 directional decision.

For a query day `q`, the system must choose:

```text
y_hat(q) in {BUY, HOLD, SELL}
```

using only information that would have been available at or before day `q`.
The realized label is:

```text
y(q) = BUY   if r_7(q) >  tau
     = SELL  if r_7(q) < -tau
     = HOLD  otherwise
```

where:

- `r_7(q)` is `future_return_7d`,
- `tau = 2.0` percentage points.

The pipeline is successful if it:

1. beats naive current-context LLM prompting,
2. remains reproducible and point-in-time safe,
3. exposes historical evidence rather than only a black-box answer,
4. identifies which mechanism contributes to the gain.

### 4.1 Formal Objective Decomposition

The StockMem problem is intentionally decomposed into retrieval and decision
layers. Let:

```text
X_t = (F_t, I_t, P_t, E_t)
```

where `F_t`, `I_t`, `P_t`, and `E_t` are factor, indicator, price, and event
representations for day `t`. Let the matured historical pool for query day `t`
be:

```text
M_t = {c : date(c) + 7 days <= date(t)}
```

A retriever `R` maps a query and matured pool to an ordered list:

```text
R(X_t, M_t) = (c_1, c_2, ..., c_k)
```

A decision head `g` maps retrieved evidence to a prediction:

```text
y_hat_t = g(R(X_t, M_t))
```

The project is not simply optimizing `y_hat_t` directly. It asks which
subsystem contributes:

```text
retriever contribution = change R while holding g fixed
head contribution      = change g while holding R fixed
pipeline contribution  = change both R and g
```

This decomposition is why the strict structured table includes:

- fixed retriever + fixed head,
- fixed retriever + learned head,
- learned retriever + fixed head,
- learned retriever + learned head.

Without this decomposition, the project could incorrectly attribute the
learned-head gain to learned retrieval.

### 4.2 Evidence Utility Objective

The evidence utility objective is not identical to classifier accuracy. A
retrieved set is useful if it contains historical records with the same D7
direction as the query:

```text
U(E_k(t), y_t) =
  1 if exists c in E_k(t) with y_c = y_t
  0 otherwise
```

The pipeline may achieve a high decision score even if `U` does not improve,
because a head can aggregate mixed evidence better. Conversely, a retriever may
improve `U` without improving the final prediction if the head is misaligned.
This is the central reason the report distinguishes retrieval metrics from
decision metrics.

### 4.3 Operational Objective

For a production-oriented application, the desired system should satisfy:

```text
maximize   decision_quality + evidence_quality + interpretability
subject to no_lookahead_leakage
           reproducible_outputs
           stable_inference
           manageable_operational_cost
```

The naive LLM baseline is cheap to implement but weak on evidence quality and
stability. The learned retriever is more flexible but can overfit. The fixed
kNN retriever is stable and interpretable. The learned stable head improves
decision quality while keeping the retrieval evidence stable.

---

## 5. Data And Temporal Splits

### 5.1 Dataset

The official evaluation uses a StockMem NDJSON export of BTC daily records. The
clean branch does not commit the dataset; it is treated as an external input at
`data/exports/stockmem_records.ndjson` or a user-supplied `--dataset` path.

Each usable row includes:

- date,
- factor vector,
- indicator vector,
- price vector,
- optional event vector,
- realized future returns across multiple horizons.

### 5.2 Split Protocol

The maintained split is:

| Split | Date range | Rows |
| --- | --- | ---: |
| Train | `2018-01-05` to `2024-12-24` | 2363 |
| Validation | `2025-01-01` to `2025-06-23` | 174 |
| Test | `2025-07-01` to `2026-05-01` | 305 |
| Embargo/other | outside usable split windows | 43 |

The final claims use the `305`-row test split. The validation split is used for
configuration and head selection. The test split is not used for model or head
tuning.

### 5.3 Maturity Guard

For D7 evaluation, a candidate historical day `c` is eligible for query day `q`
only if:

```text
date(c) + 7 days <= date(q)
```

This ensures that the candidate's D7 return would have been known before the
query day. It prevents lookahead leakage in retrieval and decision-head
aggregation.

---

## 6. Representation

### 6.1 Market-State Blocks

The fixed retrieval representation uses three normalized blocks:

| Block | Dimensionality | Meaning |
| --- | ---: | --- |
| `factor_vec` | 75 | Factor taxonomy, extracted event/factor structure, market context. |
| `indicator_vec` | 5 | Compact indicators and sentiment-related summary features. |
| `price_vec` | 60 | Price, volume, range, and recent market path structure. |

Learned retrieval paths may add:

| Block | Dimensionality | Meaning |
| --- | ---: | --- |
| `event_vec` | 85 | Event-state representation used in learned metric paths. |

All cosine similarities assume block-normalized vectors.

### 6.2 Fixed kNN Similarity

The fixed kNN retriever computes:

```text
s_fixed(q,c) =
  w_f * cos(f_q, f_c)
  + w_i * cos(i_q, i_c)
  + w_p * cos(p_q, p_c)
```

where:

- `f` is `factor_vec`,
- `i` is `indicator_vec`,
- `p` is `price_vec`.

The tuned weights are:

| Weight | Value |
| --- | ---: |
| `w_f` | 0.5443920554 |
| `w_i` | 0.3090805325 |
| `w_p` | 0.1415662727 |

The score is deterministic. If the query vector, candidate pool, and weights
are unchanged, the retrieved ranking is unchanged.

### 6.3 Learned Diagonal Metric

The learned retriever adapts the similarity function:

```text
s_learned(q,c) =
  sum_{b in B} alpha_b * cos(D_b x_{q,b}, D_b x_{c,b})
```

where:

- `B` is the set of feature blocks,
- `alpha_b >= 0`,
- `sum_b alpha_b = 1`,
- `D_b` is a non-negative diagonal reweighting matrix,
- `x_{q,b}` and `x_{c,b}` are block vectors for query and candidate.

This is a metric-learning approach to nearest-neighbor retrieval. It can
emphasize individual dimensions and blocks that are predictive under the
training objective.

### 6.4 Hybrid Reranking

Hybrid reranking separates candidate generation from ordering:

1. fixed kNN retrieves the top 30 candidates;
2. a convex fusion score reranks those candidates;
3. the final evidence set is the top 5 reranked candidates.

The fusion score is:

```text
s_hybrid(q,c) =
  w_knn     * s_knn(q,c)
  + w_lrn   * s_learned(q,c)
  + w_reg   * s_regime(q,c)
  + w_prior * s_prior(q,c)
```

subject to:

```text
w_knn, w_lrn, w_reg, w_prior >= 0
w_knn + w_lrn + w_reg + w_prior = 1
```

The corrected stable hybrid selected:

| Component | Weight |
| --- | ---: |
| `w_knn` | 0.6 |
| `w_lrn` | 0.4 |
| `w_reg` | 0.0 |
| `w_prior` | 0.0 |

---

## 7. Decision Heads

### 7.1 Fixed Stable Head

The fixed stable head maps retrieved neighbors to a decision. It aggregates
future returns from the top candidates and applies thresholds. This makes the
retriever and decision layer separable: the same retrieved set can be evaluated
with different heads, and the same head can be tested with different retrievers.

### 7.2 Learned Stable Head

The strongest strict-test variant uses fixed kNN retrieval plus the learned
stable head. The learned head is validation-selected, not a neural network. It
uses:

```text
H(c) =
  beta_1  * r_1(c)
  + beta_3  * r_3(c)
  + beta_7  * r_7(c)
  + beta_15 * r_15(c)
  + beta_30 * r_30(c)
```

with:

| Horizon | Weight |
| --- | ---: |
| `1d` | 0.0161 |
| `3d` | 0.1459 |
| `7d` | 0.4549 |
| `15d` | 0.1005 |
| `30d` | 0.2827 |

For the top `k=5` retrieved candidates:

```text
H_bar(q) = (1/k) * sum_{c in top_k(q)} H(c)
```

The head emits:

```text
y_hat(q) = BUY   if H_bar(q) >  1.45
         = SELL  if H_bar(q) < -1.47
         = HOLD  otherwise
```

This head is the best current decision layer for the strict held-out test when
paired with fixed kNN retrieval.

### 7.3 Why The Head Is Not A Black-Box Model

The learned stable head is sometimes easy to misunderstand. It is called
"learned" because its hyperparameters were selected from validation evidence,
not because it is a neural network. It has three interpretable parts:

1. `k`, the number of retrieved neighbors used;
2. return-horizon weights;
3. buy and sell thresholds.

This makes the head auditable. For any prediction, the evaluator can expose:

```text
top retrieved dates
their future returns by horizon
the weighted average per neighbor
the final mean score
the threshold comparison
```

The model remains closer to a transparent scoring rule than to an opaque
classifier. This is important for a graduation-project defense because the
system can explain how a signal was constructed from historical evidence.

### 7.4 Decision Head Error Modes

The head can still fail. Typical failure modes include:

- the retrieved neighbors are historically similar but belong to a different
  macro regime;
- the top neighbors contain mixed future returns and the average crosses a
  threshold by a small margin;
- D7 labels are noisy around the `+/-2%` threshold;
- a head tuned for active accuracy may increase coverage while reducing
  conservative `HOLD` precision.

These failure modes are preferable to silent LLM failure because they are
inspectable. A developer can audit neighbor dates, horizon weights, and the
threshold margin.

---

## 8. Baselines And Model Variants

### 8.1 Naive Current-Context LLM

The naive baseline sends the current day context to an LLM:

- current candle and market snapshot,
- recent one-day and three-day price changes,
- current indicators,
- one-day aggregated news sentiment,
- compact news headline bundle.

It does not receive retrieved historical neighbors. This baseline tests whether
raw current-context reasoning is enough without StockMem.

### 8.2 Fixed kNN Rolling Stable

`fixed_knn_rolling_stable` uses the fixed weighted kNN retriever and the fixed
stable head. It is the deterministic structured baseline.

### 8.3 Fixed Retriever With Learned Head

`fixed_retriever_learned_head` keeps fixed kNN retrieval but uses the learned
stable decision head. This isolates the head's contribution.

### 8.4 Learned Retriever With Fixed Head

`learned_retriever_fixed_head` changes the retriever while holding the fixed
head constant. This isolates retriever replacement.

### 8.5 Learned FinBERT Rolling Stable

`learned_finbert_rolling_stable` changes both learned retrieval and learned
stable head. It tests the full learned pipeline.

### 8.6 Hybrid Reranker

The hybrid reranker generates candidates with fixed kNN and reranks using a
weighted fusion of fixed, learned, regime, and prior scores.

### 8.7 Head-Aligned Retriever

The head-aligned retriever was trained to rank candidates that support the
frozen learned head. It is included as a negative result because it overfit
validation and did not improve the held-out test.

---

## 9. Metrics

### 9.1 Overall Accuracy

```text
overall_acc = (1/N) * sum_j 1[y_hat_j = y_j]
```

where `y_j` is the three-class D7 label.

### 9.2 Active Accuracy

Active accuracy ignores `HOLD` predictions and asks whether active trades have
the correct sign:

```text
active_acc =
  (# correct BUY/SELL predictions) / (# BUY/SELL predictions)
```

A `BUY` is active-correct if `future_return_7d > 0`. A `SELL` is active-correct
if `future_return_7d < 0`.

### 9.3 Coverage

```text
coverage = (# BUY predictions + # SELL predictions) / N
```

Coverage matters because a model can improve accuracy by emitting too many
`HOLD` decisions.

### 9.4 Hit@5 Same D7 Sign

For a query `q`, let `E_5(q)` be the top five retrieved evidence records.

```text
Hit@5_same_sign(q) =
  1 if exists c in E_5(q) such that label(c) = label(q)
  0 otherwise
```

The metric is averaged across queries.

### 9.5 nDCG@5

Normalized discounted cumulative gain measures whether more relevant evidence
is ranked closer to the top:

```text
DCG@5 = sum_{r=1}^{5} rel_r / log2(r + 1)
nDCG@5 = DCG@5 / IDCG@5
```

where `IDCG@5` is the ideal DCG for the same candidate pool.

### 9.6 Bootstrap Confidence Intervals

For paired model comparison, the evaluation resamples rows with replacement,
computes metric deltas, and reports the 2.5% and 97.5% quantiles:

```text
Delta_m = m(challenger) - m(baseline)
CI_95 = [Q_0.025(Delta_m), Q_0.975(Delta_m)]
```

### 9.7 McNemar Exact Test

McNemar's test uses discordant paired correctness outcomes:

| | Challenger wrong | Challenger correct |
| --- | ---: | ---: |
| Baseline correct | `b` | - |
| Baseline wrong | - | `c` |

The exact two-sided p-value is computed from the binomial tail over `b + c`
discordant pairs. This tests whether the two classifiers differ in paired
correctness frequency.

---

## 10. Experimental Design

### 10.1 Experiment A: Naive LLM Versus Structured StockMem

This experiment asks:

```text
Can a current-context LLM replace structured historical memory?
```

The LLM sees current market and news context only. It does not see retrieved
neighbors. The structured baselines use StockMem retrieval and deterministic
heads.

### 10.2 Experiment B: Strict Structured Model Comparison

This experiment asks:

```text
Which structured mechanism is strongest on the same held-out split?
```

It compares:

- fixed retriever + fixed head,
- fixed retriever + learned head,
- learned retriever + fixed head,
- learned retriever + learned head.

This is the most important mechanism table because it isolates where the
strict-test gain comes from.

### 10.3 Experiment C: Feature-Block Ablation

This experiment asks:

```text
Does disabling each fixed-kNN block reduce performance?
```

The tested variants are:

- full fixed kNN,
- no factor block,
- no indicator block,
- no price block,
- factor only,
- indicator only,
- price only.

The result is important because it prevents an overclaim. The current evidence
does not prove that every block is individually necessary for strict
classification.

### 10.4 Experiment D: Hybrid Reranking

This experiment asks:

```text
Can learned reranking improve fixed-kNN evidence?
```

It tests whether learned scores improve D7-consistent evidence when applied
only to fixed-kNN candidates.

### 10.5 Experiment E: Head-Aligned Retriever Training

This experiment asks:

```text
Can training the retriever directly against the learned head improve the final
decision?
```

The answer from the first run is no. The artifact overfit validation and
collapsed nearly all block weight onto the event block.

### 10.6 Algorithmic Description

The official structured evaluation can be summarized as:

```text
Input:
  rows D sorted by date
  query split T
  retriever R
  decision head g
  horizon h = 7 days
  threshold tau = 2.0

For each query q in T:
  pool = {c in D : date(c) + h <= date(q)}
  ranked = R(q, pool)
  evidence = top_k(ranked)
  y_hat = g(evidence)
  y = label(future_return_7d(q), tau)
  record prediction row

Aggregate:
  overall_acc
  active_acc
  coverage
  Hit@5_same_sign
  confusion matrix
```

The hybrid evaluation differs only in the retriever:

```text
candidate_pool = top_30_fixed_knn(q, pool)
ranked = sort_by_hybrid_score(q, candidate_pool)
evidence = top_5(ranked)
```

The naive LLM evaluation differs more substantially:

```text
prompt = current_market_context(q) + compact_news_context(q)
y_hat = LLM(prompt)
```

It has no `pool`, no historical evidence set, and no deterministic neighbor
audit trail.

### 10.7 Why The Test Is Strict

The strict test is intentionally conservative:

1. The held-out window is chronological.
2. Retrieval uses only matured historical records.
3. The label threshold creates three classes rather than a binary direction
   shortcut.
4. Active accuracy and coverage are reported together to avoid trivial
   `HOLD`-heavy solutions.
5. Paired tests compare models on the same dates.

This design is more defensible than reporting only backtest return, because
return metrics can be dominated by a few large BTC moves and by assumptions
about position sizing, transaction costs, and overlapping trades.

---

## 11. Results

### 11.1 Naive LLM Versus Structured StockMem

Source: `artifacts/current_context_ai_eval/summary.json`.

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_current_ai` | 305 | 0.2787 | 0.4031 | 0.6426 | 0.8361 | 0.6164 | 0.3574 | 0.0262 |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 | 0.6295 | 0.2492 | 0.1213 |
| `knn_returns` | 305 | 0.2918 | 0.4146 | 0.6721 | 0.8361 | 0.5574 | 0.3279 | 0.1148 |

The structured fixed-kNN baseline beats the naive LLM baseline by:

```text
overall_acc delta = 0.3180 - 0.2787 = +0.0393
active_acc delta  = 0.4236 - 0.4031 = +0.0205
coverage delta    = 0.7508 - 0.6426 = +0.1082
```

The LLM's main failure mode is action distribution. It emits `SELL` only
`2.62%` of the time, while the test set contains 149 actual `SELL` labels out
of 305 rows. This suggests that current-context language reasoning is biased
toward bullish or non-committal predictions and does not recover enough
negative evidence from one-day news context alone.

### 11.2 Primary Structured Model Comparison

Source: `artifacts/learned_strict_test_v3/summary.json`.

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 |
| `fixed_retriever_learned_head` | 305 | **0.3508** | **0.4500** | **0.8525** | 0.8361 |
| `learned_retriever_fixed_head` | 305 | 0.3148 | 0.4182 | 0.7213 | **0.8459** |
| `learned_finbert_rolling_stable` | 305 | 0.3410 | 0.4393 | 0.7836 | **0.8459** |

The strongest model is fixed retrieval plus learned head. This shows that the
strict-test gain is not primarily from replacing kNN retrieval. The learned
retriever improves `Hit@5_same_sign` from `0.8361` to `0.8459`, but that
retrieval improvement does not translate into the best overall decision score.

### 11.3 Paired Statistics

Primary pair: `fixed_knn_rolling_stable` versus
`fixed_retriever_learned_head`.

| Metric | Delta | 95% bootstrap CI |
| --- | ---: | ---: |
| overall_acc | +0.0330 | [+0.0000, +0.0689] |
| active_acc | +0.0264 | [+0.0013, +0.0529] |
| coverage | +0.1014 | [+0.0623, +0.1410] |
| hit_at_5_same_sign | +0.0000 | [+0.0000, +0.0000] |

McNemar exact test:

```text
p = 0.087159
discordant pairs = 28
fixed-only correct = 9
learned-head-only correct = 19
```

The effect is directionally positive and practically meaningful, especially
for coverage and active accuracy. The McNemar p-value is borderline rather
than definitive, so the report should avoid overstating statistical
significance.

Secondary pair: `fixed_knn_rolling_stable` versus
`learned_finbert_rolling_stable`.

```text
overall_acc delta = +0.0222
95% bootstrap CI = [-0.0393, +0.0852]
McNemar p = 0.550709
```

The full learned pipeline is descriptively better than fixed stable, but the
paired uncertainty is wider.

### 11.4 Feature-Block Ablation

Source: `artifacts/fixed_knn_component_ablation/summary.json`.

| Variant | Overall Acc | Active Acc | Coverage | Hit@5 same sign |
| --- | ---: | ---: | ---: | ---: |
| `full_fixed_knn` | 0.3180 | 0.4236 | 0.7508 | 0.8361 |
| `no_factor_block` | 0.2918 | 0.4685 | 0.7279 | 0.8525 |
| `no_indicator_block` | 0.3115 | 0.4361 | 0.7443 | 0.8197 |
| `no_price_block` | 0.3443 | 0.4498 | 0.7508 | 0.8492 |
| `factor_only` | 0.3475 | 0.4789 | 0.8557 | 0.8459 |
| `indicator_only` | 0.3475 | 0.5144 | 0.6820 | 0.8557 |
| `price_only` | 0.3344 | 0.4656 | 0.8098 | 0.8754 |

This ablation produces a critical caveat. It does not support the claim that
all three blocks are individually necessary under the strict classifier
metric. Several reduced variants improve overall or active accuracy. The
correct interpretation is narrower:

- the full fixed-kNN configuration is the official tuned operational baseline;
- each block contains useful signal;
- the present ablation does not prove monotonic necessity of every block;
- future work should retune weights under the strict objective if block-level
  necessity is a primary claim.

### 11.5 Hybrid Reranking

The corrected stable hybrid selected:

```text
w_knn = 0.6
w_learned = 0.4
w_regime = 0.0
w_prior = 0.0
```

Final retrieval comparison:

| Method | Hit@5 same D7 sign | nDCG@5 | Downstream DA | Active Acc | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn` | **0.9312** | **0.3011** | 0.2820 | 0.3320 | 0.8000 |
| `learned_only` | 0.9109 | 0.2932 | **0.3377** | **0.3782** | 0.7803 |
| `hybrid_reranker_0.6_0.4` | 0.9069 | 0.3004 | **0.3377** | 0.3775 | **0.8164** |
| `fixed_knn_production_head` | **0.9312** | **0.3011** | 0.2852 | 0.3681 | 0.5344 |

Two-score ablation:

| Method | Hit@5 same D7 sign | nDCG@5 | Downstream DA | Active Acc | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn` | **0.9312** | 0.3011 | 0.2820 | 0.3320 | 0.8000 |
| `learned_only` | 0.9109 | 0.2932 | **0.3377** | **0.3782** | 0.7803 |
| `hybrid_0.7_knn_0.3_learned` | 0.9150 | **0.3038** | 0.3279 | 0.3699 | 0.8066 |
| `hybrid_0.8_knn_0.2_learned` | 0.9109 | 0.2998 | 0.2951 | 0.3500 | 0.7869 |

Hybrid reranking did not beat fixed kNN on the main retrieval objective. The
`0.7/0.3` hybrid slightly improved `nDCG@5`, which implies the learned score
contains ranking information, but it did not improve the main top-5 sign
consistency target.

### 11.6 Head-Aligned Retriever

The head-aligned retriever was trained to rank candidates that support the
frozen learned head. It overfit:

```text
validation overall = 0.4540
validation active  = 0.5890
held-out overall   = 0.2820
held-out active    = 0.3942
held-out coverage  = 0.7902
```

The trained artifact collapsed to:

```text
block_scales = [0.999701, 0.0001, 0.0001, 0.0001]
```

This means the model placed almost all weight on the event block. The result is
a useful negative audit: optimizing directly against a head-aligned target can
find validation shortcuts that do not generalize.

---

## 12. Discussion

### 12.1 Why StockMem Beats Naive LLM Prompting

The naive LLM baseline sees only current context. It has no structured way to
ask, "what happened after similar historical states?" StockMem directly answers
that question by retrieving mature historical analogs. This matters in market
data because current news sentiment is often ambiguous. Negative news during a
bull regime can be absorbed; positive news during a bear regime may fail to
reverse price. Historical analog retrieval adds regime context that raw
headlines do not provide.

The naive LLM's SELL avoidance is consistent with this limitation. It emits
almost no `SELL` predictions even though the test window contains many negative
D7 outcomes. The structured model does not solve SELL perfectly, but it emits a
more balanced active distribution because it uses historical return evidence.

### 12.2 Why Fixed kNN Is Strong

Fixed kNN is strong because it is not naive. It uses a domain-engineered market
state representation and tuned weights. Its advantages are:

1. deterministic inference;
2. point-in-time retrieval;
3. interpretable neighbor evidence;
4. a geometry aligned with repeated BTC regimes;
5. lower overfit risk in a small-to-medium sample.

This explains why the learned and hybrid methods do not automatically win.
More flexibility can help, but it can also displace historically useful market
analogs.

### 12.3 Why The Learned Head Helps

The learned head improves the conversion from evidence to decision. The fixed
retriever supplies a stable candidate set; the learned head changes how
candidate outcomes across horizons are aggregated. The strict-test winner keeps
retrieval stable and learns the decision surface. Mechanistically, this is a
cleaner improvement than replacing the retriever, because it reduces the risk
of retrieving semantically plausible but outcome-inconsistent evidence.

### 12.4 Why Learned Retrieval Is Still Valuable

The learned retriever improves `Hit@5_same_sign` in the strict structured
table. It also improves some downstream metrics in older candidate-level
experiments. This means learned retrieval is not noise. Its issue is alignment:
the learned objective and the final decision target are not identical. It may
retrieve event-similar cases that are semantically reasonable but less useful
for the D7 decision head.

### 12.5 Why Hybrid Reranking Did Not Win

Hybrid reranking assumes fixed and learned scores are complementary. The
experiments show partial complementarity: `nDCG@5` can improve slightly, and
downstream DA can improve. But the primary retrieval target,
`Hit@5_same_D7_sign`, remains strongest under fixed kNN. The learned score may
reorder the candidate pool in a way that helps some graded relevance metrics
but hurts the binary top-5 same-sign objective.

### 12.6 What The Feature Ablation Really Says

The feature ablation is a useful guardrail against overclaiming. It shows that
single-block variants can outperform the full fixed configuration under the
strict test. Therefore, the paper should not claim that factor, indicator, and
price blocks are each strictly necessary. The better claim is:

```text
StockMem's structured representation provides useful signal, but the best
feature weighting depends on the evaluation objective and should be validated
under the target protocol.
```

### 12.7 Case-Level Interpretability

A practical advantage of StockMem is that every structured prediction can be
expanded into historical evidence. A typical audit view can show:

```text
query date
query market features
top-5 retrieved historical dates
similarity scores
retrieved future returns
weighted head score
final threshold comparison
```

This is useful for presentation and debugging. If the model emits `BUY`, the
system can show whether the retrieved neighbors mostly had positive returns or
whether the signal came from one extreme neighbor. If the model emits `HOLD`,
the system can show whether evidence was mixed or whether the weighted score
fell inside the neutral band. A naive LLM prompt cannot provide this same
structured audit unless retrieval is added.

### 12.8 Why Negative Results Are Part Of The Contribution

The hybrid and head-aligned experiments are not failures to hide. They are
evidence that the project tested plausible alternatives:

- learned retrieval alone;
- learned retrieval with the fixed head;
- learned retrieval with the learned head;
- convex hybrid reranking;
- head-aligned retriever training;
- feature-block removal.

This is important because a simple report could cherry-pick only the winning
pipeline. The present report instead shows that the final recommendation is
the result of a mechanism search. Fixed kNN plus learned head is selected not
because it is the newest method, but because it gives the best strict-test
tradeoff among tested variants.

### 12.9 Why This Is Defensible As An Applied Academic Project

For an applied graduation project, the strongest contribution is an end-to-end
pipeline with evidence, not a theoretical proof of market predictability.
StockMem is defensible because it provides:

1. a clear data representation;
2. a leakage-controlled retrieval protocol;
3. deterministic baseline retrieval;
4. validation-selected decision rules;
5. held-out evaluation;
6. statistical paired comparisons;
7. negative-result ablations.

The results are not overstated. Overall accuracy is modest because the task is
hard and the `HOLD` class is nontrivial. But the improvement over naive LLM
prompting and the mechanism-level audit are enough to support the central
engineering claim: structured historical memory is useful for the MarketLens
pipeline.

---

## 13. Threats To Validity

### 13.1 Single-Asset Scope

The experiments are BTC-centric. Results may change for assets with lower
liquidity, different narrative cycles, or weaker regime persistence.

### 13.2 Limited Test Size

The official test has 305 rows. This is sufficient for a graduation-project
application study, but statistical power is limited. McNemar significance is
borderline for the primary structured winner.

### 13.3 Objective Mismatch

Retrieval metrics and downstream decision metrics do not always move together.
This is not a bug, but it complicates claims. The paper must state whether a
method is better as evidence retrieval, classifier support, or trading policy.

### 13.4 Naive LLM Baseline Stability

The LLM baseline depends on provider behavior, prompt style, and rate limits.
The evaluator now supports resume and retry behavior, but the cleanest future
baseline would freeze one prompt, one model, one decoding configuration, and a
complete 305-row output.

### 13.5 Local Data Dependency

The clean branch does not commit the NDJSON dataset. Reproduction requires
placing the export at the expected path or passing `--dataset`.

### 13.6 Prompt And Provider Drift

The naive LLM baseline uses an external provider. Even with temperature,
max-token, and prompt settings fixed, provider-side model updates can change
outputs. This is a weaker reproducibility surface than fixed kNN retrieval.
For a final thesis artifact, the report should include the frozen compact
tables and should treat any future LLM rerun as a new experiment unless the
model and prompt are exactly matched.

### 13.7 Threshold Sensitivity

The D7 label threshold is `+/-2%`. This is reasonable for crypto volatility,
but it is still a design choice. A different threshold could alter class
balance and the apparent value of `HOLD`. Future work should include a
threshold sensitivity table, for example:

```text
tau in {1%, 2%, 3%, 5%}
```

and report whether the ranking of methods changes.

### 13.8 Regime Dependence

BTC regimes can shift quickly. A method that works during a volatile sideways
period may not behave the same in a strong bull or bear regime. The learned
head may be partly learning the regime distribution of the validation split.
This is why rolling validation and repeated test windows are important next
steps.

---

## 14. Implementation And Reproducibility Details

### 14.1 Official Code Paths

The clean branch keeps official entrypoints separate from archived research
scripts:

| Script | Purpose |
| --- | --- |
| `aihub/scripts/evaluate_naive_llm_baseline.py` | Naive LLM baseline. |
| `stockmem/scripts/evaluate_stockmem_strict_models.py` | Primary structured model comparison. |
| `stockmem/scripts/evaluate_stockmem_feature_ablation.py` | Feature-block ablation. |
| `stockmem/scripts/export_stockmem_report_tables.py` | Compact table export. |
| `stockmem/scripts/run_submission_reproduction.py` | End-to-end reproduction runner. |

Experimental hybrid and head-aligned scripts live under
`stockmem/scripts/experimental/`. Historical top-level scripts live under
`scripts/archive/`.

### 14.2 Output Policy

Generated artifacts are not part of the clean submission commit:

```text
artifacts/
results_tables/
submission/
data/exports/
data/backtests/
stockmem/data/
```

This keeps the repository readable. The code can still regenerate compact
tables when the dataset is supplied.

### 14.3 Reproduction Runner

The reproduction runner executes:

1. strict structured model evaluation;
2. feature-block ablation;
3. optional naive LLM evaluation;
4. compact Markdown/CSV table export;
5. manifest generation with dataset checksum.

When `--skip-llm` is used, the runner requires an existing naive LLM summary:

```text
--naive-summary artifacts/current_context_ai_eval/summary.json
```

This prevents silently mixing missing LLM outputs with structured metrics.

### 14.4 Expected Submission Bundle

The generated bundle contains:

```text
submission/stockmem_2026_07/
  manifest.json
  learned_strict_test/summary.json
  fixed_knn_component_ablation/summary.json
  current_context_ai_eval/summary.json       # if LLM rerun is enabled
  tables/
    primary_structured_models.md
    naive_llm_vs_stockmem.md
    feature_ablation.md
    paired_stat_tests.md
```

The thesis should use the compact tables, not raw per-row JSONL logs.

---

## 15. Future Work

### 15.1 Paper-Clean LLM Baseline

The LLM baseline should be frozen with:

- one model;
- one prompt style;
- one temperature setting;
- complete 305-row output;
- no default-to-`HOLD` behavior on failure.

This would make the naive LLM comparison more reproducible.

### 15.2 Multi-Window Evaluation

The strongest next validation step is to repeat the strict test over multiple
non-overlapping held-out windows. This would determine whether the learned-head
advantage survives regime changes.

### 15.3 Retuned Feature-Block Weights

The feature ablation suggests that the current full fixed weights are not
optimal for the strict classifier objective. A future experiment should retune
block weights specifically for the strict held-out protocol, using validation
only, then test whether the full block combination can beat single-block
variants.

### 15.4 Better Learned Retrieval Objective

The learned retriever should not be retrained only to match semantic or event
similarity. A better objective would optimize:

```text
retrieval relevance + head decision quality + regime robustness
```

without allowing event-block collapse. Constraints on block weights or
regularization against single-block dominance may help.

### 15.5 Multi-Asset Generalization

The StockMem method should be tested on assets beyond BTC. If fixed kNN remains
strong across assets, it strengthens the memory-based thesis. If learned
retrieval helps more on smaller assets, it would clarify when semantic/event
learning is most valuable.

---

## 16. Practical Implications

For production and thesis presentation, the recommended pipeline is:

```text
current market/news context
        |
        v
structured StockMem record
        |
        v
fixed weighted kNN retrieval
        |
        v
top-5 historical evidence
        |
        v
learned stable decision head
        |
        v
BUY / HOLD / SELL
```

This pipeline is defensible because:

1. the retrieval stage is deterministic and interpretable;
2. the evidence pool is leakage-controlled;
3. the decision head is validation-selected;
4. the held-out test improves over naive LLM prompting;
5. negative experiments show that more complex retrievers were tested but not
   blindly adopted.

---

## 17. Conclusion

StockMem provides a structured historical-memory layer for crypto market
decision support. The experiments show that a naive LLM given only current
market and news context underperforms the structured StockMem pipeline on the
shared held-out test. They also show that fixed kNN retrieval is a strong
baseline and should not be dismissed as a weak heuristic. The best strict
structured variant uses fixed kNN retrieval with a learned stable decision
head, indicating that the main current gain comes from evidence aggregation
rather than learned retriever replacement.

The broader academic lesson is that retrieval-augmented market systems should
evaluate mechanisms separately. Evidence retrieval, semantic similarity,
decision aggregation, and trading policy are not the same objective. StockMem's
current evidence supports a conservative and useful design: keep fixed kNN as
the robust retrieval engine, use the learned stable head for decision support,
and treat learned retrieval and hybrid reranking as research directions until
they beat fixed kNN under the primary evidence target.

---

## Appendix A. Maintained Reproduction Commands

Full reproduction:

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /bin/sh \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python stockmem/scripts/run_submission_reproduction.py \
    --dataset data/exports/stockmem_records.ndjson \
    --out-dir submission/stockmem_2026_07"
```

Structured-only reproduction without LLM rerun:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /bin/sh \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python stockmem/scripts/run_submission_reproduction.py \
    --dataset data/exports/stockmem_records.ndjson \
    --out-dir submission/stockmem_2026_07 \
    --skip-llm \
    --naive-summary artifacts/current_context_ai_eval/summary.json"
```

---

## Appendix B. Report-Ready Tables

The compact table exporter writes:

- `primary_structured_models.md`
- `naive_llm_vs_stockmem.md`
- `feature_ablation.md`
- `paired_stat_tests.md`

and matching `.csv` files.

---

## Appendix C. Mathematical Summary

This appendix collects the main definitions in one place. It is written so the
method can be reimplemented without reading the full narrative.

### C.1 Records, Labels, And Matured Pools

Let the ordered StockMem dataset be:

```text
D = {z_1, z_2, ..., z_N}
```

Each daily record has:

```text
z_t = (date_t, F_t, I_t, P_t, E_t, R_t)
```

where:

- `F_t` is the factor vector;
- `I_t` is the indicator vector;
- `P_t` is the price-path vector;
- `E_t` is the optional event vector;
- `R_t` contains realized future returns.

The D7 target return is:

```text
r_t = future_return_7d(t)
```

The three-class label is:

```text
y_t =
  BUY   if r_t >  tau
  SELL  if r_t < -tau
  HOLD  otherwise
```

with:

```text
tau = 2.0 percentage points
```

For query day `t`, the eligible evidence pool is:

```text
M_t = {z_c in D : date_c + 7 days <= date_t}
```

The maturity constraint is part of the model definition. A system that removes
this constraint is solving a different problem because it can retrieve records
whose future D7 returns would not yet be known.

### C.2 Block Normalization

Each vector block is normalized before cosine scoring. For a block vector
`x_b`, the normalized vector is:

```text
x'_b = x_b / max(||x_b||_2, epsilon)
```

where `epsilon` is a small numerical guard. Cosine similarity is:

```text
cos(x, y) = (x dot y) / (||x||_2 ||y||_2)
```

If the vectors are already normalized, cosine similarity becomes a dot product.

### C.3 Fixed kNN Retriever

The fixed retriever computes:

```text
s_fixed(t,c) =
  w_f * cos(F_t, F_c)
  + w_i * cos(I_t, I_c)
  + w_p * cos(P_t, P_c)
```

The official weights are:

```text
w_f = 0.5443920554
w_i = 0.3090805325
w_p = 0.1415662727
```

The ranking is:

```text
R_fixed(t) = argsort_{c in M_t} s_fixed(t,c) descending
```

and the evidence set is:

```text
E_k(t) = top_k(R_fixed(t))
```

The deterministic property follows from this definition. If `D`, `t`, the
vectors, and weights are unchanged, the output ranking is unchanged.

### C.4 Learned Diagonal Retriever

The learned retriever generalizes fixed kNN by learning dimension and block
weights:

```text
s_learned(t,c) =
  sum_{b in B} alpha_b * cos(D_b X_{t,b}, D_b X_{c,b})
```

with constraints:

```text
alpha_b >= 0
sum_b alpha_b = 1
D_b is diagonal and non-negative
```

This form is intentionally simpler than a deep neural retriever. It changes
the retrieval geometry while preserving interpretability: high diagonal values
identify important dimensions, and high `alpha_b` values identify important
blocks.

### C.5 Hybrid Reranker

The hybrid method defines a two-stage retrieval function:

```text
C_30(t) = top_30(R_fixed(t))
```

then:

```text
s_hybrid(t,c) =
  w_knn * s_fixed(t,c)
  + w_lrn * s_learned(t,c)
  + w_reg * s_regime(t,c)
  + w_prior * s_prior(t,c)
```

for `c in C_30(t)`, with:

```text
w_knn + w_lrn + w_reg + w_prior = 1
w_* >= 0
```

The final evidence set is:

```text
E_5(t) = top_5(argsort_{c in C_30(t)} s_hybrid(t,c))
```

The key methodological point is that the reranker is not allowed to introduce
new candidates outside the fixed-kNN top-30 pool. This tests whether learned
signals improve ordering, not whether they replace candidate generation.

### C.6 Learned Stable Head

For a retrieved candidate `c`, the horizon-weighted historical outcome is:

```text
H(c) =
  beta_1  r_1(c)
  + beta_3  r_3(c)
  + beta_7  r_7(c)
  + beta_15 r_15(c)
  + beta_30 r_30(c)
```

The official learned stable head uses:

```text
beta = [0.0161, 0.1459, 0.4549, 0.1005, 0.2827]
```

For top-5 evidence:

```text
H_bar(t) = (1/5) * sum_{c in E_5(t)} H(c)
```

The prediction rule is:

```text
y_hat_t =
  BUY   if H_bar(t) >  1.45
  SELL  if H_bar(t) < -1.47
  HOLD  otherwise
```

This makes the head a validation-selected scoring rule. It is learned in the
same sense that a tuned linear decision rule is learned, not in the sense of a
large neural classifier.

### C.7 Evaluation Metrics

Overall accuracy:

```text
Acc = (1/N) sum_t 1[y_hat_t = y_t]
```

Active coverage:

```text
Coverage = (1/N) sum_t 1[y_hat_t in {BUY, SELL}]
```

Active accuracy:

```text
ActiveAcc =
  sum_t 1[sign_active(y_hat_t) = sign(r_t)]
  /
  sum_t 1[y_hat_t in {BUY, SELL}]
```

where active signs ignore `HOLD`.

Hit@5 same sign:

```text
Hit@5(t) =
  1 if exists c in E_5(t) such that y_c = y_t
  0 otherwise
```

Average Hit@5:

```text
Hit@5 = (1/N) sum_t Hit@5(t)
```

nDCG@5:

```text
DCG@5 = sum_{r=1}^{5} rel_r / log2(r + 1)
nDCG@5 = DCG@5 / IDCG@5
```

The report keeps retrieval metrics separate from decision metrics because a
retriever can improve evidence quality without improving the final head, and a
head can improve classification without changing evidence quality.

### C.8 Paired Model Comparison

For two models `A` and `B`, paired correctness is:

```text
a_t = 1[y_hat^A_t = y_t]
b_t = 1[y_hat^B_t = y_t]
```

The paired metric delta is:

```text
Delta = mean_t b_t - mean_t a_t
```

Bootstrap confidence intervals sample the set of dates with replacement and
recompute `Delta`. McNemar's test counts only discordant cases:

```text
n_01 = count(a_t = 0, b_t = 1)
n_10 = count(a_t = 1, b_t = 0)
```

The exact test asks whether the two discordant directions are equally likely
under the null hypothesis:

```text
H0: P(n_01) = P(n_10)
```

This is more appropriate than comparing independent accuracies because both
models are evaluated on the same dates.

---

## Appendix D. Claim-To-Artifact Traceability

This appendix is the practical checklist for maintaining the paper.

| Paper claim | Required artifact | Required table | Update rule |
| --- | --- | --- | --- |
| StockMem beats naive LLM. | `current_context_ai_eval/summary.json` | Naive LLM versus StockMem | Update if prompt, model, provider, or retry policy changes. |
| Fixed kNN is the robust retriever. | strict model comparison and hybrid retrieval outputs | Primary structured model comparison and hybrid table | Update if retrieval weights or pool size changes. |
| Learned head is the strict-test winner. | `learned_strict_test_v3/summary.json` | Primary structured model comparison | Update if learned-head grid or validation split changes. |
| Learned retrieval helps diagnostics. | strict comparison, hybrid outputs | Hit@5 and nDCG rows | Keep as diagnostic unless final decision score also wins. |
| Hybrid reranking is not yet superior. | `hybrid_retrieval_frozen_v2` | Hybrid reranking table | Re-evaluate if fusion grid or learned score source changes. |
| Feature blocks are useful but not individually necessary. | `fixed_knn_component_ablation/summary.json` | Feature-block ablation | Do not claim monotonic necessity unless future retuning supports it. |

The intended update process is:

1. Regenerate artifacts.
2. Export compact tables.
3. Update the claim ledger.
4. Update the interpretation text.
5. Commit docs separately from generated artifacts.

This process keeps the submission branch clean and makes the report resilient
to later reruns.

---

## Appendix E. Paper Outline For Thesis Conversion

If this Markdown report is converted into a formal thesis chapter or paper,
the recommended structure is:

1. **Introduction**
   State the problem: current-context LLM prompting is weak because it lacks
   historical evidence and stable audit trails.
2. **Related Work**
   Cover financial language models, financial RAG, event memory, metric
   learning, hybrid retrieval, and time-aware validation.
3. **System Architecture**
   Describe MarketLens, StockMem, daily records, point-in-time constraints,
   and the role of retrieval in the pipeline.
4. **Method**
   Formalize labels, fixed kNN, learned diagonal retrieval, hybrid reranking,
   and learned stable head.
5. **Experimental Protocol**
   Define train/validation/test windows, maturity guard, metrics, confidence
   intervals, and McNemar tests.
6. **Results**
   Present naive LLM comparison, structured model comparison, feature ablation,
   hybrid reranking, and head-aligned negative result.
7. **Discussion**
   Explain why StockMem works, why fixed kNN remains strong, why learned head
   helps, and why learned retrieval is not yet the production winner.
8. **Threats To Validity**
   Discuss BTC-only scope, test size, threshold sensitivity, prompt drift, and
   regime dependence.
9. **Conclusion**
   State the final applied claim: structured historical memory improves the
   MarketLens decision pipeline and provides a reproducible evidence trail.

The report should avoid claiming that the system is a profitable trading
strategy. The supported claim is narrower and stronger: the structured memory
pipeline improves directional decision support and auditability under the
current D7 evaluation.

---

## References

[1] Dogu Araci. *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv:1908.10063, 2019. https://arxiv.org/abs/1908.10063

[2] Sylvain Arlot and Alain Celisse. *A survey of cross-validation procedures for model selection*. arXiv:0907.4728, 2009. https://arxiv.org/abs/0907.4728

[3] Sebastian Bruch, Siyu Gai, and Amir Ingber. *An Analysis of Fusion Functions for Hybrid Retrieval*. arXiv:2210.11934, 2022. https://arxiv.org/abs/2210.11934

[4] Marius Koeppel, Alexander Segner, Martin Wagener, Lukas Pensel, Andreas Karwath, and Stefan Kramer. *Pairwise Learning to Rank by Neural Networks Revisited: Reconstruction, Theoretical Analysis and Practical Performance*. arXiv:1909.02768, 2019. https://arxiv.org/abs/1909.02768

[5] Bharath K. Sriperumbudur and Gert R. G. Lanckriet. *Metric Embedding for Nearest Neighbor Classification*. arXiv:0706.3499, 2007. https://arxiv.org/abs/0706.3499

[6] FinSeer financial time-series retrieval-augmented generation reference material. arXiv:2502.05878. Local copy retained under `docs/upgrade/FinSeer financial time-series RAG/`.

[7] StockMem event-reflection memory reference. arXiv:2512.02720. Local copy retained under `docs/upgrade/StockMem event-reflection memory.pdf`.

[8] Dissemination-aware FinGPT reference. OpenReview: https://openreview.net/forum?id=l2nHuTk6nc

[9] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. arXiv:2211.14730, 2022. https://arxiv.org/abs/2211.14730

[10] Matthew J. Page, Joanne E. McKenzie, Patrick M. Bossuyt, Isabelle Boutron, Tammy C. Hoffmann, Cynthia D. Mulrow, et al. *The PRISMA 2020 statement: an updated guideline for reporting systematic reviews*. BMJ 372:n71, 2021. https://doi.org/10.1136/bmj.n71
