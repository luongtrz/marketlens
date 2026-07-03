# StockMem Methodology

StockMem is the historical-memory component of MarketLens. For a current market
snapshot, it retrieves prior days that can act as structured evidence for a D7
directional decision. The final evaluation uses `BUY`, `HOLD`, and `SELL`
labels derived from `future_return_7d` with a `±2%` threshold.

The maintained evidence retriever is now trend-aware learned memory:

```text
learned_recency_50_50
```

This is distinct from the older strict decision-head table. The evidence
retriever is selected by `majority_same@10`, then a lightweight decision head is
selected on validation over the retrieved top-10 evidence.

## Feature Blocks

The core fixed-kNN representation uses three normalized market-state blocks:

| Block | Dimension | Role |
| --- | ---: | --- |
| `factor_vec` | 75 | Factor/event taxonomy and market context. |
| `indicator_vec` | 5 | Compact indicator and sentiment-related features. |
| `price_vec` | 60 | Recent price, volume, and range dynamics. |

Learned retrieval paths can also include an `event_vec` block. The strict
experiments separate retrieval quality from downstream decision quality because
the best evidence ranking is not always the best classifier head.

## Fixed kNN Retriever

The fixed baseline computes weighted cosine similarity:

```text
score(q,c) =
  w_factor * cos(factor_q, factor_c)
  + w_indicator * cos(indicator_q, indicator_c)
  + w_price * cos(price_q, price_c)
```

The tuned fixed weights used by the official evaluation are:

```text
w_factor    = 0.5443920554
w_indicator = 0.3090805325
w_price     = 0.1415662727
```

This is deterministic at inference: the same query, candidate pool, and weights
produce the same ranking.

## Learned Retriever

The learned retriever uses a diagonal metric over feature blocks:

```text
score(q,c) = sum_b alpha_b * cos(D_b q_b, D_b c_b)
```

`D_b` reweights dimensions and `alpha_b` controls block scales. This follows the
metric-adaptation view of nearest-neighbor methods: changing the metric can be
more important than changing the nearest-neighbor classifier itself.

The current learned FinBERT artifact is useful diagnostically, but the strict
test shows that replacing the fixed retriever alone does not clearly improve the
decision metric.

## Trend-Aware Learned-Memory Retriever

The majority-consensus audit showed that `Hit@5_same_sign` was too permissive:
it only asks whether at least one retrieved record shares the query's D7 class.
The stricter evidence target is:

```text
majority_same@10 = 1 if count_same_D7(top_10) >= 5
```

Under this objective, pure fixed and pure learned similarity are not the best
retrievers. D7 direction has strong temporal persistence, so the retriever must
be aware of trend continuity without collapsing into recency alone.

The maintained evidence retriever is:

```text
score(q,c) =
  0.5 * s_learned(q,c)
  + 0.5 * exp(-age_days(q,c) / 21)
```

The config artifact is:

```text
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

This model keeps learned historical similarity while using recency as a trend
awareness signal. It is the current recommended evidence retriever because it
has the best held-out and full-history `majority_same@10` among maintained
variants.

Important limitation: recency-heavy evidence can fail during market reversals.
The next methodological improvement should be a learned gate that decides when
to trust recent trend evidence and when to favor older historical analogs.

## Consensus Decision Head

The current recommended decision head is:

```text
count_vote_buy3_sell4
```

over the top-10 records returned by `learned_recency_50_50`.

Each retrieved record is first labeled by its matured `future_return_7d`:

```text
BUY  if future_return_7d > +2%
SELL if future_return_7d < -2%
HOLD otherwise
```

The head then applies:

```text
SELL if sell_count >= 4 and sell_count >= buy_count
BUY  if buy_count  >= 3 and buy_count  >  sell_count
HOLD otherwise
```

This head is deterministic and validation-selected. On the held-out test split
(`2025-07-01` to `2026-05-01`, `n=305`), it reaches `0.5475` overall accuracy,
`0.6826` active accuracy, `0.9607` active coverage, and `0.7114` SELL DA.

The important limitation is HOLD behavior: HOLD DA is `0.0000` in the current
audit. Therefore, the maintained model should be described as a high-coverage
directional head over structured historical evidence, not a balanced three-way
classifier.

## Older Learned Stable Head

The older strongest strict structured variant used the fixed retriever with a
tuned stable decision head. This head aggregates neighbor future returns across
multiple horizons:

| Horizon | Weight |
| --- | ---: |
| `1d` | 0.0161 |
| `3d` | 0.1459 |
| `7d` | 0.4549 |
| `15d` | 0.1005 |
| `30d` | 0.2827 |

The learned head thresholds are:

```text
BUY  if weighted average >  1.45
SELL if weighted average < -1.47
HOLD otherwise
```

This head is validation-selected, not a separately trained neural network. It is
kept as a baseline because it had better HOLD behavior, but it is no longer the
recommended primary decision head after the consensus-head audit.

## Hybrid And Head-Aligned Audits

Hybrid reranking was tested as a two-stage design: fixed kNN generates
candidates, then a convex fusion score reranks them. It is academically
defensible and interpretable, but it did not beat the strongest structured
baseline on the main strict classifier table.

Head-aligned retraining was also tested by training a retriever against the
frozen learned head. It overfit validation and collapsed almost entirely onto
the event block, so it is kept as a negative result rather than the recommended
production path.

## Leakage Control

All official retrieval evaluations use matured historical pools. A candidate
day must be at least seven days older than the query day before its D7 return is
eligible as evidence. The official split is:

| Split | Date range | Rows |
| --- | --- | ---: |
| Train | through `2024-12-24` | 2363 |
| Validation | `2025-01-01` to `2025-06-23` | 174 |
| Test | `2025-07-01` to `2026-05-01` | 305 |

The final claim uses the untouched test split only.

For the full-history retrieval audit, the query range is `2018-01-01` to
`2026-06-08`; the evaluator skips early rows until at least 10 matured
candidates are available.
