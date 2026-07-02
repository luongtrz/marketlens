# StockMem Methodology

StockMem is the historical-memory component of MarketLens. For a current market
snapshot, it retrieves prior days that can act as structured evidence for a D7
directional decision. The final evaluation uses `BUY`, `HOLD`, and `SELL`
labels derived from `future_return_7d` with a `±2%` threshold.

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

## Learned Stable Head

The strongest strict structured variant uses the fixed retriever with a tuned
stable decision head. This head aggregates neighbor future returns across
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

This head is validation-selected, not a separately trained neural network.

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
