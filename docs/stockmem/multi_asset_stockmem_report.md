# Multi-Asset StockMem Report: BTC And ETH

This report summarizes the current StockMem result after extending the
BTC-focused pipeline to ETH. It is intended as a compact thesis/report section:
what the system is, how the mechanism works, what changed for ETH, and what the
numbers mean.

The report uses the maintained StockMem documentation and artifacts as source
of truth. Related references are listed in [References](../references.md).

## 1. Objective

The objective is to evaluate whether StockMem can operate as a structured
historical-memory layer across assets, not only as a BTC-specific retrieval
experiment.

The core forecasting target is the D7 directional class:

```text
BUY  if future_return_7d > +2%
SELL if future_return_7d < -2%
HOLD otherwise
```

The maintained design is asset-specific:

```text
BTC endpoint/profile: BTC-trained learned-memory retriever + BTC-selected head
ETH endpoint/profile: ETH-trained learned-memory retriever + ETH-selected head
```

The profile map is:

```text
stockmem/config/model_profiles.json
```

## 2. Mechanism

StockMem stores each daily market state as a structured memory record:

```text
date, symbol, market snapshot, event/factor fields,
event_vec, factor_vec, indicator_vec, price_vec,
future_return_1d/3d/7d/15d/30d
```

At inference time, a current query is compared against historical records from
the same asset. The learned retriever computes:

```text
learned_similarity(q, c)
```

where `q` is the current query and `c` is a historical candidate. This score is
not the fixed-kNN formula. It is a learned metric trained from StockMem records
so that historically relevant records are ranked closer.

The evidence retriever then combines learned memory with trend awareness:

```text
score(q,c) =
  w_learned * learned_similarity(q,c)
+ w_recency * recency(q,c)
```

where:

```text
recency(q,c) = exp(-age_days(q,c) / half_life_days)
```

The top-10 records after this scoring become the evidence set. A lightweight
decision head then aggregates the known D7 outcomes of those retrieved records.

This design follows three research ideas:

1. **Financial text and event representation.** FinBERT-style representations
   support domain-specific financial sentiment and event encoding [1].
2. **Metric retrieval.** Fixed kNN and learned retrieval are both metric
   retrieval mechanisms: they estimate whether a historical state is useful for
   the current query. The fixed model uses manually selected vector blocks,
   while the learned retriever estimates a similarity metric from StockMem
   records [5].
3. **Fusion with validation.** Combining semantic similarity with recency is a
   hybrid retrieval function. The weights are treated as validation-selected
   configuration, not as a purely manual belief [2,3].

## 3. Evaluation Protocol

Both BTC and ETH use the same held-out test window:

```text
2025-07-01 to 2026-05-01
```

The test split has:

```text
n = 305 rows
```

The validation split has:

```text
n = 174 rows
```

All retrieval uses matured historical pools: a candidate's D7 outcome must be
known before it can be used as evidence for a query.

The main evidence metric is:

```text
majority_same@10 =
  1 if at least 5 of the top-10 evidence records share the query D7 class
```

This is stricter than hit-style retrieval metrics because it requires a
directionally coherent evidence set, not merely one matching historical case.

The decision metrics are:

```text
overall_DA = correct D7 class predictions / all evaluated rows
active_DA  = correct BUY or SELL predictions / active BUY or SELL predictions
coverage   = non-HOLD predictions / all evaluated rows
BUY_DA     = class accuracy on true BUY rows
HOLD_DA    = class accuracy on true HOLD rows
SELL_DA    = class accuracy on true SELL rows
```

This is intentionally not optimized directly for backtest profit. The academic
claim is about structured historical evidence and D7 directional consistency.
Return metrics can be added as downstream analysis, but they are not the
primary evidence that the retrieval mechanism works.

## 4. BTC Maintained Profile

BTC's maintained evidence retriever is:

```text
learned_recency_50_50
```

Config:

```text
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

Mechanism:

```text
0.5 * BTC learned_similarity + 0.5 * recency(half_life=21d)
```

BTC-selected head:

```text
count_vote_buy3_sell4
```

BTC test result:

| Profile | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC learned-recency + count vote | 305 | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.0000 | 0.7114 | 0.5443 |

Interpretation: BTC's final pipeline is a high-coverage directional evidence
system. Its main strength is SELL recognition. Its main weakness is HOLD
classification.

## 5. ETH Zero-Shot Transfer

ETH was first evaluated using the existing BTC artifacts without ETH-specific
training.

Data:

```text
data/exports/stockmem_records_eth.ndjson
```

ETH export summary:

| Field | Value |
| --- | ---: |
| Raw rows | 2908 |
| Rows with matured D7 return | 2903 |
| Date range | `2018-01-05` to `2026-07-01` |

Zero-shot ETH with the BTC-maintained learned-recency pipeline:

| Profile | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC artifact on ETH | 305 | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |

Interpretation: zero-shot transfer was already useful. This supports the
claim that StockMem's learned memory mechanism is not BTC-only. However,
`majority@10` was weaker than BTC, so ETH-specific tuning was justified.

## 6. ETH Fine-Tuning

ETH fine-tuning produced:

```text
stockmem/config/learned_retriever_finbert.eth.json
```

The ETH fixed-kNN diagnostic weights were also tuned:

| Block | BTC fixed weight | ETH fixed weight |
| --- | ---: | ---: |
| `factor_vec` | 0.5444 | 0.3488 |
| `indicator_vec` | 0.3091 | 0.2885 |
| `price_vec` | 0.1416 | 0.3627 |

This shows that ETH relies more heavily on price-state similarity than the BTC
fixed-kNN baseline. This is useful diagnostically, even though the maintained
ETH endpoint remains a learned-recency profile.

## 7. ETH Fusion And Head Selection

Fusion tuning found that the strongest evidence-only diagnostic config included
fixed and regime components:

```json
{
  "w_fixed": 0.2,
  "w_learned": 0.3,
  "w_recency": 0.4,
  "w_regime": 0.1,
  "recency_half_life_days": 21.0
}
```

That config reached:

```text
test majority@10 = 0.5508
```

However, for endpoint consistency and interpretability, the maintained ETH
profile keeps the same learned-memory plus recency mechanism as BTC:

```text
eth_learned_recency_50_50_h30
```

Mechanism:

```text
0.5 * ETH learned_similarity + 0.5 * recency(half_life=30d)
```

ETH-selected decision head:

```text
mean_learned_weights_buy0.50_sell0.75
```

Maintained ETH test result:

| Profile | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH learned-recency h30 + mean head | 305 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.0526 | 0.6496 | 0.5246 |

## 8. Before And After ETH Fine-Tuning

| ETH profile | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot BTC artifact | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |
| ETH fine-tuned learned-recency h30 | **0.6000** | **0.6793** | 0.9508 | **0.7077** | 0.0526 | **0.6496** | **0.5246** |

Fine-tuning improved:

```text
overall:     +0.0656
active:      +0.0779
BUY DA:      +0.1308
SELL DA:     +0.0292
majority@10: +0.0492
```

The result is important because the improvement appears after the full
pipeline:

```text
ETH learned retriever
→ learned+recency evidence fusion
→ validation-selected decision head
```

The strict learned-only table did not improve in the same way. Therefore, the
correct claim is not that ETH fine-tuning improves every learned metric. The
correct claim is:

```text
ETH fine-tuning improves the final StockMem evidence pipeline.
```

## 9. BTC And ETH Side By Side

| Asset | Maintained retriever | Head | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | `learned_recency_50_50` | `count_vote_buy3_sell4` | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.0000 | **0.7114** | **0.5443** |
| ETH | `eth_learned_recency_50_50_h30` | `mean_learned_weights_buy0.50_sell0.75` | **0.6000** | 0.6793 | 0.9508 | **0.7077** | 0.0526 | 0.6496 | 0.5246 |

BTC has stronger SELL recognition and slightly stronger evidence consistency.
ETH has stronger overall accuracy and BUY recognition after fine-tuning.

Both profiles have weak HOLD recognition. This is expected because the
maintained heads behave more like high-coverage directional heads than balanced
three-class classifiers.

## 10. Endpoint Plan

The endpoint-level plan is:

```text
symbol=BTC
  learned_retriever_artifact:
    stockmem/config/learned_retriever_finbert.json
  retriever_config:
    stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
  head:
    count_vote_buy3_sell4

symbol=ETH
  learned_retriever_artifact:
    stockmem/config/learned_retriever_finbert.eth.json
  retriever_config:
    stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json
  head:
    mean_learned_weights_buy0.50_sell0.75
```

This makes StockMem a multi-asset system without forcing one global artifact to
serve all markets.

## 11. Conclusion

The ETH extension strengthens the StockMem thesis. The result is no longer only
"BTC historical memory works." It is now:

```text
Structured StockMem memory can transfer across assets, and asset-specific
fine-tuning improves the final evidence pipeline.
```

The most defensible final claim is:

1. StockMem beats naive current-context prompting on BTC.
2. BTC learned-recency retrieval provides a strong maintained evidence pipeline.
3. ETH zero-shot transfer is already useful.
4. ETH-specific fine-tuning improves the final learned-recency endpoint.
5. BTC and ETH should use separate StockMem profiles while sharing the same
   overall architecture.

Remaining caveat: HOLD classification is still weak. The current profiles are
best described as high-coverage directional decision systems, not balanced
three-class classifiers.

## 12. Citations

[1] Dogu Araci. *FinBERT: Financial Sentiment Analysis with Pre-trained
Language Models*. arXiv:1908.10063, 2019.

[2] Sylvain Arlot and Alain Celisse. *A survey of cross-validation procedures
for model selection*. arXiv:0907.4728, 2009.

[3] Sebastian Bruch, Siyu Gai, and Amir Ingber. *An Analysis of Fusion
Functions for Hybrid Retrieval*. arXiv:2210.11934, 2022.

[4] Marius Koeppel, Alexander Segner, Martin Wagener, Lukas Pensel, Andreas
Karwath, and Stefan Kramer. *Pairwise Learning to Rank by Neural Networks
Revisited*. arXiv:1909.02768, 2019.

[5] Bharath K. Sriperumbudur and Gert R. G. Lanckriet. *Metric Embedding for
Nearest Neighbor Classification*. arXiv:0706.3499, 2007.
