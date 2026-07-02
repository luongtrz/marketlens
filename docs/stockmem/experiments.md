# StockMem Experiments

The maintained result set answers two practical questions:

1. Does the structured StockMem pipeline outperform naive current-context LLM
   prompting?
2. Which StockMem mechanism is strongest: fixed kNN, learned retrieval, learned
   head, or hybrid reranking?

All official tables use the held-out `2025-07-01` to `2026-05-01` split with
`305` rows and D7 labels from `future_return_7d` at `±2%`.

## Primary Structured Model Table

Source: `artifacts/learned_strict_test_v3/summary.json`.

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 |
| `fixed_retriever_learned_head` | 305 | **0.3508** | **0.4500** | **0.8525** | 0.8361 |
| `learned_retriever_fixed_head` | 305 | 0.3148 | 0.4182 | 0.7213 | **0.8459** |
| `learned_finbert_rolling_stable` | 305 | 0.3410 | 0.4393 | 0.7836 | **0.8459** |

Primary paired statistics against `fixed_knn_rolling_stable`:

| Challenger | Metric | Delta | 95% bootstrap CI | McNemar p |
| --- | --- | ---: | ---: | ---: |
| `fixed_retriever_learned_head` | overall_acc | +0.0330 | [+0.0000, +0.0689] | 0.087159 |
| `fixed_retriever_learned_head` | active_acc | +0.0264 | [+0.0013, +0.0529] | 0.087159 |
| `fixed_retriever_learned_head` | coverage | +0.1014 | [+0.0623, +0.1410] | 0.087159 |
| `learned_finbert_rolling_stable` | overall_acc | +0.0222 | [-0.0393, +0.0852] | 0.550709 |

Interpretation: the best strict structured classifier is fixed retrieval plus
the learned stable head. Learned retrieval improves Hit@5 but does not dominate
the decision metric.

## Naive LLM Baseline

Source: `artifacts/current_context_ai_eval/summary.json`.

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_current_ai` | 305 | 0.2787 | 0.4031 | 0.6426 | 0.8361 | 0.6164 | 0.3574 | 0.0262 |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 | 0.6295 | 0.2492 | 0.1213 |
| `knn_returns` | 305 | 0.2918 | 0.4146 | 0.6721 | 0.8361 | 0.5574 | 0.3279 | 0.1148 |

The naive LLM baseline receives current market context, recent price changes,
one-day news sentiment, and compact headlines. It does not receive historical
retrieval evidence. Its weak SELL rate is the main behavioral failure.

## Feature-Block Ablation

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

This weakens the claim that every feature block is individually necessary.
The safer claim is that the full pipeline is the official tuned configuration,
while individual blocks contain different useful signals.

## Hybrid And Head-Aligned Negative Results

Hybrid reranking clarified that fixed kNN remains a strong candidate generator.
The tuned hybrid improved some diagnostic scores but did not beat fixed kNN on
the primary strict decision target.

The head-aligned retriever training run overfit:

| Model | Overall Acc | Active Acc | Coverage | Hit@5 same sign |
| --- | ---: | ---: | ---: | ---: |
| head-aligned retriever + learned head | 0.2820 | 0.3942 | 0.7902 | 0.8459 |

Its artifact collapsed to event-block scale `0.9997`, which is evidence of a
validation shortcut rather than a production improvement.

## Practical Conclusion

The most defensible production recommendation is:

```text
fixed kNN retriever + learned stable decision head
```

The broader research conclusion is that StockMem is useful as structured
historical memory, but more learned retrieval is not automatically better.

For the full academic write-up, see [academic_paper.md](academic_paper.md).
