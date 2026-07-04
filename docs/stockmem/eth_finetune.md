# ETH Fine-Tune StockMem Report

This report records the ETH-specific StockMem fine-tuning run after the
zero-shot evaluation.

## Trained Artifacts

| Artifact | Path |
| --- | --- |
| ETH vectorized optimizer dataset | `stockmem/data/real_optimizer_finbert_eth.json` |
| ETH learned retriever | `stockmem/config/learned_retriever_finbert.eth.json` |
| ETH fixed-kNN weights | `stockmem/config/weights.eth.auto.json` |
| ETH pure learned+recency profile | `stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json` |
| ETH diagnostic tuned fusion profile | `stockmem/config/majority_consensus_retriever.eth.tuned_eth_weights.json` |

The model profile map is:

```text
stockmem/config/model_profiles.json
```

## ETH Learned Retriever Training

Source log:

```text
artifacts/eth_finetune_20260704/train.log
```

The ETH learned retriever was initialized from the BTC learned artifact and
fine-tuned on ETH StockMem records.

Final training summary:

| Metric | Value |
| --- | ---: |
| Validation combined | 0.3045 |
| Validation hit@k | 0.9862 |
| Validation nDCG@k | 0.2909 |
| Seed std | 0.0108 |

## ETH Fixed-kNN Weight Tuning

Source:

```text
stockmem/config/weights.eth.auto.json
```

ETH fixed-kNN weights:

| Block | Weight |
| --- | ---: |
| `factor_vec` | 0.3488 |
| `indicator_vec` | 0.2885 |
| `price_vec` | 0.3627 |

This differs materially from BTC, where the fixed baseline is more
factor-heavy. ETH gives substantially more weight to the price block.

## Fusion Tuning

Source:

```text
artifacts/eth_fusion_tune_eth_weights_20260704/summary.json
```

Validation-selected fusion under the majority-consensus objective:

```json
{
  "w_fixed": 0.2,
  "w_learned": 0.3,
  "w_recency": 0.4,
  "w_regime": 0.1,
  "recency_half_life_days": 21.0
}
```

This gave the strongest evidence metric:

| Retriever | Test Majority@10 | Mean Same@10 | BUY Majority | SELL Majority |
| --- | ---: | ---: | ---: | ---: |
| `learned_recency_50_50` | 0.5279 | 5.3869 | 0.6231 | 0.5693 |
| `selected_majority_consensus` | **0.5508** | 5.2000 | **0.6538** | **0.6058** |

However, the selected fusion includes fixed and regime components. For the
maintained ETH endpoint, the cleaner learned-memory design is kept:

```text
eth_learned_recency_50_50_h30
```

This is validation-ranked close to the tuned fusion and keeps the same
mechanism as BTC:

```text
0.5 * ETH learned similarity + 0.5 * recency
```

with a 30-day recency half-life.

## Decision Head Selection

The ETH decision head was selected on validation over top-10 evidence.

| Candidate profile | Selected head | Val score | Test overall | Test active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero-shot BTC artifact + h21 | `mean_learned_weights_buy0.50_sell0.50` | 0.6550 | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |
| ETH learned h21 | `mean_learned_weights_buy0.50_sell0.75` | 0.7067 | **0.6033** | **0.6815** | 0.9574 | **0.7308** | 0.0263 | 0.6423 | 0.5279 |
| ETH learned h30 pure | `mean_learned_weights_buy0.50_sell0.75` | **0.7070** | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.0526 | **0.6496** | 0.5246 |
| ETH tuned fusion with BTC fixed weights | `mean_learned_weights_buy0.50_sell0.50` | 0.7025 | 0.5902 | 0.6782 | 0.9475 | 0.7000 | 0.0263 | 0.6423 | 0.5279 |
| ETH tuned fusion with ETH fixed weights | `mean_learned_weights_buy0.50_sell0.75` | 0.6984 | 0.5967 | 0.6678 | 0.9377 | 0.6923 | 0.1316 | 0.6350 | **0.5508** |

## Maintained ETH Profile

The maintained ETH endpoint profile is:

```text
retriever: eth_learned_recency_50_50_h30
head: mean_learned_weights_buy0.50_sell0.75
```

This choice is validation-driven and keeps the same mechanism as BTC while
allowing ETH-specific training. It gives:

| Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 174 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.0714 | 0.8817 | 0.5575 |
| Test | 305 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.0526 | 0.6496 | 0.5246 |

## Interpretation

ETH fine-tuning improves the final learned+recency pipeline substantially over
zero-shot:

```text
overall: 0.5344 -> 0.6000
active:  0.6014 -> 0.6793
SELL DA: 0.6204 -> 0.6496
```

The strict learned-only table did not improve after fine-tuning, so the claim
should not be "fine-tuning improves every learned metric." The defensible claim
is narrower and stronger:

```text
ETH fine-tuning improves the final StockMem evidence pipeline after recency
fusion and validation-selected head aggregation.
```

