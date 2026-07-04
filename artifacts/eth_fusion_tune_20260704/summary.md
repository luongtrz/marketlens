# Majority-Consensus Retriever Training

- Data: `data/exports/stockmem_records_eth.ndjson`
- Learned metric: `stockmem/config/learned_retriever_finbert.eth.json`
- Fixed weights: `stockmem/config/weights.auto.json`
- Top-k: `10`
- Label threshold: `±2.00%` on `future_return_7d`
- Selection split: validation
- Test split: held out
- Constraints: `{'min_memory_weight': 0.5, 'max_recency_weight': 0.7, 'min_learned_weight': 0.3, 'exclude_recent_days': 0}`

## Selected Config

```json
{
  "w_fixed": 0.2,
  "w_learned": 0.3,
  "w_recency": 0.5,
  "w_regime": 0.0,
  "recency_half_life_days": 30.0
}
```

## Validation

| Objective | Hit@k | Majority | Mean same | SELL majority | BUY majority | HOLD weighted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5770 | 0.8276 | 0.5632 | 5.0977 | 0.8495 | 0.3019 | 0.1679 |

## Test Comparison

| Retriever | Hit@k | Majority | Mean same | Weighted same | BUY majority | HOLD majority | SELL majority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 0.9770 | 0.2852 | 3.5934 | 0.3593 | 0.2923 | 0.0000 | 0.3577 |
| `learned_only` | 0.9738 | 0.2918 | 3.5607 | 0.3561 | 0.3077 | 0.0000 | 0.3577 |
| `recency_only` | 0.9016 | 0.5410 | 5.4295 | 0.5430 | 0.6308 | 0.1053 | 0.5766 |
| `regime_only` | 0.8656 | 0.4820 | 4.2000 | 0.4200 | 0.6692 | 0.0000 | 0.4380 |
| `fixed_recency_50_50` | 0.8951 | 0.5311 | 5.3607 | 0.5361 | 0.6308 | 0.0263 | 0.5766 |
| `learned_recency_50_50` | 0.9115 | 0.5246 | 5.3803 | 0.5380 | 0.6154 | 0.0263 | 0.5766 |
| `memory_first_learned_recency_60_30_10` | 0.9279 | 0.5672 | 5.1377 | 0.5138 | 0.6692 | 0.0000 | 0.6277 |
| `memory_first_fixed_recency_60_30_10` | 0.9213 | 0.5279 | 4.9213 | 0.4921 | 0.6615 | 0.0000 | 0.5474 |
| `balanced_fixed_learned_recency_regime` | 0.9180 | 0.5574 | 5.0361 | 0.5036 | 0.6923 | 0.0000 | 0.5839 |
| `selected_majority_consensus` | 0.9082 | 0.5279 | 5.3672 | 0.5367 | 0.6231 | 0.0263 | 0.5766 |

## Composite Test Comparison

| Retriever | Recent slots | Hit@k | Majority | Mean same | BUY majority | HOLD majority | SELL majority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected_memory_plus_2_recent` | 2 | 0.9082 | 0.5279 | 5.3639 | 0.6231 | 0.0263 | 0.5766 |
| `selected_memory_plus_3_recent` | 3 | 0.9082 | 0.5311 | 5.3705 | 0.6231 | 0.0526 | 0.5766 |
| `learned_memory_plus_2_recent` | 2 | 0.9836 | 0.3869 | 3.8754 | 0.3769 | 0.0000 | 0.5036 |
| `learned_memory_plus_3_recent` | 3 | 0.9803 | 0.4361 | 4.0754 | 0.4846 | 0.0000 | 0.5109 |
| `balanced_memory_plus_2_recent` | 2 | 0.9738 | 0.3508 | 3.8361 | 0.3846 | 0.0000 | 0.4161 |
| `balanced_memory_plus_3_recent` | 3 | 0.9705 | 0.3902 | 4.0098 | 0.4308 | 0.0000 | 0.4599 |

## Top Validation Candidates

| Rank | Objective | Config | Val majority | Val SELL majority | Val mean same |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0.5770 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 30.0}` | 0.5632 | 0.8495 | 5.0977 |
| 2 | 0.5765 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 30.0}` | 0.5632 | 0.8387 | 5.1494 |
| 3 | 0.5750 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 30.0}` | 0.5575 | 0.8387 | 5.1724 |
| 4 | 0.5738 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 14.0}` | 0.5575 | 0.8280 | 5.1552 |
| 5 | 0.5721 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 45.0}` | 0.5575 | 0.8495 | 5.0230 |
| 6 | 0.5706 | `{'w_fixed': 0.0, 'w_learned': 0.6, 'w_recency': 0.3, 'w_regime': 0.1, 'recency_half_life_days': 30.0}` | 0.5575 | 0.8495 | 4.9540 |
| 7 | 0.5704 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 45.0}` | 0.5517 | 0.8387 | 5.0920 |
| 8 | 0.5699 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 21.0}` | 0.5517 | 0.8280 | 5.1494 |
| 9 | 0.5695 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 45.0}` | 0.5517 | 0.8280 | 5.1322 |
| 10 | 0.5693 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 60.0}` | 0.5517 | 0.8495 | 5.0345 |
| 11 | 0.5690 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 14.0}` | 0.5517 | 0.8172 | 5.1264 |
| 12 | 0.5689 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 30.0}` | 0.5517 | 0.8387 | 4.9885 |
| 13 | 0.5689 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 45.0}` | 0.5517 | 0.8280 | 5.0287 |
| 14 | 0.5689 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 21.0}` | 0.5517 | 0.8280 | 5.0460 |
| 15 | 0.5681 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 21.0}` | 0.5517 | 0.8172 | 5.1609 |
| 16 | 0.5678 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 21.0}` | 0.5517 | 0.8172 | 5.0862 |
| 17 | 0.5666 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 45.0}` | 0.5517 | 0.8280 | 4.9770 |
| 18 | 0.5664 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 60.0}` | 0.5517 | 0.8495 | 4.9425 |
| 19 | 0.5662 | `{'w_fixed': 0.1, 'w_learned': 0.5, 'w_recency': 0.3, 'w_regime': 0.1, 'recency_half_life_days': 30.0}` | 0.5517 | 0.8495 | 4.9310 |
| 20 | 0.5654 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 21.0}` | 0.5460 | 0.8172 | 5.1322 |
