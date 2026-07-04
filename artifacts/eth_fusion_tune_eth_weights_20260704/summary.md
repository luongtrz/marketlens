# Majority-Consensus Retriever Training

- Data: `data/exports/stockmem_records_eth.ndjson`
- Learned metric: `stockmem/config/learned_retriever_finbert.eth.json`
- Fixed weights: `stockmem/config/weights.eth.auto.json`
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
  "w_recency": 0.4,
  "w_regime": 0.1,
  "recency_half_life_days": 21.0
}
```

## Validation

| Objective | Hit@k | Majority | Mean same | SELL majority | BUY majority | HOLD weighted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5836 | 0.8391 | 0.5747 | 5.0690 | 0.8602 | 0.3208 | 0.1786 |

## Test Comparison

| Retriever | Hit@k | Majority | Mean same | Weighted same | BUY majority | HOLD majority | SELL majority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 0.9607 | 0.3410 | 3.7344 | 0.3734 | 0.3615 | 0.0526 | 0.4015 |
| `learned_only` | 0.9738 | 0.2918 | 3.5607 | 0.3561 | 0.3077 | 0.0000 | 0.3577 |
| `recency_only` | 0.9016 | 0.5410 | 5.4295 | 0.5430 | 0.6308 | 0.1053 | 0.5766 |
| `regime_only` | 0.8656 | 0.4820 | 4.2000 | 0.4200 | 0.6692 | 0.0000 | 0.4380 |
| `fixed_recency_50_50` | 0.9049 | 0.5213 | 5.2885 | 0.5289 | 0.6077 | 0.0263 | 0.5766 |
| `learned_recency_50_50` | 0.9115 | 0.5279 | 5.3869 | 0.5387 | 0.6231 | 0.0526 | 0.5693 |
| `memory_first_learned_recency_60_30_10` | 0.9279 | 0.5410 | 5.0066 | 0.5007 | 0.6385 | 0.0000 | 0.5985 |
| `memory_first_fixed_recency_60_30_10` | 0.9607 | 0.4262 | 4.2459 | 0.4246 | 0.4615 | 0.0000 | 0.5109 |
| `balanced_fixed_learned_recency_regime` | 0.9377 | 0.4754 | 4.6066 | 0.4607 | 0.5308 | 0.0000 | 0.5547 |
| `selected_majority_consensus` | 0.9279 | 0.5508 | 5.2000 | 0.5200 | 0.6538 | 0.0000 | 0.6058 |

## Composite Test Comparison

| Retriever | Recent slots | Hit@k | Majority | Mean same | BUY majority | HOLD majority | SELL majority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected_memory_plus_2_recent` | 2 | 0.9279 | 0.5443 | 5.1967 | 0.6538 | 0.0000 | 0.5912 |
| `selected_memory_plus_3_recent` | 3 | 0.9279 | 0.5443 | 5.1934 | 0.6462 | 0.0263 | 0.5912 |
| `learned_memory_plus_2_recent` | 2 | 0.9836 | 0.3869 | 3.8754 | 0.3769 | 0.0000 | 0.5036 |
| `learned_memory_plus_3_recent` | 3 | 0.9803 | 0.4361 | 4.0754 | 0.4846 | 0.0000 | 0.5109 |
| `balanced_memory_plus_2_recent` | 2 | 0.9672 | 0.3770 | 3.8721 | 0.4154 | 0.0263 | 0.4380 |
| `balanced_memory_plus_3_recent` | 3 | 0.9738 | 0.4393 | 4.1049 | 0.4692 | 0.0263 | 0.5255 |

## Top Validation Candidates

| Rank | Objective | Config | Val majority | Val SELL majority | Val mean same |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0.5836 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 21.0}` | 0.5747 | 0.8602 | 5.0690 |
| 2 | 0.5833 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 30.0}` | 0.5747 | 0.8602 | 5.0517 |
| 3 | 0.5756 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 21.0}` | 0.5632 | 0.8387 | 5.0690 |
| 4 | 0.5750 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 30.0}` | 0.5575 | 0.8387 | 5.1724 |
| 5 | 0.5750 | `{'w_fixed': 0.3, 'w_learned': 0.3, 'w_recency': 0.4, 'w_regime': 0.0, 'recency_half_life_days': 30.0}` | 0.5632 | 0.8710 | 4.9540 |
| 6 | 0.5724 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 45.0}` | 0.5575 | 0.8495 | 5.0115 |
| 7 | 0.5723 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 14.0}` | 0.5575 | 0.8280 | 5.1552 |
| 8 | 0.5715 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 30.0}` | 0.5575 | 0.8387 | 5.1264 |
| 9 | 0.5715 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 21.0}` | 0.5575 | 0.8387 | 5.1264 |
| 10 | 0.5714 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 30.0}` | 0.5575 | 0.8280 | 5.1379 |
| 11 | 0.5706 | `{'w_fixed': 0.0, 'w_learned': 0.6, 'w_recency': 0.3, 'w_regime': 0.1, 'recency_half_life_days': 30.0}` | 0.5575 | 0.8495 | 4.9540 |
| 12 | 0.5699 | `{'w_fixed': 0.3, 'w_learned': 0.3, 'w_recency': 0.4, 'w_regime': 0.0, 'recency_half_life_days': 21.0}` | 0.5517 | 0.8495 | 4.9885 |
| 13 | 0.5695 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 45.0}` | 0.5517 | 0.8280 | 5.1322 |
| 14 | 0.5693 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 60.0}` | 0.5517 | 0.8495 | 5.0345 |
| 15 | 0.5689 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 45.0}` | 0.5517 | 0.8280 | 5.0287 |
| 16 | 0.5689 | `{'w_fixed': 0.2, 'w_learned': 0.3, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 45.0}` | 0.5575 | 0.8387 | 5.0460 |
| 17 | 0.5681 | `{'w_fixed': 0.0, 'w_learned': 0.5, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 21.0}` | 0.5517 | 0.8172 | 5.1609 |
| 18 | 0.5677 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 30.0}` | 0.5517 | 0.8280 | 5.0690 |
| 19 | 0.5673 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.5, 'w_regime': 0.0, 'recency_half_life_days': 21.0}` | 0.5517 | 0.8172 | 5.1437 |
| 20 | 0.5672 | `{'w_fixed': 0.1, 'w_learned': 0.4, 'w_recency': 0.4, 'w_regime': 0.1, 'recency_half_life_days': 60.0}` | 0.5517 | 0.8495 | 4.9713 |
