# Consensus Retriever Decision Head Evaluation

- Data: `data/exports/stockmem_records_eth.ndjson`
- Retriever config: `stockmem/config/majority_consensus_retriever.eth.tuned_validation.json`
- Top-k: `10`
- Label threshold: `±2.00%`

## Selected Head

- Head: `mean_learned_weights_buy0.50_sell0.50`
- Validation score: `0.7025`

## Comparison

| Model | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 | Mean Same@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | val | 174 | 0.6437 | 0.7294 | 0.9770 | 0.5472 | 0.0357 | 0.8817 | 0.5632 | 5.0977 |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | test | 305 | 0.5902 | 0.6782 | 0.9475 | 0.7000 | 0.0263 | 0.6423 | 0.5279 | 5.3672 |
| `fixed_knn_rolling_stable` | test_old_strict | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6327 | 0.2759 | 0.1275 | n/a | n/a |
| `fixed_retriever_learned_head` | test_old_strict | 305 | 0.3508 | 0.4500 | 0.8525 | 0.7041 | 0.1552 | 0.1946 | n/a | n/a |
| `learned_finbert_rolling_stable` | test_old_strict | 305 | 0.3410 | 0.4393 | 0.7836 | 0.5408 | 0.2414 | 0.2483 | n/a | n/a |

## Top Validation Heads

| Rank | Head | Score | Val Overall | Val Active | Val Coverage | Val BUY DA | Val SELL DA | Test Overall | Test Active | Test Coverage | Test BUY DA | Test SELL DA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mean_learned_weights_buy0.50_sell0.50` | 0.7025 | 0.6437 | 0.7294 | 0.9770 | 0.5472 | 0.8817 | 0.5902 | 0.6782 | 0.9475 | 0.7000 | 0.6423 |
| 2 | `mean_learned_weights_buy0.75_sell0.50` | 0.7025 | 0.6437 | 0.7294 | 0.9770 | 0.5472 | 0.8817 | 0.5803 | 0.6796 | 0.9311 | 0.6769 | 0.6423 |
| 3 | `mean_learned_weights_buy1.00_sell0.50` | 0.7025 | 0.6437 | 0.7294 | 0.9770 | 0.5472 | 0.8817 | 0.5803 | 0.6857 | 0.9180 | 0.6692 | 0.6423 |
| 4 | `mean_learned_weights_buy1.25_sell0.50` | 0.7025 | 0.6437 | 0.7294 | 0.9770 | 0.5472 | 0.8817 | 0.5836 | 0.6871 | 0.9115 | 0.6692 | 0.6423 |
| 5 | `mean_learned_weights_buy0.50_sell1.50` | 0.6980 | 0.6379 | 0.7455 | 0.9483 | 0.5472 | 0.8710 | 0.5803 | 0.6738 | 0.9148 | 0.7000 | 0.5985 |
| 6 | `mean_learned_weights_buy0.75_sell1.50` | 0.6980 | 0.6379 | 0.7455 | 0.9483 | 0.5472 | 0.8710 | 0.5705 | 0.6752 | 0.8984 | 0.6769 | 0.5985 |
| 7 | `mean_learned_weights_buy1.00_sell1.50` | 0.6980 | 0.6379 | 0.7455 | 0.9483 | 0.5472 | 0.8710 | 0.5705 | 0.6815 | 0.8852 | 0.6692 | 0.5985 |
| 8 | `mean_learned_weights_buy1.25_sell1.50` | 0.6980 | 0.6379 | 0.7455 | 0.9483 | 0.5472 | 0.8710 | 0.5738 | 0.6828 | 0.8787 | 0.6692 | 0.5985 |
| 9 | `mean_learned_weights_buy0.50_sell1.00` | 0.6980 | 0.6379 | 0.7410 | 0.9540 | 0.5472 | 0.8710 | 0.5869 | 0.6784 | 0.9279 | 0.7000 | 0.6204 |
| 10 | `mean_learned_weights_buy0.50_sell1.25` | 0.6980 | 0.6379 | 0.7410 | 0.9540 | 0.5472 | 0.8710 | 0.5770 | 0.6750 | 0.9180 | 0.7000 | 0.5985 |
| 11 | `mean_learned_weights_buy0.75_sell1.00` | 0.6980 | 0.6379 | 0.7410 | 0.9540 | 0.5472 | 0.8710 | 0.5770 | 0.6799 | 0.9115 | 0.6769 | 0.6204 |
| 12 | `mean_learned_weights_buy0.75_sell1.25` | 0.6980 | 0.6379 | 0.7410 | 0.9540 | 0.5472 | 0.8710 | 0.5672 | 0.6764 | 0.9016 | 0.6769 | 0.5985 |
| 13 | `mean_learned_weights_buy1.00_sell1.00` | 0.6980 | 0.6379 | 0.7410 | 0.9540 | 0.5472 | 0.8710 | 0.5770 | 0.6861 | 0.8984 | 0.6692 | 0.6204 |
| 14 | `mean_learned_weights_buy1.00_sell1.25` | 0.6980 | 0.6379 | 0.7410 | 0.9540 | 0.5472 | 0.8710 | 0.5672 | 0.6827 | 0.8885 | 0.6692 | 0.5985 |
| 15 | `mean_learned_weights_buy1.25_sell1.00` | 0.6980 | 0.6379 | 0.7410 | 0.9540 | 0.5472 | 0.8710 | 0.5803 | 0.6875 | 0.8918 | 0.6692 | 0.6204 |
| 16 | `mean_learned_weights_buy1.25_sell1.25` | 0.6980 | 0.6379 | 0.7410 | 0.9540 | 0.5472 | 0.8710 | 0.5705 | 0.6840 | 0.8820 | 0.6692 | 0.5985 |
| 17 | `mean_learned_weights_buy0.50_sell0.75` | 0.6980 | 0.6379 | 0.7365 | 0.9598 | 0.5472 | 0.8710 | 0.5902 | 0.6760 | 0.9410 | 0.7000 | 0.6350 |
| 18 | `mean_learned_weights_buy0.75_sell0.75` | 0.6980 | 0.6379 | 0.7365 | 0.9598 | 0.5472 | 0.8710 | 0.5803 | 0.6773 | 0.9246 | 0.6769 | 0.6350 |
| 19 | `mean_learned_weights_buy1.00_sell0.75` | 0.6980 | 0.6379 | 0.7365 | 0.9598 | 0.5472 | 0.8710 | 0.5803 | 0.6835 | 0.9115 | 0.6692 | 0.6350 |
| 20 | `mean_learned_weights_buy1.25_sell0.75` | 0.6980 | 0.6379 | 0.7365 | 0.9598 | 0.5472 | 0.8710 | 0.5836 | 0.6848 | 0.9049 | 0.6692 | 0.6350 |
