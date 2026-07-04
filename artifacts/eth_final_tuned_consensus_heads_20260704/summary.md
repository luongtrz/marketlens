# Consensus Retriever Decision Head Evaluation

- Data: `data/exports/stockmem_records_eth.ndjson`
- Retriever config: `stockmem/config/majority_consensus_retriever.eth.tuned_eth_weights.json`
- Top-k: `10`
- Label threshold: `±2.00%`

## Selected Head

- Head: `mean_learned_weights_buy0.50_sell0.75`
- Validation score: `0.6984`

## Comparison

| Model | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 | Mean Same@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.75` | val | 174 | 0.6379 | 0.7546 | 0.9368 | 0.5283 | 0.0357 | 0.8817 | 0.5747 | 5.0690 |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.75` | test | 305 | 0.5967 | 0.6678 | 0.9377 | 0.6923 | 0.1316 | 0.6350 | 0.5508 | 5.2000 |
| `fixed_knn_rolling_stable` | test_old_strict | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6327 | 0.2759 | 0.1275 | n/a | n/a |
| `fixed_retriever_learned_head` | test_old_strict | 305 | 0.3508 | 0.4500 | 0.8525 | 0.7041 | 0.1552 | 0.1946 | n/a | n/a |
| `learned_finbert_rolling_stable` | test_old_strict | 305 | 0.3410 | 0.4393 | 0.7836 | 0.5408 | 0.2414 | 0.2483 | n/a | n/a |

## Top Validation Heads

| Rank | Head | Score | Val Overall | Val Active | Val Coverage | Val BUY DA | Val SELL DA | Test Overall | Test Active | Test Coverage | Test BUY DA | Test SELL DA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mean_learned_weights_buy0.50_sell0.75` | 0.6984 | 0.6379 | 0.7546 | 0.9368 | 0.5283 | 0.8817 | 0.5967 | 0.6678 | 0.9377 | 0.6923 | 0.6350 |
| 2 | `mean_learned_weights_buy0.50_sell1.00` | 0.6984 | 0.6379 | 0.7546 | 0.9368 | 0.5283 | 0.8817 | 0.5967 | 0.6678 | 0.9279 | 0.6923 | 0.6277 |
| 3 | `mean_learned_weights_buy0.50_sell1.25` | 0.6984 | 0.6379 | 0.7546 | 0.9368 | 0.5283 | 0.8817 | 0.5934 | 0.6775 | 0.9049 | 0.6923 | 0.6131 |
| 4 | `mean_learned_weights_buy0.50_sell1.50` | 0.6984 | 0.6379 | 0.7546 | 0.9368 | 0.5283 | 0.8817 | 0.5902 | 0.6813 | 0.8951 | 0.6923 | 0.6058 |
| 5 | `mean_learned_weights_buy0.50_sell0.50` | 0.6983 | 0.6379 | 0.7410 | 0.9540 | 0.5283 | 0.8817 | 0.5967 | 0.6678 | 0.9377 | 0.6923 | 0.6350 |
| 6 | `mean_fixed_weights_buy0.50_sell0.50` | 0.6948 | 0.6322 | 0.7193 | 0.9828 | 0.5283 | 0.8817 | 0.6098 | 0.6713 | 0.9475 | 0.7000 | 0.6496 |
| 7 | `mean_fixed_weights_buy0.75_sell0.50` | 0.6948 | 0.6322 | 0.7193 | 0.9828 | 0.5283 | 0.8817 | 0.6000 | 0.6738 | 0.9246 | 0.6692 | 0.6496 |
| 8 | `mean_fixed_weights_buy1.00_sell0.50` | 0.6948 | 0.6322 | 0.7193 | 0.9828 | 0.5283 | 0.8817 | 0.5967 | 0.6774 | 0.9148 | 0.6615 | 0.6496 |
| 9 | `mean_learned_weights_buy0.50_sell1.75` | 0.6939 | 0.6322 | 0.7531 | 0.9310 | 0.5283 | 0.8710 | 0.5902 | 0.6852 | 0.8852 | 0.6923 | 0.5985 |
| 10 | `mean_learned_weights_buy0.75_sell0.75` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5967 | 0.6690 | 0.9311 | 0.6846 | 0.6350 |
| 11 | `mean_learned_weights_buy0.75_sell1.00` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5967 | 0.6690 | 0.9213 | 0.6846 | 0.6277 |
| 12 | `mean_learned_weights_buy0.75_sell1.25` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5934 | 0.6788 | 0.8984 | 0.6846 | 0.6131 |
| 13 | `mean_learned_weights_buy0.75_sell1.50` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5902 | 0.6827 | 0.8885 | 0.6846 | 0.6058 |
| 14 | `mean_learned_weights_buy1.00_sell0.75` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5934 | 0.6714 | 0.9180 | 0.6692 | 0.6350 |
| 15 | `mean_learned_weights_buy1.00_sell1.00` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5934 | 0.6715 | 0.9082 | 0.6692 | 0.6277 |
| 16 | `mean_learned_weights_buy1.00_sell1.25` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5902 | 0.6815 | 0.8852 | 0.6692 | 0.6131 |
| 17 | `mean_learned_weights_buy1.00_sell1.50` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5869 | 0.6854 | 0.8754 | 0.6692 | 0.6058 |
| 18 | `mean_learned_weights_buy1.25_sell0.75` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5902 | 0.6739 | 0.9049 | 0.6538 | 0.6350 |
| 19 | `mean_learned_weights_buy1.25_sell1.00` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5902 | 0.6740 | 0.8951 | 0.6538 | 0.6277 |
| 20 | `mean_learned_weights_buy1.25_sell1.25` | 0.6937 | 0.6322 | 0.7531 | 0.9310 | 0.5094 | 0.8817 | 0.5869 | 0.6842 | 0.8721 | 0.6538 | 0.6131 |
