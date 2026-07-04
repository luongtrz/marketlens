# ETH Zero-Shot StockMem Report

This note records the first ETH evaluation using the existing BTC-oriented
StockMem artifacts without ETH-specific fine-tuning.

## Data

Source export:

```text
data/exports/stockmem_records_eth.ndjson
```

The file was pulled from Supabase `stockmem_records` with `symbol = ETH`.

| Field | Value |
| --- | ---: |
| Raw rows | 2908 |
| Rows with matured `future_return_7d` | 2903 |
| Date range | `2018-01-05` to `2026-07-01` |
| Validation rows | 174 |
| Test rows | 305 |
| D7 label threshold | `±2%` |

Vector coverage is complete for `price_vec`, `factor_vec`, and
`indicator_vec`. Five latest rows do not yet have matured D7 returns.

## Zero-Shot Question

The objective is not to select a final ETH model yet. The objective is to test
whether the BTC StockMem mechanism transfers to ETH before training an ETH
specialist.

The tested artifacts were:

```text
stockmem/config/weights.auto.json
stockmem/config/learned_retriever_finbert.json
stockmem/config/knn_head.fixed_knn_rolling_stable.json
stockmem/config/knn_head.learned_finbert_rolling_stable.json
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

## Strict Zero-Shot Result

Source:

```text
artifacts/eth_zero_shot_learned_strict_test_20260704/summary.json
```

Held-out ETH test split: `2025-07-01` to `2026-05-01`, `n = 305`.

| Model | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Hit@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 0.3180 | 0.4498 | 0.7508 | 0.4077 | 0.1842 | 0.2701 | 0.8656 |
| `fixed_retriever_learned_head` | 0.3443 | 0.4661 | 0.8230 | 0.4154 | 0.0789 | 0.3504 | 0.8656 |
| `learned_retriever_fixed_head` | 0.3803 | 0.4792 | 0.7869 | 0.5769 | 0.2895 | 0.2190 | 0.8820 |
| `learned_finbert_rolling_stable` | **0.4098** | **0.5020** | **0.8361** | 0.5308 | 0.2368 | 0.3431 | **0.8820** |

Paired comparison against `fixed_knn_rolling_stable`:

| Challenger | Metric | Delta | 95% bootstrap CI | McNemar p |
| --- | --- | ---: | ---: | ---: |
| `learned_finbert_rolling_stable` | overall_acc | +0.0918 | [+0.0230, +0.1607] | 0.012603 |

Interpretation: the learned FinBERT retriever plus learned stable head transfers
better to ETH than the fixed-kNN strict baseline. This is encouraging
zero-shot evidence for a multi-asset StockMem architecture.

## Maintained BTC Pipeline On ETH

The BTC-maintained evidence retriever is:

```text
learned_recency_50_50 =
  0.5 * learned_finbert_similarity
  + 0.5 * recency_decay(half_life = 21d)
```

Source:

```text
artifacts/eth_zero_shot_majority_consensus_20260704/summary.json
```

| Retriever | Val Majority@10 | Test Majority@10 | Full Majority@10 |
| --- | ---: | ---: | ---: |
| `learned_recency_50_50` | 0.5000 | 0.4754 | 0.5112 |

This is weaker than the BTC result, where the same maintained profile reached
test `majority@10 = 0.5443`. The likely reason is that the learned similarity
artifact was trained on BTC-style memory and has not yet adapted to ETH
microstructure and event/price relationships.

Diagnostic note: recency-heavy variants did better on ETH `majority@10`, but
the selected product direction remains learned memory plus recency. The next
step is therefore ETH-specific learned retriever fine-tuning, not switching the
main design back to fixed-only evidence.

## Consensus Head Zero-Shot

Source:

```text
artifacts/eth_zero_shot_consensus_heads_20260704/summary.json
```

Using the BTC-maintained `learned_recency_50_50` evidence retriever, validation
selected:

```text
mean_learned_weights_buy0.50_sell0.50
```

| Model | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | val | 174 | 0.5805 | 0.6821 | 0.9943 | 0.5094 | 0.0000 | 0.7957 | 0.5000 |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | test | 305 | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |

This result is stronger than the strict zero-shot classifier on overall
accuracy and SELL DA, but it still inherits the same limitation as the BTC
directional head: HOLD recognition is weak.

## Conclusion

The zero-shot result supports a two-profile deployment plan:

```text
BTC endpoint/profile: BTC-trained learned_recency_50_50 + BTC-selected head
ETH endpoint/profile: ETH-trained learned_recency_50_50 + ETH-selected head
```

The ETH result is good enough to justify fine-tuning. It is not yet the final
ETH model because the learned component has not been trained on ETH.

## Reproduction

Pull ETH StockMem records from Supabase:

```bash
python3 scripts/archive/pull_stockmem_records_from_supabase.py \
  --output data/exports/stockmem_records_eth.ndjson \
  --symbol ETH
```

Run strict zero-shot evaluation:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/evaluate_learned_strict_test.py \
    --data data/exports/stockmem_records_eth.ndjson \
    --weights stockmem/config/weights.auto.json \
    --artifact stockmem/config/learned_retriever_finbert.json \
    --fixed-head stockmem/config/knn_head.fixed_knn_rolling_stable.json \
    --learned-head stockmem/config/knn_head.learned_finbert_rolling_stable.json \
    --out-dir artifacts/eth_zero_shot_learned_strict_test_20260704
```

Run maintained evidence-retriever zero-shot evaluation:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/experimental/evaluate_majority_consensus_retrievers.py \
    --data data/exports/stockmem_records_eth.ndjson \
    --weights stockmem/config/weights.auto.json \
    --artifact stockmem/config/learned_retriever_finbert.json \
    --out-dir artifacts/eth_zero_shot_majority_consensus_20260704 \
    --top-k 10 \
    --min-pool-size 10 \
    --full-start-date 2018-01-01 \
    --config learned_recency_50_50:stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

Run zero-shot decision-head search over the maintained evidence retriever:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py \
    --data data/exports/stockmem_records_eth.ndjson \
    --weights stockmem/config/weights.auto.json \
    --artifact stockmem/config/learned_retriever_finbert.json \
    --config stockmem/config/majority_consensus_retriever.learned_recency_50_50.json \
    --out-dir artifacts/eth_zero_shot_consensus_heads_20260704 \
    --top-k 10
```

