# StockMem Trend-Aware Learned-Memory Retrieval

This note records the current evidence-retrieval recommendation after the
majority-consensus audit. It supersedes the earlier interpretation that fixed
kNN alone was the best retrieval path for historical evidence.

## Current Recommendation

The current primary evidence retriever is:

```text
learned_recency_50_50
```

It combines learned historical similarity with trend awareness:

```text
score(q,c) =
  0.5 * learned_finbert_similarity(q,c)
  + 0.5 * recency_decay(q,c)
```

with:

```text
recency_decay(q,c) = exp(-age_days(q,c) / 21)
```

The maintained config artifact is:

```text
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

This is not a pure recency model. It retains learned historical memory while
acknowledging that D7 direction has strong temporal persistence.

## Why The Metric Changed

Earlier reports used `Hit@5_same_D7_sign`, which is too weak for evidence
quality. It only asks whether at least one of the retrieved records has the
same D7 direction as the query. A top-5 set such as:

```text
[SELL, BUY, BUY, HOLD, BUY]
```

is counted as a hit for an actual `SELL` query, even though the evidence set is
mostly not bearish.

The stricter evidence target is majority consistency:

```text
majority_same@10 = 1 if at least 5 of top-10 records share the query D7 class
```

This better matches how retrieved evidence is used by a decision layer: the
evidence set should be directionally coherent, not merely contain one relevant
case.

## Audit Findings

The audit found three important mechanisms:

1. `Hit@5` overstated retrieval quality because it measured recall of at least
   one matching case, not evidence-set consistency.
2. Pure fixed/learned similarity was only modestly better than random
   expectation on `majority@10`.
3. D7 labels have strong temporal clustering, so trend-aware retrieval is a
   necessary component.

The strongest practical model is learned-memory plus recency:

| Model | Val Majority@10 | Test Majority@10 | Full Majority@10 |
| --- | ---: | ---: | ---: |
| `fixed_only` | 0.4368 | 0.3639 | 0.3817 |
| `learned_only` | 0.5057 | 0.3541 | 0.3755 |
| `recency_only` | 0.6264 | 0.5180 | 0.5075 |
| `unconstrained` | **0.6379** | 0.5246 | 0.5092 |
| `memory_first_learned030` | 0.5977 | 0.5279 | 0.4967 |
| `learned_recency_50_50` | 0.5920 | **0.5443** | **0.5106** |

The full-history row evaluates `2871` eligible rows from `2018-01-01` to
`2026-06-08` with a minimum matured pool size of 10.

## Interpretation

The audit does not prove that a pure learned retriever is better. In fact,
`learned_only` is unstable: it improves validation evidence consistency but
does not beat fixed-only on the held-out test or full-history run.

The defensible conclusion is:

```text
Historical memory is useful when it is trend-aware.
Recency alone is a strong continuation baseline.
Learned memory plus recency is the best current evidence retriever.
```

This changes the StockMem narrative from:

```text
fixed kNN vs learned retriever
```

to:

```text
memory similarity + trend awareness
```

## Failure Mode

A trend-aware retriever can fail during reversals. If the recent market trend
is about to reverse, recency will retrieve evidence from the old trend regime.
This is why the model should not be described as a complete forecasting model.
It is an evidence retriever whose output should still be calibrated by a
decision head or reversal-aware guard.

The next research step is therefore a learned gate:

```text
retrieved evidence + reversal-risk features -> trust recency or trust historical analogs
```

Candidate reversal-risk features include:

- RSI exhaustion,
- MACD histogram weakening,
- volatility spike,
- price/news sentiment divergence,
- top-k evidence disagreement,
- recent trend versus current candle conflict.

## Decision Head Result

The retrieval audit was followed by a validation-selected decision-head search
over the top-10 evidence returned by `learned_recency_50_50`.

Source:

```text
artifacts/consensus_retriever_heads_20260703/summary.json
```

The selected head is:

```text
count_vote_buy3_sell4
```

Test result on the held-out `2025-07-01` to `2026-05-01` split:

| Model | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + count_vote_buy3_sell4` | 305 | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.0000 | 0.7114 |
| old `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6327 | 0.2759 | 0.1275 |
| old `fixed_retriever_learned_head` | 305 | 0.3508 | 0.4500 | 0.8525 | 0.7041 | 0.1552 | 0.1946 |

Mean-return heads were competitive. The best validation-ranked mean head,
`mean_d7_only_buy0.50_sell0.50`, reached `0.5508` overall test accuracy and
`0.7047` SELL DA, but it was not the official selection because the count-vote
head had the highest validation score.

The result changes the practical recommendation: the maintained StockMem
pipeline now uses trend-aware learned-memory retrieval followed by a simple
evidence-consensus head.

## Reproduction

Train constrained majority-consensus retrievers:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/experimental/train_majority_consensus_retriever.py \
    --data data/exports/stockmem_records.ndjson \
    --weights stockmem/config/weights.auto.json \
    --artifact stockmem/config/learned_retriever_finbert.json \
    --out-dir artifacts/memory_first_retriever_20260703/min_learned_030 \
    --top-k 10 \
    --grid-step 0.1 \
    --min-memory-weight 0.5 \
    --min-learned-weight 0.3 \
    --max-recency-weight 0.4
```

Evaluate maintained configs on validation, held-out test, and full history:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/experimental/evaluate_majority_consensus_retrievers.py \
    --data data/exports/stockmem_records.ndjson \
    --weights stockmem/config/weights.auto.json \
    --artifact stockmem/config/learned_retriever_finbert.json \
    --out-dir artifacts/majority_consensus_retriever_eval_20260703 \
    --top-k 10 \
    --min-pool-size 10 \
    --full-start-date 2018-01-01 \
    --config learned_recency_50_50:stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

Primary output:

```text
artifacts/majority_consensus_retriever_eval_20260703/summary.md
```

Evaluate the decision head over the maintained retriever:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py \
    --data data/exports/stockmem_records.ndjson \
    --weights stockmem/config/weights.auto.json \
    --artifact stockmem/config/learned_retriever_finbert.json \
    --config stockmem/config/majority_consensus_retriever.learned_recency_50_50.json \
    --out-dir artifacts/consensus_retriever_heads_20260703 \
    --top-k 10
```
