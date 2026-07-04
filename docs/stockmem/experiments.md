# StockMem Experiments

The maintained result set answers two practical questions:

1. Does the structured StockMem pipeline outperform naive current-context LLM
   prompting?
2. Which StockMem mechanism is strongest: fixed kNN, learned retrieval, learned
   head, or hybrid reranking?
3. Which retriever gives the most directionally coherent historical evidence
   under the stricter `majority_same@10` target?
4. Does the new trend-aware evidence retriever support a stronger decision
   head than the older strict classifier table?

The strict decision tables use the held-out `2025-07-01` to `2026-05-01`
split with `305` rows and D7 labels from `future_return_7d` at `±2%`.
The evidence-retrieval audit additionally reports validation and full-history
results from `2018-01-01` to `2026-06-08`.

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

Interpretation: this table is now an older strict baseline. At that stage, the
best structured classifier was fixed retrieval plus the learned stable head.
The newer consensus-head audit below supersedes it for the recommended
decision path.

## Consensus Retriever Decision Head Audit

Source: `artifacts/consensus_retriever_heads_20260703/summary.json`.

After selecting `learned_recency_50_50` as the evidence retriever, a second
validation-only search compared simple decision heads over its top-10 evidence:

- count-vote heads over D7 `BUY/HOLD/SELL` classes,
- median D7-return threshold heads,
- mean future-return threshold heads.

The selected head was:

```text
count_vote_buy3_sell4
```

It predicts `BUY` when at least 3 of the top-10 records are BUY and BUY is
stronger than SELL; it predicts `SELL` when at least 4 of the top-10 records are
SELL and SELL is at least as strong as BUY; otherwise it emits `HOLD`.

| Model | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + count_vote_buy3_sell4` | val | 174 | 0.6379 | 0.7658 | 0.9080 | 0.7614 | 0.0345 | 0.7544 | 0.5920 |
| `learned_recency_50_50 + count_vote_buy3_sell4` | test | 305 | **0.5475** | **0.6826** | **0.9607** | 0.6224 | 0.0000 | **0.7114** | 0.5443 |
| old `fixed_knn_rolling_stable` | test | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6327 | 0.2759 | 0.1275 | n/a |
| old `fixed_retriever_learned_head` | test | 305 | 0.3508 | 0.4500 | 0.8525 | **0.7041** | 0.1552 | 0.1946 | n/a |
| old `learned_finbert_rolling_stable` | test | 305 | 0.3410 | 0.4393 | 0.7836 | 0.5408 | 0.2414 | 0.2483 | n/a |

The best validation-selected mean-return head was close but not the official
winner:

| Head | Test Overall | Test Active | Test Coverage | Test BUY DA | Test SELL DA |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mean_d7_only_buy0.50_sell0.50` | 0.5508 | 0.6840 | 0.9443 | 0.6122 | 0.7047 |
| `count_vote_buy3_sell4` | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.7114 |

Interpretation: the new retriever does support a much stronger decision layer.
The main gain is downside recognition: SELL DA rises from `0.1275-0.2483` in
the older strict baselines to `0.7114`. The cost is that HOLD DA is effectively
zero, so this should be described as a high-coverage directional decision head,
not as a balanced three-class classifier.

## Evidence Retrieval Majority Audit

Source: `artifacts/majority_consensus_retriever_eval_20260703/summary.json`.

The earlier `Hit@5_same_sign` metric is useful as a recall check, but it is too
easy for evidence quality because it only requires one matching case in the
top-k set. The stricter retrieval target is:

```text
majority_same@10 = at least 5 of top-10 historical records share the query D7 class
```

The current primary evidence retriever is:

```text
learned_recency_50_50
```

with config:

```text
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

It combines learned historical similarity and trend awareness:

```text
score(q,c) = 0.5 * learned_similarity(q,c) + 0.5 * exp(-age_days(q,c) / 21)
```

| Model | Val Majority@10 | Test Majority@10 | Full Majority@10 |
| --- | ---: | ---: | ---: |
| `fixed_only` | 0.4368 | 0.3639 | 0.3817 |
| `learned_only` | 0.5057 | 0.3541 | 0.3755 |
| `recency_only` | 0.6264 | 0.5180 | 0.5075 |
| `unconstrained` | **0.6379** | 0.5246 | 0.5092 |
| `memory_first_learned030` | 0.5977 | 0.5279 | 0.4967 |
| `learned_recency_50_50` | 0.5920 | **0.5443** | **0.5106** |

Full-history uses `2871` eligible rows from `2018-01-01` to `2026-06-08`
after skipping 14 early rows with insufficient matured pool.

Interpretation: pure learned retrieval is not enough, and pure recency is a
strong continuation baseline. The best current evidence retriever is learned
memory plus trend awareness. This is a stronger and more honest claim than
"learned retrieval beats fixed kNN."

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

The maintained recommendation is now layered:

```text
evidence retriever: learned_recency_50_50
decision head: count_vote_buy3_sell4 over the top-10 evidence set
```

The broader research conclusion is that StockMem is useful as structured
historical memory, but retrieval must be trend-aware. Pure fixed kNN and pure
learned similarity both underperform the new evidence retriever on
`majority_same@10`, and the evidence-consensus head substantially improves
directional test accuracy relative to the older strict classifier table.

For the full academic write-up, see [academic_paper.md](academic_paper.md).
For the retrieval-specific audit, see
[trend_aware_retrieval.md](trend_aware_retrieval.md).

## ETH Zero-Shot Extension

Source: [eth_zero_shot.md](eth_zero_shot.md).

ETH was evaluated with the existing BTC artifacts before ETH-specific training.
The test split remains `2025-07-01` to `2026-05-01` with `305` rows.

| Model | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Hit@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 0.3180 | 0.4498 | 0.7508 | 0.4077 | 0.1842 | 0.2701 | 0.8656 |
| `fixed_retriever_learned_head` | 0.3443 | 0.4661 | 0.8230 | 0.4154 | 0.0789 | 0.3504 | 0.8656 |
| `learned_retriever_fixed_head` | 0.3803 | 0.4792 | 0.7869 | 0.5769 | 0.2895 | 0.2190 | 0.8820 |
| `learned_finbert_rolling_stable` | **0.4098** | **0.5020** | **0.8361** | 0.5308 | 0.2368 | 0.3431 | **0.8820** |

The BTC-maintained `learned_recency_50_50` profile also transferred to ETH:

| Model | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |

The zero-shot result supports a two-profile deployment plan:

```text
BTC profile: BTC-trained learned_recency_50_50 + BTC-selected head
ETH profile: ETH-trained learned_recency_50_50 + ETH-selected head
```

The ETH report keeps diagnostic fixed/recency comparisons for audit, but the
chosen product direction remains learned memory plus recency `50/50` for both
assets.
