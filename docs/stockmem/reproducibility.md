# StockMem Reproducibility

Generated outputs are not committed. Reproduce them into `submission/` or
`results_tables/` when needed for the report.

The official NDJSON dataset is also not committed in the clean submission
branch. Place it at `data/exports/stockmem_records.ndjson`, or pass another
path with `--dataset`.

## Official Docker Command

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /bin/sh \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python stockmem/scripts/run_submission_reproduction.py \
    --dataset data/exports/stockmem_records.ndjson \
    --out-dir submission/stockmem_2026_07"
```

If Groq quota is unavailable:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /bin/sh \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python stockmem/scripts/run_submission_reproduction.py \
    --dataset data/exports/stockmem_records.ndjson \
    --out-dir submission/stockmem_2026_07 \
    --skip-llm \
    --naive-summary artifacts/current_context_ai_eval/summary.json"
```

## Official Scripts

| Script | Purpose |
| --- | --- |
| `aihub/scripts/evaluate_naive_llm_baseline.py` | Current-context LLM baseline with resume/retry behavior. |
| `stockmem/scripts/evaluate_stockmem_strict_models.py` | Strict structured model comparison. |
| `stockmem/scripts/evaluate_stockmem_feature_ablation.py` | Fixed-kNN feature-block ablation. |
| `stockmem/scripts/experimental/train_majority_consensus_retriever.py` | Train trend-aware majority-consensus retriever configs. |
| `stockmem/scripts/experimental/evaluate_majority_consensus_retrievers.py` | Evaluate `majority_same@10` on val, test, and full history. |
| `stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py` | Select and test decision heads over the maintained consensus retriever. |
| `stockmem/scripts/export_stockmem_report_tables.py` | Compact Markdown/CSV table export. |
| `stockmem/scripts/run_submission_reproduction.py` | End-to-end reproduction orchestrator. |

## Expected Outputs

| Path | Contents |
| --- | --- |
| `submission/stockmem_2026_07/manifest.json` | Dataset checksum, split, output map. |
| `submission/stockmem_2026_07/current_context_ai_eval/summary.json` | Naive LLM vs structured baseline summary. |
| `submission/stockmem_2026_07/learned_strict_test/summary.json` | Primary structured model table and paired tests. |
| `submission/stockmem_2026_07/fixed_knn_component_ablation/summary.json` | Feature-block ablation. |
| `submission/stockmem_2026_07/tables/*.md` | Report-ready compact tables. |
| `submission/stockmem_2026_07/tables/*.csv` | Spreadsheet-friendly tables. |

## Trend-Aware Retrieval Reproduction

The maintained evidence retriever is stored at:

```text
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
```

Re-evaluate it against fixed-only, learned-only, recency-only, and constrained
memory-first variants:

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

Expected maintained result for `learned_recency_50_50`:

| Split | n | Majority@10 | Mean same@10 | SELL majority@10 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 174 | 0.5920 | 5.7874 | 0.7193 |
| Test | 305 | 0.5443 | 5.2754 | 0.6779 |
| Full history | 2871 | 0.5106 | 4.9721 | 0.5544 |

Evaluate the validation-selected decision head over that retriever:

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

Expected maintained test result:

| Model | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + count_vote_buy3_sell4` | 305 | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.0000 | 0.7114 |

To retrain the constrained memory-first candidate:

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

## Audit Rules

- Do not silently convert failed LLM calls into `HOLD`.
- Use resume mode for LLM runs to avoid paying for completed rows again.
- Keep `artifacts/`, `results_tables/`, and `submission/` out of git.
- Treat per-record JSONL files and logs as audit artifacts, not thesis text.

## ETH Fine-Tune Preparation

The multi-asset profile map is:

```text
stockmem/config/model_profiles.json
```

The ETH profile reserves these paths:

```text
data/exports/stockmem_records_eth.ndjson
stockmem/data/real_optimizer_finbert_eth.json
stockmem/config/learned_retriever_finbert.eth.json
stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50.json
```

Pull ETH StockMem records:

```bash
python3 scripts/archive/pull_stockmem_records_from_supabase.py \
  --output data/exports/stockmem_records_eth.ndjson \
  --symbol ETH
```

Train the ETH learned retriever without overwriting BTC artifacts:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/retrain_finbert_retriever.py \
    --input-ndjson data/exports/stockmem_records_eth.ndjson \
    --dataset-output stockmem/data/real_optimizer_finbert_eth.json \
    --artifact-output stockmem/config/learned_retriever_finbert.eth.json \
    --trials 10 \
    --epochs 40 \
    --seeds 5 \
    --selection-metric hybrid
```

Evaluate the maintained learned+recency `50/50` design with the ETH-trained
learned metric:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/experimental/evaluate_majority_consensus_retrievers.py \
    --data data/exports/stockmem_records_eth.ndjson \
    --weights stockmem/config/weights.auto.json \
    --artifact stockmem/config/learned_retriever_finbert.eth.json \
    --out-dir artifacts/eth_finetuned_majority_consensus \
    --top-k 10 \
    --min-pool-size 10 \
    --full-start-date 2018-01-01 \
    --config eth_learned_recency_50_50:stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50.json
```

Select the ETH decision head over the ETH-trained retriever:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /usr/local/bin/python \
  marketlens-aihub:latest \
  stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py \
    --data data/exports/stockmem_records_eth.ndjson \
    --weights stockmem/config/weights.auto.json \
    --artifact stockmem/config/learned_retriever_finbert.eth.json \
    --config stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50.json \
    --out-dir artifacts/eth_finetuned_consensus_heads \
    --top-k 10
```
