# AIHub Naive LLM Baseline

The naive baseline tests whether a general LLM can classify D7 direction from
current-day context without StockMem retrieval evidence.

## Input Context

The prompt includes:

- current candle/market snapshot,
- recent 1d and 3d price changes,
- current indicators,
- one-day aggregated news sentiment,
- compact raw news titles from the daily summary.

It does not include historical neighbors or retrieved evidence.

## Runtime Policy

The evaluator uses Groq through AIHub. Important safeguards:

- response parsing accepts JSON embedded in surrounding text;
- strict JSON mode is retried without `response_format` if a Groq model rejects it;
- rate limits parse Groq retry-after text and wait instead of defaulting;
- completed rows can be resumed;
- failed predictions should remain explicit failures, not `HOLD`.

## Current Result

On the official `305`-row test split, the naive LLM baseline underperforms the
structured StockMem baseline:

| Model | Overall Acc | Active Acc | Coverage | SELL rate |
| --- | ---: | ---: | ---: | ---: |
| `naive_current_ai` | 0.2787 | 0.4031 | 0.6426 | 0.0262 |
| `fixed_knn_rolling_stable` | 0.3180 | 0.4236 | 0.7508 | 0.1213 |

The main failure mode is SELL avoidance: the LLM emits almost no `SELL` signals
even though the test set contains many negative D7 outcomes.
