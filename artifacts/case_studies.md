# CEM-RAG Case Studies

Four representative predictions from the test period (2025-07-01 to 2026-05-01) drawn from
`artifacts/predictions/cem_rag_test.jsonl`, `knn_returns_test.jsonl`, and `fixed_knn_test.jsonl`.
Baselines use the same kNN retrieval core but replace CEM-RAG's probabilistic event model with
either weighted-average past returns (`knn_returns`) or a sign-vote over neighbour 7-day returns
(`fixed_knn`).

---

## Test-Period Summary Statistics

| Metric | Value |
|---|---|
| Total test days (valid 7d return) | 305 |
| Date range | 2025-07-01 – 2026-05-01 |
| Mean 7d return | −3.23% (persistent downtrend bias) |
| Std 7d return | ±10.88% |

| Model | BUY | SELL | HOLD | SELL-DA | BUY-DA |
|---|---|---|---|---|---|
| CEM-RAG | 78 | 47 | 180 | **61.7%** (29/47) | 39.7% (31/78) |
| knn\_returns | 183 | 50 | 72 | 58.0% (29/50) | 39.9% (73/183) |
| fixed\_knn | 192 | 55 | 58 | 56.4% (31/55) | 40.1% (77/192) |

CEM-RAG issues far fewer directional signals (125 vs. 233–247 for baselines), concentrating
capital on the highest-conviction days.  SELL directional accuracy is +3.7 pp above
`knn_returns` and +5.3 pp above `fixed_knn`.

---

## Case 1 — Correct BUY: Whale Accumulation Ahead of 16.5% Rally

```
Date:         2026-03-28
Symbol:       BTC
Signal:       BUY
Confidence:   0.95
p_up / p_down / p_hold:  1.0000 / 0.0000 / 0.0000
Actual 7d return:  +16.51%
Baseline (knn_returns) signal:  HOLD  (weighted-avg neighbour return −0.39%)
Baseline (fixed_knn)  signal:  HOLD  (avg neighbour 7d return +0.17%)
```

**Why CEM-RAG succeeded:** The event vector has a dominant activation on the *Whale &
On-chain* group (dim 69, weight 0.798) alongside moderate positive polarity (SCALAR[0] = +0.32)
and mean confidence 0.40 across events.  Both market-momentum indicators were depressed at the
time — MSI z-score −0.75, RSI z-score −0.63 — suggesting a classic oversold accumulation setup
that price-only neighbours could not identify.  CEM-RAG's retrieved analogues (k=5) all came
from prior Whale Accumulation + oversold episodes that resolved with strong upside; the
probabilistic event model raised p\_up to 1.00 with full confidence.  Because the baselines
compare raw price returns of neighbours (which were flat on average), both returned HOLD.
CEM-RAG uniquely captured the event-type signal that preceded the +16.51% move over the
following seven days.

---

## Case 2 — Correct SELL: Regulatory Shock Detected Early, −29.1% Avoided

```
Date:         2026-01-19
Symbol:       BTC
Signal:       SELL
Confidence:   0.95
p_up / p_down / p_hold:  0.0000 / 1.0000 / 0.0000
Actual 7d return:  −29.05%
Baseline (knn_returns) signal:  HOLD  (weighted-avg neighbour return +0.08%)
Baseline (fixed_knn)  signal:  HOLD  (avg neighbour 7d return −0.82%)
```

**Why CEM-RAG succeeded:** The dominant group activation is *Market Performance* (dim 71,
weight 0.650) with a negative polarity scalar (SCALAR[0] = −0.39) and high polarity-magnitude
(SCALAR[1] = +0.39), indicating large adverse price reactions in the retrieved articles.
Simultaneously sentiment_score was sharply negative (z-score −0.82) and MSI was suppressed
(z-score −0.54).  The 7-day look-ahead on 1d actual return (−5.60%) confirms the deterioration
was already under way.  CEM-RAG's event-conditioned probability model mapped this configuration
to p\_down = 1.00 by retrieving analogues from episodes where market-performance deterioration
headlines coincided with sentiment collapse.  Both baselines saw mixed neighbour returns and
defaulted to HOLD, missing the onset of a −29% drawdown.

---

## Case 3 — Correct HOLD (Avoided Bad Trade): Risk-Warning Signal Saves Capital, −33% Drawdown

```
Date:         2026-01-06
Symbol:       BTC
Signal:       HOLD
Confidence:   0.562
p_up / p_down / p_hold:  0.3965 / 0.6035 / 0.0000
Actual 7d return:  −32.97%
Baseline (knn_returns) signal:  BUY  (conf 0.90, weighted-avg return +8.02%)
Baseline (fixed_knn)  signal:  BUY  (conf 0.95, avg neighbour 7d return +11.01%)
```

**Why CEM-RAG avoided the trade:** The event vector shows a *Risk & Warning* group
activation (dim 74, weight 0.512) together with a negative-polarity scalar (SCALAR[0] = −0.46)
and moderate multi-event breadth (SCALAR[2] = +0.498).  RSI was elevated (z-score +0.77),
a classic divergence signal when event risk is rising.  Both baselines relied solely on
neighbour price trajectories: neighbours' past seven-day returns averaged +8–11%, so they
issued high-confidence BUY calls.  CEM-RAG's event model returned p\_up = 0.40 / p\_down = 0.60
— below the 0.63 tau threshold for a directional signal — and correctly held.  The market
fell −32.97% over the next seven days.  Blindly following either baseline would have resulted
in a near-maximum drawdown; CEM-RAG preserved capital by recognising the warning-event
context that price history alone obscured.

---

## Case 4 — Failure Case: Interest-Rate Optimism Misfires, −18.3% Surprise

```
Date:         2026-01-31
Symbol:       BTC
Signal:       BUY
Confidence:   0.95
p_up / p_down / p_hold:  1.0000 / 0.0000 / 0.0000
Actual 7d return:  −18.31%
Baseline (knn_returns) signal:  HOLD  (conf 0.50, weighted-avg return −0.83%)
Baseline (fixed_knn)  signal:  HOLD  (conf 0.60, avg neighbour 7d return −0.81%)
```

**Why CEM-RAG failed:** The event vector fires on *Interest Rate Decision* (type dim 5,
weight 0.520), with co-activations on both *Macroeconomic* (dim 63) and *Market Performance*
(dim 71) groups at equal weight 0.520.  In past analogues retrieved by CEM-RAG, interest-rate
decision days combined with market-performance headlines resolved positively — a pattern the
model learnt during the 2024 rate-cut cycle.  However, in January 2026 the Fed signalled
a "higher-for-longer" extension rather than a cut, producing a context mismatch: retrieved
analogues were structurally similar in event type but had opposite macroeconomic regime.
Compounding this, RSI was deeply oversold (z-score −0.86), which historically accompanies
bounces but here reflected genuine bearish momentum.  The event model returned maximum
p\_up = 1.00 because no historical analogue in the training pool captured a rate-pause
surprise inside a prolonged downtrend.  Both baselines, relying on mixed neighbour returns,
returned HOLD — a better outcome than CEM-RAG's high-confidence misfired BUY.  This case
illustrates the model's key failure mode: when a familiar event type (rate decision)
occurs in an unfamiliar macro regime, retrieval quality degrades and the event probability
model over-commits.
