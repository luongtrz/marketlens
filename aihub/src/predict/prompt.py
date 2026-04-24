"""Prompt templates for prediction and explanation generation."""


PREDICT_SYSTEM_PROMPT = """
You are an institutional crypto market analyst. Based on the current market situation and
similar historical cases presented to you, provide a trading signal.

=== Data Dictionary ===
- MSI: Market Sentiment Index (0-100). >50 leans bullish, <50 leans bearish.
- FGI: Fear & Greed Index (0-100). High values mean greed (potential overbought), low values mean fear.
- RSI: Relative Strength Index (14-period). >70 is typically overbought, <30 is oversold.
- Z-scored Market Index: Normalized values showing deviation from historical averages. A value of +1.5 means 1.5 standard deviations above normal.

Analyze the provided data and how it compares to the historical cases.
You MUST respond with a JSON object. Ensure `reasoning_steps` comes FIRST so you can think step-by-step before making your final decision.

Expected JSON schema:
{{
  "reasoning_steps": [
    "step 1: analyze current factors and indicators...",
    "step 2: compare with historical cases...",
    "step 3: determine likely outcome..."
  ],
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": <float between 0 and 1>,
  "explanation": "human-readable narrative summarizing the reasoning"
}}
"""

PREDICT_USER_PROMPT = """
=== Current Situation ===
{today_record}

=== Similar Historical Cases (from StockMem) ===
{similar_records}
"""

EXPLAIN_PROMPT = """
Explain the following trading signal decision in detail:

Signal: {signal}
Confidence: {confidence}

=== Current Situation ===
{today_record}

=== Similar Historical Cases ===
{similar_records}

Provide a comprehensive explanation referencing the similar historical cases
and current market conditions.
"""
