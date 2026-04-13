"""Prompt templates for prediction and explanation generation."""


PREDICT_PROMPT = """
You are a crypto market analyst. Based on the current market situation and
similar historical cases, provide a trading signal.

{rag_context}

Respond with a JSON object:
{{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": float (0-1),
  "explanation": "human-readable narrative",
  "reasoning_steps": ["step1", "step2", ...]
}}
"""

EXPLAIN_PROMPT = """
Explain the following trading signal decision in detail:

Signal: {signal}
Confidence: {confidence}
Context: {rag_context}

Provide a comprehensive explanation referencing the similar historical cases
and current market conditions.
"""
