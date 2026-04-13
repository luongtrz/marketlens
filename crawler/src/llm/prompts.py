"""Prompt templates for LLM-based article enrichment."""


SENTIMENT_PROMPT = """
Analyze the following crypto news article and provide a sentiment score
from -1.0 (very bearish) to 1.0 (very bullish).

Article: {text}

Respond with a JSON object: {{ "score": float, "label": "bullish"|"bearish"|"neutral" }}
"""

SUMMARY_PROMPT = """
Summarize the following crypto news article in 2-3 sentences:

Article: {text}
"""

FACTOR_EXTRACTION_PROMPT = """
Extract key market-moving factors from the following crypto news article.
Return a JSON list of factor strings.

Article: {text}

Respond with: {{ "factors": ["factor1", "factor2", ...] }}
"""
