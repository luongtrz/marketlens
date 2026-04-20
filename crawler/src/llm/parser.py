"""Parser for LLM responses — extracts structured fields from raw LLM output."""

import json
import re

from shared.models.article import EnrichedFields


class LLMResponseParser:
    """Parses raw LLM text/JSON output into structured EnrichedFields."""

    def parse_enrichment(self, raw_response: str) -> EnrichedFields:
        """Parse a combined enrichment response from the LLM.

        Args:
            raw_response: Raw text/JSON response from the LLM.

        Returns:
            Parsed EnrichedFields.
        """
        score, _ = self.parse_sentiment(raw_response)
        factors = self.parse_factors(raw_response)
        summary = None
        try:
            payload = json.loads(raw_response)
            summary = payload.get("summary")
        except Exception:
            summary = None
        return EnrichedFields(sentiment_score=score, summary=summary, factors=factors)

    def parse_sentiment(self, raw_response: str) -> tuple[float, str]:
        """Parse a sentiment response into (score, label).

        Args:
            raw_response: Raw LLM response for sentiment analysis.

        Returns:
            Tuple of (score: float, label: str).
        """
        try:
            payload = json.loads(raw_response)
            score = float(payload.get("score", 0.0))
        except Exception:
            m = re.search(r"-?\d+(?:\.\d+)?", raw_response or "")
            score = float(m.group(0)) if m else 0.0
        score = max(-1.0, min(1.0, score))
        if score > 0.15:
            return score, "bullish"
        if score < -0.15:
            return score, "bearish"
        return score, "neutral"

    def parse_factors(self, raw_response: str) -> list[str]:
        """Parse a factors response into a list of factor strings.

        Args:
            raw_response: Raw LLM response for factor extraction.

        Returns:
            List of extracted factor strings.
        """
        try:
            payload = json.loads(raw_response)
            factors = payload.get("factors", [])
            if isinstance(factors, list):
                return [str(x).strip() for x in factors if str(x).strip()]
        except Exception:
            pass
        lines = [line.strip("- ").strip() for line in (raw_response or "").splitlines()]
        return [line for line in lines if line]
