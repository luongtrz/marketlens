"""Parser for LLM responses — extracts structured fields from raw LLM output."""

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
        raise NotImplementedError

    def parse_sentiment(self, raw_response: str) -> tuple[float, str]:
        """Parse a sentiment response into (score, label).

        Args:
            raw_response: Raw LLM response for sentiment analysis.

        Returns:
            Tuple of (score: float, label: str).
        """
        raise NotImplementedError

    def parse_factors(self, raw_response: str) -> list[str]:
        """Parse a factors response into a list of factor strings.

        Args:
            raw_response: Raw LLM response for factor extraction.

        Returns:
            List of extracted factor strings.
        """
        raise NotImplementedError
