"""SKGP (Structured Knowledge-Guided Parsing) technique implementation.

Parses article text to extract named factors (entities, events, macro signals)
relevant to crypto markets.
"""

from shared.models.factor import Factor, FactorType


class SKGPExtractor:
    """Structured Knowledge-Guided Parser for factor extraction.

    Extracts named factors from article text, classifying each by type,
    polarity, and confidence.
    """

    def extract(self, text: str) -> list[Factor]:
        """Extract market factors from article text.

        Args:
            text: Article text to analyze.

        Returns:
            List of Factor objects with name, type, polarity, and confidence.
        """
        raise NotImplementedError
