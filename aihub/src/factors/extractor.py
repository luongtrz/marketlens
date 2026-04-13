"""Text → factor list extractor using SKGP and supplementary heuristics."""

from shared.models.factor import Factor


class FactorExtractor:
    """Orchestrates factor extraction from article text.

    Combines SKGP with rule-based heuristics for comprehensive
    factor identification.
    """

    def extract(self, text: str) -> list[Factor]:
        """Extract factors from the given text.

        Args:
            text: Article text.

        Returns:
            Deduplicated list of Factor objects.
        """
        raise NotImplementedError
