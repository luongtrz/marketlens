"""Factor cleaner: dedup, lowercase, trim noise."""


class FactorCleaner:
    """Cleans raw factor strings by deduplicating, lowercasing, and removing noise."""

    def clean(self, factors: list[str]) -> list[str]:
        """Clean a list of raw factor strings.

        Operations:
        - Lowercase all strings
        - Remove duplicates
        - Remove stopwords and noise tokens
        - Trim whitespace

        Args:
            factors: Raw factor strings.

        Returns:
            Cleaned factor strings.
        """
        raise NotImplementedError
