"""Text preprocessing for BERT model input."""


class SentimentPreprocessor:
    """Cleans and tokenizes text for CryptoBert input."""

    def clean(self, text: str) -> str:
        """Clean and normalize text for sentiment analysis.

        Removes HTML tags, normalizes whitespace, handles crypto-specific
        tokens and abbreviations.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text suitable for BERT tokenization.
        """
        raise NotImplementedError

    def truncate(self, text: str, max_tokens: int = 512) -> str:
        """Truncate text to fit within the model's max token window.

        Args:
            text: Cleaned text.
            max_tokens: Maximum number of tokens.

        Returns:
            Truncated text.
        """
        raise NotImplementedError
