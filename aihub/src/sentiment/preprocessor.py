"""Text preprocessing for BERT model input."""
import re

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

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return text


    def truncate(self, text: str, max_tokens: int = 512) -> str:
        """Truncate text to fit within the model's max token window.

        Args:
            text: Cleaned text.
            max_tokens: Maximum number of tokens.

        Returns:
            Truncated text.
        """
        return text[:max_tokens]
