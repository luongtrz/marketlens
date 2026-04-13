"""Record → embedding vector converter for similarity search."""

import numpy as np

from shared.models.memory import StockMemRecord


class RecordEmbedder:
    """Converts a StockMemRecord to a dense vector for similarity search.

    Embedding strategy:
      - Concatenate: [sentiment_score, rsi, macd_hist, ...factors_tfidf...]
      - Normalize to unit sphere
    """

    def embed(self, record: StockMemRecord) -> np.ndarray:
        """Convert a StockMemRecord to a dense embedding vector.

        Args:
            record: The record to embed.

        Returns:
            Unit-normalized numpy array representing the record.
        """
        raise NotImplementedError

    def embed_batch(self, records: list[StockMemRecord]) -> np.ndarray:
        """Embed multiple records in a batch.

        Args:
            records: List of records.

        Returns:
            2D numpy array of shape (n_records, embedding_dim).
        """
        raise NotImplementedError
