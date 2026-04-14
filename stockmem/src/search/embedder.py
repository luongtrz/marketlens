from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

import numpy as np

from ..models import StockMemRecord


class RecordEmbedder:
    """
    Converts a StockMemRecord to a dense vector for similarity search.
    Embedding strategy:
      - Concatenate: [sentiment_score, rsi, macd_hist, ...factors_tfidf...]
      - Normalize to unit sphere
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._doc_freq: Counter[str] = Counter()
        self._doc_count: int = 0

    def rebuild_corpus(self, records: Iterable[StockMemRecord]) -> None:
        self._vocab.clear()
        self._doc_freq.clear()
        self._doc_count = 0

        for record in records:
            self._doc_count += 1
            unique_factors = set(record.factors)
            for factor in unique_factors:
                self._doc_freq[factor] += 1
                if factor not in self._vocab:
                    self._vocab[factor] = len(self._vocab)

    def embed(self, record: StockMemRecord) -> np.ndarray:
        numeric = np.array(
            [
                float(record.sentiment_score),
                float(record.market_snapshot.rsi),
                float(record.market_snapshot.macd_hist),
            ],
            dtype=np.float32,
        )

        tfidf = np.zeros(len(self._vocab), dtype=np.float32)
        if record.factors and self._vocab:
            counts = Counter(record.factors)
            total = max(1, len(record.factors))
            for factor, count in counts.items():
                idx = self._vocab.get(factor)
                if idx is None:
                    continue
                tf = count / total
                df = self._doc_freq.get(factor, 0)
                idf = math.log((1 + self._doc_count) / (1 + df)) + 1.0
                tfidf[idx] = float(tf * idf)

        dense = np.concatenate([numeric, tfidf]).astype(np.float32)
        norm = np.linalg.norm(dense)
        if norm <= 1e-12:
            return dense
        return dense / norm
