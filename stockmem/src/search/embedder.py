from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from ..models import StockMemRecord


@dataclass(frozen=True)
class SplitEmbedding:
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray


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

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            return vec
        return vec / norm

    def embed_split(self, record: StockMemRecord) -> SplitEmbedding:
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

        indicator = np.array(
            [
                float(record.sentiment_score),
                float(record.market_snapshot.rsi),
                float(record.market_snapshot.macd_hist),
                float(record.market_snapshot.msi or 0.0),
                float(record.market_snapshot.fear_greed_index or 0.0),
            ],
            dtype=np.float32,
        )

        price = np.array(
            [
                float(record.market_snapshot.price_change_pct or 0.0),
            ],
            dtype=np.float32,
        )

        return SplitEmbedding(
            factor_vec=self._l2_normalize(tfidf.astype(np.float32)),
            indicator_vec=self._l2_normalize(indicator.astype(np.float32)),
            price_vec=self._l2_normalize(price.astype(np.float32)),
        )

    def embed(self, record: StockMemRecord) -> np.ndarray:
        split = self.embed_split(record)
        dense = np.concatenate(
            [split.factor_vec, split.indicator_vec, split.price_vec],
            dtype=np.float32,
        )
        return self._l2_normalize(dense.astype(np.float32))
