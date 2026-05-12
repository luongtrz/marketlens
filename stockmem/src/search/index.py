from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


@dataclass
class ScoredId:
    record_id: str
    score: float


class MemoryVectorIndex:
    """
    In-memory vector index.
    Uses FAISS IndexFlatIP when available; falls back to NumPy cosine search.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._faiss_index = None
        self._dim: int | None = None
        self._faiss_dirty = False

    def rebuild(self, entries: Iterable[tuple[str, np.ndarray]]) -> None:
        self._vectors = {rid: vec.astype(np.float32) for rid, vec in entries}
        self._ids = list(self._vectors.keys())
        self._dim = None
        self._faiss_index = None
        self._faiss_dirty = False

        if not self._ids:
            return

        first = self._vectors[self._ids[0]]
        self._dim = int(first.shape[0])

        if faiss is not None:
            mat = np.vstack([self._vectors[rid] for rid in self._ids]).astype(np.float32)
            self._faiss_index = faiss.IndexFlatIP(self._dim)
            self._faiss_index.add(mat)

    def upsert(self, record_id: str, vector: np.ndarray) -> None:
        vec = vector.astype(np.float32)
        if record_id not in self._vectors:
            self._ids.append(record_id)
        self._vectors[record_id] = vec
        self._dim = int(vec.shape[0])
        # For FAISS, updating an arbitrary id requires rebuild. Defer until search.
        self._faiss_dirty = True
        if faiss is None:
            self._faiss_index = None

    def remove(self, record_id: str) -> None:
        if record_id not in self._vectors:
            return
        del self._vectors[record_id]
        self._ids = [rid for rid in self._ids if rid != record_id]
        self._faiss_dirty = True
        if not self._ids:
            self._dim = None
            self._faiss_index = None

    def _refresh_faiss_if_needed(self) -> None:
        if faiss is None:
            return
        if not self._faiss_dirty and self._faiss_index is not None:
            return
        if not self._ids:
            self._faiss_index = None
            self._faiss_dirty = False
            return
        self._dim = int(self._vectors[self._ids[0]].shape[0])
        mat = np.vstack([self._vectors[rid] for rid in self._ids]).astype(np.float32)
        self._faiss_index = faiss.IndexFlatIP(self._dim)
        self._faiss_index.add(mat)
        self._faiss_dirty = False

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredId]:
        if not self._ids:
            return []

        k_eff = max(1, min(k, len(self._ids)))
        q = query_vector.astype(np.float32).reshape(1, -1)

        self._refresh_faiss_if_needed()
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, k_eff)
            out: list[ScoredId] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                out.append(ScoredId(record_id=self._ids[idx], score=float(score)))
            return out

        mat = np.vstack([self._vectors[rid] for rid in self._ids]).astype(np.float32)
        sims = mat @ q[0]
        order = np.argsort(-sims)[:k_eff]
        return [ScoredId(record_id=self._ids[i], score=float(sims[i])) for i in order]
