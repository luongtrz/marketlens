"""Vector index for similarity search (FAISS or pgvector)."""

import numpy as np


class VectorIndex:
    """Vector index for k-NN similarity search.

    Supports FAISS (in-memory) and pgvector (PostgreSQL) backends.

    Args:
        backend: "faiss" | "pgvector" | "memory".
        dimension: Embedding vector dimension.
    """

    def __init__(self, backend: str = "memory", dimension: int = 128) -> None:
        self._backend = backend
        self._dimension = dimension

    async def add(self, record_id: str, vector: np.ndarray) -> None:
        """Add a vector to the index.

        Args:
            record_id: ID associated with the vector.
            vector: Embedding vector.
        """
        raise NotImplementedError

    async def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query_vector: Query embedding vector.
            k: Number of nearest neighbors.

        Returns:
            List of (record_id, similarity_score) tuples, sorted by similarity descending.
        """
        raise NotImplementedError

    async def build(self) -> None:
        """Build or rebuild the index from stored vectors."""
        raise NotImplementedError
