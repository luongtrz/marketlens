"""k-NN search over the vector index."""

from shared.models.memory import StockMemRecord, SimilarRecord


class RecordSearcher:
    """Searches for similar historical records using vector similarity.

    Args:
        db_url: Database URL for record retrieval.
        vector_backend: Backend for vector index ("faiss" | "pgvector" | "memory").
    """

    def __init__(self, db_url: str, vector_backend: str = "memory") -> None:
        self._db_url = db_url
        self._vector_backend = vector_backend

    async def search(self, query: StockMemRecord, k: int = 5) -> list[SimilarRecord]:
        """Find the k most similar historical records.

        Args:
            query: Current record to find similar records for.
            k: Number of similar records to retrieve.

        Returns:
            List of SimilarRecord objects with similarity scores.
        """
        raise NotImplementedError
