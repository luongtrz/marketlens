"""RAG context builder — assembles prompt context from current + similar records."""

from shared.models.memory import StockMemRecord, SimilarRecord


class RAGContextBuilder:
    """Builds RAG prompt context from current record and similar historical cases."""

    def build(self, current: StockMemRecord, similar: list[SimilarRecord]) -> str:
        """Assemble a prompt context string from the current record and k similar cases.

        Format::

            === Current Situation ===
            <current record fields>
            === Similar Historical Cases ===
            Case 1 (similarity=0.92, date=...):
              <fields>
              Outcome: <what happened>
            ...

        Args:
            current: The current pipeline run's StockMemRecord.
            similar: List of similar historical records with similarity scores.

        Returns:
            Formatted context string for the prediction prompt.
        """
        raise NotImplementedError
