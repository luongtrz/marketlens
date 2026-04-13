"""Factor processing pipeline: clean → normalize → enrich."""

from shared.models.factor import NormalizedFactor


class FactorPipeline:
    """Orchestrates the factor normalization pipeline.

    Steps:
    1. Cleaner: lowercase, dedup, remove stopwords, trim noise tokens
    2. Scorer: assign a weight [0,1] to each factor based on frequency + recency
    3. Enricher: attach { sector, asset_class, related_symbols } from a lookup table
    """

    def run(self, raw: list[str], article_id: str) -> list[NormalizedFactor]:
        """Run the full factor normalization pipeline.

        Args:
            raw: List of raw factor strings from the LLM.
            article_id: ID of the source article.

        Returns:
            List of cleaned, scored, and enriched NormalizedFactor objects.
        """
        raise NotImplementedError
