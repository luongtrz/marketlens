"""Factor enricher: attach sector and asset metadata."""


class FactorEnricher:
    """Enriches factors with sector, asset class, and related symbol metadata.

    Uses a local static lookup table — no external network calls required.
    """

    def enrich(self, factor_name: str) -> dict:
        """Enrich a single factor with metadata from the lookup table.

        Args:
            factor_name: Cleaned factor name.

        Returns:
            Dict with keys: sector, asset_class, related_symbols.
        """
        raise NotImplementedError
