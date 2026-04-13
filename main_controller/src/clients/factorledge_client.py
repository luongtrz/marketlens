"""FactorLedge module HTTP client."""

from shared.http_client import get_client
from shared.models.factor import NormalizedFactor


class FactorLedgeClient:
    """Async HTTP client for the FactorLedge module.

    Args:
        base_url: FactorLedge service base URL.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def health_check(self) -> bool:
        """Check if the FactorLedge service is healthy."""
        raise NotImplementedError

    async def ingest(
        self, article_id: str, factors: list[str], source: str
    ) -> list[NormalizedFactor]:
        """Send raw factors to FactorLedge for normalization.

        Args:
            article_id: Source article ID.
            factors: Raw factor strings.
            source: Source identifier.

        Returns:
            List of processed NormalizedFactor objects.
        """
        raise NotImplementedError
