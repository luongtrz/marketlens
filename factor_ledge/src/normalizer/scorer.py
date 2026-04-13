"""Factor scorer: assign weights based on frequency and recency."""


class FactorScorer:
    """Assigns importance weights to cleaned factors."""

    def score(self, factors: list[str]) -> dict[str, float]:
        """Assign a weight [0, 1] to each factor based on frequency and recency.

        Args:
            factors: List of cleaned factor strings.

        Returns:
            Dict mapping factor name to weight.
        """
        raise NotImplementedError
