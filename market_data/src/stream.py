"""On-demand indicator calculation stream."""

from typing import Any

from shared.models.market import OHLCV


class IndicatorStream:
    """Provides on-demand indicator calculation from live or cached OHLCV data."""

    async def calculate(
        self, symbol: str, interval: str, indicators: list[str]
    ) -> dict[str, Any]:
        """Fetch OHLCV data and calculate requested indicators.

        Args:
            symbol: Trading pair.
            interval: Candle interval.
            indicators: List of indicator names to calculate.

        Returns:
            Dict mapping indicator name to computed result.
        """
        raise NotImplementedError
