"""Bollinger Bands calculation."""

from typing import Any

from shared.models.market import OHLCV


def calculate_bollinger(
    ohlcv: list[OHLCV], period: int = 20, num_std: float = 2.0
) -> dict[str, Any]:
    """Calculate Bollinger Bands from OHLCV data.

    Args:
        ohlcv: List of OHLCV candles.
        period: Moving average period.
        num_std: Number of standard deviations for bands.

    Returns:
        Dict with keys: upper, middle, lower (each a list of floats).
    """
    raise NotImplementedError
