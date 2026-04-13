"""MACD (Moving Average Convergence Divergence) calculation."""

from typing import Any

from shared.models.market import OHLCV


def calculate_macd(
    ohlcv: list[OHLCV],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, Any]:
    """Calculate MACD indicator from OHLCV data.

    Args:
        ohlcv: List of OHLCV candles.
        fast_period: Fast EMA period.
        slow_period: Slow EMA period.
        signal_period: Signal line EMA period.

    Returns:
        Dict with keys: macd, signal, histogram (each a list of floats).
    """
    raise NotImplementedError
