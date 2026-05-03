"""MACD (Moving Average Convergence Divergence) calculation."""

from typing import Any

import numpy as np

from shared.models.market import OHLCV


def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    ema = np.zeros(len(data))
    multiplier = 2 / (period + 1)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = (data[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
    return ema


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
        Returns empty lists if insufficient data.
    """
    if len(ohlcv) < slow_period:
        return {"macd": [], "signal": [], "histogram": []}

    closes = np.array([c.close for c in ohlcv])

    fast_ema = _calculate_ema(closes, fast_period)
    slow_ema = _calculate_ema(closes, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = _calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return {
        "macd": macd_line.tolist(),
        "signal": signal_line.tolist(),
        "histogram": histogram.tolist(),
    }
