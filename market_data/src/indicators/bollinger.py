"""Bollinger Bands calculation."""

from typing import Any

import numpy as np

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
        Returns empty lists if insufficient data.
    """
    if len(ohlcv) < period:
        return {"upper": [], "middle": [], "lower": []}

    closes = np.array([c.close for c in ohlcv])

    # Calculate rolling mean and std
    middle = np.zeros(len(closes))
    upper = np.zeros(len(closes))
    lower = np.zeros(len(closes))

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        mean = np.mean(window)
        std = np.std(window)
        middle[i] = mean
        upper[i] = mean + (num_std * std)
        lower[i] = mean - (num_std * std)

    return {
        "upper": upper.tolist(),
        "middle": middle.tolist(),
        "lower": lower.tolist(),
    }
