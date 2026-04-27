"""RSI (Relative Strength Index) calculation."""

import numpy as np

from shared.models.market import OHLCV


def calculate_rsi(ohlcv: list[OHLCV], period: int = 14) -> float:
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        ohlcv: List of OHLCV data points
        period: RSI calculation period
        
    Returns:
        RSI value (0-100), or 50.0 if insufficient data.
    """
    if len(ohlcv) < period + 1:
        return 50.0  # Neutral RSI when insufficient data

    closes = np.array([c.close for c in ohlcv[-(period + 1):]])
    deltas = np.diff(closes)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)
