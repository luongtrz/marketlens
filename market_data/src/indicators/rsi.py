"""RSI (Relative Strength Index) calculation."""

import numpy as np

from shared.models.market import OHLCV


def calculate_rsi(ohlcv: list[OHLCV], period: int = 14) -> float:
    closes = [c.close for c in ohlcv]
    if len(closes) < period + 1:
        return 50.0

    deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

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
