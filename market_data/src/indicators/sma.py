"""SMA (Simple Moving Average) — relative deviation from moving average."""

from shared.models.market import OHLCV


def calculate_sma(ohlcv: list[OHLCV], period: int = 20) -> float:
    """Calculate relative SMA deviation: (close[-1] - sma_20) / sma_20.

    Returns how far the current price is from its moving average as a fraction.
    Scale-invariant: +0.05 means 5% above MA regardless of absolute price level.

    Args:
        ohlcv: List of OHLCV candles (oldest first).
        period: Moving average period (default 20).

    Returns:
        Relative deviation in (-inf, +inf), typically small (e.g. ±0.10).
        Returns 0.0 when insufficient data or sma is zero.
    """
    closes = [c.close for c in ohlcv]
    if len(closes) < 2:
        return 0.0

    window = closes[-period:] if len(closes) >= period else closes
    sma_abs = sum(window) / len(window)

    if sma_abs == 0.0:
        return 0.0

    return (closes[-1] - sma_abs) / sma_abs
