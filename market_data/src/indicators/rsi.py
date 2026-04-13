"""RSI (Relative Strength Index) calculation."""

from shared.models.market import OHLCV


def calculate_rsi(ohlcv: list[OHLCV], period: int = 14) -> float:
    """Calculate RSI indicator from OHLCV data.

    Args:
        ohlcv: List of OHLCV candles.
        period: RSI lookback period.

    Returns:
        RSI value (0-100).
    """
    raise NotImplementedError
