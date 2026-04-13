"""Indicator registry — maps indicator names to calculation functions."""

from typing import Any, Callable

from shared.models.market import OHLCV
from market_data.src.indicators.macd import calculate_macd
from market_data.src.indicators.rsi import calculate_rsi
from market_data.src.indicators.bollinger import calculate_bollinger


def calculate_ema(ohlcv: list[OHLCV], period: int = 20) -> list[float]:
    """Calculate Exponential Moving Average.

    Args:
        ohlcv: List of OHLCV candles.
        period: EMA lookback period.

    Returns:
        List of EMA values.
    """
    raise NotImplementedError


def calculate_vwap(ohlcv: list[OHLCV]) -> list[float]:
    """Calculate Volume-Weighted Average Price.

    Args:
        ohlcv: List of OHLCV candles.

    Returns:
        List of VWAP values.
    """
    raise NotImplementedError


INDICATOR_REGISTRY: dict[str, Callable[[list[OHLCV]], Any]] = {
    "macd": calculate_macd,
    "rsi": calculate_rsi,
    "bb": calculate_bollinger,
    "ema": calculate_ema,
    "vwap": calculate_vwap,
}


def calculate_indicators(ohlcv: list[OHLCV], names: list[str]) -> dict[str, Any]:
    """Calculate multiple indicators at once.

    Args:
        ohlcv: List of OHLCV candles.
        names: List of indicator names (must exist in INDICATOR_REGISTRY).

    Returns:
        Dict mapping indicator name to its computed result.

    Raises:
        KeyError: If an indicator name is not found in the registry.
    """
    return {name: INDICATOR_REGISTRY[name](ohlcv) for name in names}
