"""Indicator registry — maps indicator names to calculation functions."""

from typing import Any, Callable

import numpy as np

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
    if len(ohlcv) < period:
        return []

    closes = np.array([c.close for c in ohlcv])
    ema = np.zeros(len(closes))
    multiplier = 2 / (period + 1)
    ema[0] = closes[0]

    for i in range(1, len(closes)):
        ema[i] = (closes[i] * multiplier) + (ema[i - 1] * (1 - multiplier))

    return ema.tolist()


def calculate_vwap(ohlcv: list[OHLCV]) -> list[float]:
    """Calculate Volume-Weighted Average Price.

    Args:
        ohlcv: List of OHLCV candles.

    Returns:
        List of VWAP values.
    """
    if not ohlcv:
        return []

    vwap = []
    cumulative_volume = 0.0
    cumulative_pv = 0.0  # price * volume

    for candle in ohlcv:
        typical_price = (candle.high + candle.low + candle.close) / 3
        cumulative_volume += candle.volume
        cumulative_pv += typical_price * candle.volume
        vwap.append(cumulative_pv / cumulative_volume if cumulative_volume > 0 else 0.0)

    return vwap


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
