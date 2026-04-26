"""MACD (Moving Average Convergence Divergence) calculation."""

from typing import Any

from shared.models.market import OHLCV


def _ema(data: list[float], period: int) -> list[float]:
    k = 2 / (period + 1)
    result = [data[0]]
    for price in data[1:]:
        result.append(price * k + result[-1] * (1 - k))
    return result


def calculate_macd(
    ohlcv: list[OHLCV],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, Any]:
    closes = [c.close for c in ohlcv]
    if len(closes) < slow_period:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    fast_ema = _ema(closes, fast_period)
    slow_ema = _ema(closes, slow_period)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = _ema(macd_line, signal_period)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]

    return {
        "macd": round(macd_line[-1], 6),
        "signal": round(signal_line[-1], 6),
        "histogram": round(histogram[-1], 6),
    }
