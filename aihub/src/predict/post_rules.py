"""Deterministic post-prediction rules shared by API and offline evaluators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shared.models.prediction import SignalType


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pct_change(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / old * 100.0


def _extract_close(snapshot: Any, offset_from_end: int = 0) -> float | None:
    ohlcv = getattr(snapshot, "ohlcv", None)
    if offset_from_end == 0 and ohlcv is not None:
        return _safe_float(getattr(ohlcv, "close", None))

    recent = list(getattr(snapshot, "recent_candles", None) or [])
    if recent:
        index = len(recent) - 1 - offset_from_end
        if 0 <= index < len(recent):
            return _safe_float(getattr(recent[index], "close", None))

    candles = list(getattr(snapshot, "candles", None) or [])
    if candles:
        index = len(candles) - 1 - offset_from_end
        if 0 <= index < len(candles):
            return _safe_float(getattr(candles[index], "close", None))

    return None


def apply_post_signal_rules(
    signal: SignalType,
    confidence: float,
    current_record: Any,
    config: Any,
) -> tuple[SignalType, list[str]]:
    """Apply deterministic guardrails after LLM signal generation."""
    if not getattr(config, "post_rule_enabled", False):
        return signal, []

    notes: list[str] = []

    snapshot = getattr(current_record, "market_snapshot", None)
    if snapshot is None:
        return signal, notes

    indicators = getattr(snapshot, "indicators", None) or {}
    macd_hist = _safe_float(indicators.get("macd_hist"))

    close_now = _extract_close(snapshot, 0)
    close_3d = _extract_close(snapshot, 3)
    close_30d = _extract_close(snapshot, 19)

    ret_3d = _pct_change(close_3d, close_now)
    ret_30d = _pct_change(close_30d, close_now)

    bullish_30d = (
        ret_30d is not None
        and ret_30d >= float(getattr(config, "post_rule_bull_30d_pct", 10.0))
    )
    bearish_30d = (
        ret_30d is not None
        and ret_30d <= float(getattr(config, "post_rule_bear_30d_pct", -10.0))
    )
    clear_up_3d = (
        ret_3d is not None and ret_3d >= float(getattr(config, "post_rule_up_3d_pct", 3.0))
    )
    clear_down_3d = (
        ret_3d is not None
        and ret_3d <= float(getattr(config, "post_rule_down_3d_pct", -3.0))
    )
    macd_eps = float(getattr(config, "post_rule_macd_confirm_eps", 0.0))
    macd_buy_ok = macd_hist is not None and macd_hist >= macd_eps
    macd_sell_ok = macd_hist is not None and macd_hist <= -macd_eps

    if signal == SignalType.SELL and bullish_30d and not (clear_down_3d and macd_sell_ok):
        signal = SignalType.HOLD
        notes.append(
            "post-rule: blocked SELL because 30d trend is strongly bullish without "
            "confirmed 3d+MACD breakdown."
        )

    hold_override_cap = float(getattr(config, "post_rule_hold_override_max_conf", 0.72))
    if signal == SignalType.HOLD and confidence <= hold_override_cap:
        if clear_up_3d and not bearish_30d and macd_buy_ok:
            signal = SignalType.BUY
            notes.append(
                "post-rule: upgraded HOLD -> BUY because 3d momentum+MACD confirms "
                "upside and 30d regime is not bearish."
            )
        elif clear_down_3d and not bullish_30d and macd_sell_ok:
            signal = SignalType.SELL
            notes.append(
                "post-rule: downgraded HOLD -> SELL because 3d momentum+MACD confirms "
                "downside and 30d regime is not bullish."
            )

    return signal, notes


def apply_knn_confirmation_rule(
    signal: SignalType,
    similar_records: Sequence[Any],
    threshold: float,
) -> tuple[SignalType, list[str]]:
    """Suppress directional signals when similar-case outcomes contradict them."""
    if threshold <= 0.0 or not similar_records:
        return signal, []

    sim7_vals = [
        rec.record.future_return_7d
        for rec in similar_records
        if getattr(rec.record, "future_return_7d", None) is not None
    ]
    notes: list[str] = []

    if not sim7_vals and signal == SignalType.SELL:
        return SignalType.HOLD, ["post-rule: knn_no_data_suppress_sell"]

    if sim7_vals:
        avg7 = sum(sim7_vals) / len(sim7_vals)
        if signal == SignalType.SELL and avg7 > threshold:
            return SignalType.HOLD, ["post-rule: knn_veto_sell (similar cases bullish)"]
        if signal == SignalType.BUY and avg7 < -threshold:
            return SignalType.HOLD, ["post-rule: knn_veto_buy (similar cases bearish)"]

    return signal, notes
