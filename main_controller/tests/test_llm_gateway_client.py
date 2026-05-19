"""Tests for LLMGatewayClient."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from main_controller.src.clients.llm_gateway_client import LLMGatewayClient
from main_controller.tests.conftest import make_connect_error_client, make_get_client
from shared.models.market import OHLCV
from shared.models.memory import SimilarRecord
from shared.models.prediction import SignalType

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_ohlcv(close: float) -> OHLCV:
    return OHLCV(timestamp=_NOW, open=close, high=close * 1.005, low=close * 0.995, close=close, volume=1000.0, interval="1d")


def _record_with_rsi_and_3d_candles(base, rsi: float, ret_3d_pct: float):
    """Return record copy with given RSI and 3-day momentum.

    ret_3d_pct > 0 means price rose over 3 days (short_up); < 0 means fell (short_down).
    Requires ≥4 candles so the 3d momentum is computed from candles[-4].
    """
    close_now = 95500.0
    close_3d_ago = close_now / (1 + ret_3d_pct / 100)
    candles = [
        _make_ohlcv(close_3d_ago),
        _make_ohlcv(close_now * 0.99),
        _make_ohlcv(close_now * 1.00),
        _make_ohlcv(close_now),
    ]
    snap = base.market_snapshot.model_copy(
        update={"indicators": {"rsi": rsi, "macd_hist": 0.0}, "recent_candles": candles}
    )
    return base.model_copy(update={"market_snapshot": snap})


async def test_predict_timeout_falls_back_deterministic(monkeypatch, sample_stockmem_record):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_connect_error_client(),
    )
    client = LLMGatewayClient()
    result = await client.predict(current=sample_stockmem_record, similar=[])
    assert result.confidence >= 0.35
    assert "policy:gateway_client_fallback" in result.reasoning_steps
    assert any(step.startswith("client_fallback:") for step in result.reasoning_steps)


async def test_predict_bad_shape_falls_back_deterministic(monkeypatch, sample_stockmem_record):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client(["unexpected-list-shape"]),
    )
    client = LLMGatewayClient()
    result = await client.predict(current=sample_stockmem_record, similar=[])
    assert result.confidence >= 0.35
    assert "policy:gateway_client_fallback" in result.reasoning_steps


async def test_predict_passes_model_query_param(monkeypatch, sample_stockmem_record):
    calls: list[dict] = []

    @asynccontextmanager
    async def _mock(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"signal": "BUY", "reason": "ok"}
        mock_resp.raise_for_status.return_value = None
        mock_client = AsyncMock()

        async def _request(method, url, **kw):
            calls.append({"url": url, "params": kw.get("params")})
            return mock_resp

        mock_client.request = AsyncMock(side_effect=_request)
        yield mock_client

    monkeypatch.setattr("main_controller.src.clients.base.get_client", _mock)
    client = LLMGatewayClient()
    _ = await client.predict(current=sample_stockmem_record, similar=[], model="qwen3.5-plus")
    assert calls, "No HTTP request captured"
    assert calls[0]["params"]["model"] == "qwen3.5-plus"


def test_policy_confidence_gate_to_hold(sample_stockmem_record):
    client = LLMGatewayClient(min_directional_confidence=0.9, hold_release_bias=9.9)
    signal, confidence, notes = client._apply_regime_policy(SignalType.BUY, sample_stockmem_record, [])
    assert signal == SignalType.HOLD
    assert confidence <= 0.55
    assert "policy:confidence_gate_to_hold" in notes


def test_policy_hold_release_by_bias(sample_stockmem_record):
    bullish_record = sample_stockmem_record.model_copy(
        update={
            "future_return_7d": 4.2,
            "future_return_30d": 8.1,
        }
    )
    similar = [SimilarRecord(record=bullish_record, similarity=0.95)]
    client = LLMGatewayClient(min_directional_confidence=0.56, hold_release_bias=2.2)
    signal, confidence, notes = client._apply_regime_policy(SignalType.HOLD, sample_stockmem_record, similar)
    assert signal == SignalType.BUY
    assert confidence >= 0.56
    assert "policy:hold_release_by_bias" in notes


def _neutral_record(base):
    """Return a copy with flat MACD so regime guardrails don't fire before kNN veto."""
    snap = base.market_snapshot.model_copy(
        update={"indicators": {"rsi": 50.0, "macd_hist": 0.0, "price_change_pct": 0.3}}
    )
    return base.model_copy(update={"market_snapshot": snap})


def test_knn_veto_sell_when_similar_cases_bullish(sample_stockmem_record):
    """SELL is suppressed to HOLD when similar cases had positive avg 7d returns."""
    current = _neutral_record(sample_stockmem_record)
    bullish_case = sample_stockmem_record.model_copy(update={"future_return_7d": 3.5})
    similar = [SimilarRecord(record=bullish_case, similarity=0.9)]
    client = LLMGatewayClient(knn_confirm_threshold=1.0)
    signal, _, notes = client._apply_regime_policy(SignalType.SELL, current, similar)
    assert signal == SignalType.HOLD
    assert "policy:knn_veto_sell" in notes


def test_knn_veto_buy_when_similar_cases_bearish(sample_stockmem_record):
    """BUY is suppressed to HOLD when similar cases had negative avg 7d returns."""
    current = _neutral_record(sample_stockmem_record)
    bearish_case = sample_stockmem_record.model_copy(update={"future_return_7d": -2.5})
    similar = [SimilarRecord(record=bearish_case, similarity=0.9)]
    client = LLMGatewayClient(knn_confirm_threshold=1.0)
    signal, _, notes = client._apply_regime_policy(SignalType.BUY, current, similar)
    assert signal == SignalType.HOLD
    assert "policy:knn_veto_buy" in notes


def test_knn_veto_disabled_at_zero_threshold(sample_stockmem_record):
    """knn_confirm_threshold=0.0 disables the veto entirely."""
    current = _neutral_record(sample_stockmem_record)
    bullish_case = sample_stockmem_record.model_copy(update={"future_return_7d": 5.0})
    similar = [SimilarRecord(record=bullish_case, similarity=0.9)]
    client = LLMGatewayClient(knn_confirm_threshold=0.0, hold_release_bias=9.9)
    _, _, notes = client._apply_regime_policy(SignalType.SELL, current, similar)
    assert "policy:knn_veto_sell" not in notes


def test_rsi_exhaustion_blocks_buy_at_top(sample_stockmem_record):
    """BUY is blocked when RSI overbought AND 3d momentum already negative (top formation)."""
    # RSI=75 (overbought), 3d return=-2.5% (price already falling) → top formation
    current = _record_with_rsi_and_3d_candles(sample_stockmem_record, rsi=75.0, ret_3d_pct=-2.5)
    client = LLMGatewayClient(min_directional_confidence=0.0, hold_release_bias=9.9)
    signal, _, notes = client._apply_regime_policy(SignalType.BUY, current, [])
    assert signal == SignalType.HOLD
    assert "policy:block_buy_rsi_exhaustion" in notes


def test_rsi_oversold_blocks_sell_at_bottom(sample_stockmem_record):
    """SELL is blocked when RSI oversold AND 3d momentum already positive (bottom formation).

    Needs a bearish similar case (avg7 < 0) so knn_no_data_suppress_sell doesn't fire first.
    """
    # RSI=25 (oversold), 3d return=+2.5% (price already recovering) → bottom formation
    current = _record_with_rsi_and_3d_candles(sample_stockmem_record, rsi=25.0, ret_3d_pct=2.5)
    # avg7=-3.5 passes kNN veto (knn_sell_threshold=-3.0 vetoes SELL when avg7 > -3.0)
    # RSI oversold + price recovery should still override to HOLD via G6
    bearish_case = sample_stockmem_record.model_copy(update={"future_return_7d": -3.5})
    similar = [SimilarRecord(record=bearish_case, similarity=0.9)]
    client = LLMGatewayClient(min_directional_confidence=0.0, hold_release_bias=9.9)
    signal, _, notes = client._apply_regime_policy(SignalType.SELL, current, similar)
    assert signal == SignalType.HOLD
    assert "policy:block_sell_rsi_oversold" in notes


def test_rsi_normal_exhaustion_does_not_fire(sample_stockmem_record):
    """RSI in normal range does not trigger exhaustion guardrail even with negative 3d momentum."""
    # RSI=55 (neutral), 3d=-2.5% — exhaustion guardrail should NOT fire
    current = _record_with_rsi_and_3d_candles(sample_stockmem_record, rsi=55.0, ret_3d_pct=-2.5)
    client = LLMGatewayClient(min_directional_confidence=0.0, hold_release_bias=9.9)
    _, _, notes = client._apply_regime_policy(SignalType.BUY, current, [])
    assert "policy:block_buy_rsi_exhaustion" not in notes


def _record_with_multi_candles(base, rsi: float, ret_3d_pct: float, ret_14d_pct: float):
    """Build a record with enough candles to compute both 3d and 14d momentum."""
    close_now = 95500.0
    close_3d_ago = close_now / (1 + ret_3d_pct / 100)
    close_14d_ago = close_now / (1 + ret_14d_pct / 100)
    # 15 candles: index 0 = 14 days ago, index 11 = 3 days ago, index 14 = today
    candles = [_make_ohlcv(close_14d_ago)]
    for i in range(10):
        candles.append(_make_ohlcv(close_now * 0.999))
    candles.append(_make_ohlcv(close_3d_ago))  # index 11 = candles[-4] for 15-candle list
    candles.append(_make_ohlcv(close_now * 1.00))
    candles.append(_make_ohlcv(close_now * 1.00))
    candles.append(_make_ohlcv(close_now))  # index 14 = candles[-1]
    snap = base.market_snapshot.model_copy(
        update={"indicators": {"rsi": rsi, "macd_hist": 0.0}, "recent_candles": candles}
    )
    return base.model_copy(update={"market_snapshot": snap})


def test_dual_momentum_blocks_buy_in_early_correction(sample_stockmem_record):
    """BUY is blocked when 14d return <= -4% AND 3d momentum negative (early bear phase)."""
    # 14d=-5% (not yet at -6% bear_regime), 3d=-2.5%, RSI=55 (no RSI exhaustion)
    current = _record_with_multi_candles(sample_stockmem_record, rsi=55.0, ret_3d_pct=-2.5, ret_14d_pct=-5.0)
    client = LLMGatewayClient(min_directional_confidence=0.0, hold_release_bias=9.9)
    signal, _, notes = client._apply_regime_policy(SignalType.BUY, current, [])
    assert signal == SignalType.HOLD
    assert "policy:block_buy_dual_momentum_bear" in notes


def test_dual_momentum_blocks_sell_in_bull_recovery(sample_stockmem_record):
    """SELL is blocked when 14d return >= +4% AND 3d momentum positive (recovery phase)."""
    # avg7=-3.5 passes kNN veto (knn_sell_threshold=-3.0 vetoes SELL when avg7 > -3.0)
    # dual-timeframe momentum (14d=+5%, 3d=+2.5%) should still block SELL via G7
    bearish_case = sample_stockmem_record.model_copy(update={"future_return_7d": -3.5})
    similar = [SimilarRecord(record=bearish_case, similarity=0.9)]
    current = _record_with_multi_candles(sample_stockmem_record, rsi=55.0, ret_3d_pct=2.5, ret_14d_pct=5.0)
    client = LLMGatewayClient(min_directional_confidence=0.0, hold_release_bias=9.9)
    signal, _, notes = client._apply_regime_policy(SignalType.SELL, current, similar)
    assert signal == SignalType.HOLD
    assert "policy:block_sell_dual_momentum_bull" in notes


def test_dual_momentum_does_not_fire_in_shallow_correction(sample_stockmem_record):
    """Dual momentum does NOT fire when 14d return is shallow (-2%, above -4% threshold)."""
    current = _record_with_multi_candles(sample_stockmem_record, rsi=55.0, ret_3d_pct=-2.5, ret_14d_pct=-2.0)
    client = LLMGatewayClient(min_directional_confidence=0.0, hold_release_bias=9.9)
    _, _, notes = client._apply_regime_policy(SignalType.BUY, current, [])
    assert "policy:block_buy_dual_momentum_bear" not in notes


def test_adaptive_bias_releases_hold_in_bear_regime(sample_stockmem_record):
    """In confirmed bear regime (ret_14d <= -6%), effective_bias drops to 1.8.

    Setup: directional_bias=-2.0 (bear_regime -1.2 + avg7 -0.8).
    G3 does not fire (no short_down), so HOLD reaches G5.
    |−2.0| >= 1.8 (adaptive) → SELL released via hold_release_by_bias.
    """
    # ret_14d=-7% → bear_regime; ret_3d=-1% → NOT short_down (< -1.8 threshold)
    current = _record_with_multi_candles(sample_stockmem_record, rsi=50.0, ret_3d_pct=-1.0, ret_14d_pct=-7.0)
    bearish_case = sample_stockmem_record.model_copy(update={"future_return_7d": -2.0})
    similar = [SimilarRecord(record=bearish_case, similarity=0.9)]
    client = LLMGatewayClient(min_directional_confidence=0.0, hold_release_bias=2.8)
    signal, _, notes = client._apply_regime_policy(SignalType.HOLD, current, similar)
    # directional_bias = -1.2 (bear) + -0.8 (avg7=-2) = -2.0; |−2.0| >= 1.8 adaptive → SELL
    assert signal == SignalType.SELL
    assert "policy:hold_release_by_bias" in notes


def test_adaptive_bias_keeps_hold_outside_bear_regime(sample_stockmem_record):
    """Outside bear regime, effective_bias stays at 2.8 — moderate signals don't release HOLD.

    Setup: directional_bias=-2.2 (short_down -0.8 + avg7 -0.8 + avg30 -0.6).
    No bear_regime (ret_14d=-3%, avg30=-2%, macd=0) → effective_bias=2.8.
    |−2.2| < 2.8 → HOLD stays.
    """
    # ret_14d=-3% → NOT bear_regime; ret_3d=-2.5% → short_down
    current = _record_with_multi_candles(sample_stockmem_record, rsi=50.0, ret_3d_pct=-2.5, ret_14d_pct=-3.0)
    bearish_case = sample_stockmem_record.model_copy(
        update={"future_return_7d": -2.0, "future_return_30d": -2.0}
    )
    similar = [SimilarRecord(record=bearish_case, similarity=0.9)]
    client = LLMGatewayClient(min_directional_confidence=0.0, hold_release_bias=2.8)
    signal, _, notes = client._apply_regime_policy(SignalType.HOLD, current, similar)
    # directional_bias = -0.8 (short_down) + -0.8 (avg7) + -0.6 (avg30) = -2.2; |−2.2| < 2.8 → HOLD
    assert signal == SignalType.HOLD
    assert "policy:hold_release_by_bias" not in notes
