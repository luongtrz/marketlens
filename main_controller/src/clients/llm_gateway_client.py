"""LLM Gateway module HTTP client."""

from shared.models.memory import SimilarRecord, StockMemRecord
from shared.models.prediction import PredictResponse, SignalType

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import LLMGatewayClientError


class LLMGatewayClient(BaseHTTPClient):
    """Async HTTP client for the llm_gateway module."""

    def __init__(
        self,
        base_url: str = "http://localhost:8006",
        *,
        min_directional_confidence: float = 0.56,
        hold_release_bias: float = 2.2,
        knn_confirm_threshold: float = 1.0,
    ) -> None:
        super().__init__(base_url, LLMGatewayClientError)
        self._min_directional_confidence = float(min_directional_confidence)
        self._hold_release_bias = float(hold_release_bias)
        self._knn_confirm_threshold = float(knn_confirm_threshold)

    async def health_check(self) -> bool:
        body = await self._get("/health")
        return body.get("status") == "ok"  # type: ignore[union-attr]

    async def predict(
        self,
        current: StockMemRecord,
        similar: list[SimilarRecord],
        model: str | None = None,
    ) -> PredictResponse:
        prompt = self._build_prompt(current, similar)
        fallback_reason = ""
        try:
            params = {"reason": "true"}
            if model:
                params["model"] = model
            body = await self._request(  # pylint: disable=protected-access
                "POST",
                "/llm",
                params=params,
                json={"prompt": prompt},
                timeout=20.0,
            )
            if not isinstance(body, dict):
                raise LLMGatewayClientError("Unexpected /llm response shape (expected JSON object)")
            raw_signal = str(body.get("signal", "HOLD")).upper()
            try:
                signal = SignalType(raw_signal)
            except ValueError:
                signal = SignalType.HOLD
            reason = str(body.get("reason", "")).strip()
            if reason.startswith("llm_gateway_fallback:"):
                signal, reason = self._deterministic_fallback(current, similar)
        except Exception as exc:  # noqa: BLE001 - prevent pipeline hard-fail on LLM instability
            signal, reason = self._deterministic_fallback(current, similar)
            fallback_reason = f"client_fallback:{type(exc).__name__}"
        signal, confidence, policy_steps = self._apply_regime_policy(signal, current, similar)
        explanation = reason or "LLM Gateway decision without reason."
        return PredictResponse(
            signal=signal,
            confidence=confidence,
            explanation=explanation,
            reasoning_steps=[
                "predictor=llm_gateway",
                f"n_similar={len(similar)}",
                *(["policy:gateway_client_fallback"] if fallback_reason else []),
                *([fallback_reason] if fallback_reason else []),
                *policy_steps,
            ],
        )

    @staticmethod
    def _build_prompt(current: StockMemRecord, similar: list[SimilarRecord]) -> str:
        current_close = getattr(current.market_snapshot.ohlcv, "close", None)
        current_rsi = (current.market_snapshot.indicators or {}).get("rsi")
        current_macd = (current.market_snapshot.indicators or {}).get("macd_hist")
        current_msi = (current.market_snapshot.indicators or {}).get("msi")
        current_fng = (current.market_snapshot.indicators or {}).get("fear_greed_index")
        current_pchg = (current.market_snapshot.indicators or {}).get("price_change_pct")
        lines = [
            f"date={current.date}",
            f"symbol={current.symbol}",
            f"sentiment_score={current.sentiment_score}",
            f"close={current_close}",
            f"rsi={current_rsi}",
            f"macd_hist={current_macd}",
            f"msi={current_msi}",
            f"fear_greed_index={current_fng}",
            f"price_change_pct={current_pchg}",
            f"factors={', '.join(current.factors[:4]) or 'none'}",
        ]
        today_record = "\n".join(lines)

        similar_lines = []
        if similar:
            ret7_vals = [c.record.future_return_7d for c in similar[:5] if c.record.future_return_7d is not None]
            if ret7_vals:
                knn_avg = sum(ret7_vals) / len(ret7_vals)
                knn_pos = sum(1 for r in ret7_vals if r > 0)
                similar_lines.append(
                    f"[knn_summary] knn_avg_7d={knn_avg:+.2f}%, "
                    f"knn_bullish_count={knn_pos}/{len(ret7_vals)}"
                )
            for i, case in enumerate(similar[:3], start=1):
                rec = case.record
                ret1 = rec.future_return_1d
                ret7 = rec.future_return_7d
                ret30 = rec.future_return_30d
                similar_lines.append(
                    f"{i}. date={rec.date}, sim={case.similarity:.4f}, "
                    f"ret1d={ret1}, ret7d={ret7}, ret30d={ret30}, sentiment={rec.sentiment_score}"
                )
        else:
            similar_lines.append("No similar cases available.")
        similar_records = "\n".join(similar_lines)
        return (
            "=== Current Situation ===\n"
            f"{today_record}\n\n"
            "=== Similar Historical Cases (from StockMem) ===\n"
            f"{similar_records}"
        )

    @classmethod
    def _deterministic_fallback(
        cls, current: StockMemRecord, similar: list[SimilarRecord]
    ) -> tuple[SignalType, str]:
        """Deterministic fallback when gateway upstream model is unavailable.

        Keeps pipeline responsive and avoids hard HOLD(0.00) on transient LLM failures.
        """
        snap = current.market_snapshot
        indicators = snap.indicators or {}
        candles = list(snap.recent_candles or [])
        close_now = cls._safe_float(getattr(snap.ohlcv, "close", None))
        close_3d = cls._safe_float(getattr(candles[-4], "close", None)) if len(candles) >= 4 else None
        close_14d = cls._safe_float(getattr(candles[-15], "close", None)) if len(candles) >= 15 else None
        ret_3d = cls._pct_change(close_3d, close_now) or 0.0
        ret_14d = cls._pct_change(close_14d, close_now) or 0.0
        macd = cls._safe_float(indicators.get("macd_hist")) or 0.0

        sim7 = [cls._safe_float(c.record.future_return_7d) for c in similar[:3]]
        sim7_vals = [v for v in sim7 if v is not None]
        avg7 = (sum(sim7_vals) / len(sim7_vals)) if sim7_vals else 0.0

        if (ret_14d >= 4.0 and ret_3d >= 1.2 and macd >= 0) or avg7 >= 2.0:
            return SignalType.BUY, "deterministic_fallback: bullish regime + positive momentum"
        if (ret_14d <= -4.0 and ret_3d <= -1.2 and macd <= 0) or avg7 <= -2.0:
            return SignalType.SELL, "deterministic_fallback: bearish regime + negative momentum"
        return SignalType.HOLD, "deterministic_fallback: mixed regime"

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @classmethod
    def _pct_change(cls, old: object, new: object) -> float | None:
        o = cls._safe_float(old)
        n = cls._safe_float(new)
        if o is None or n is None or o == 0:
            return None
        return (n - o) / o * 100.0

    def _apply_regime_policy(
        self, signal: SignalType, current: StockMemRecord, similar: list[SimilarRecord]
    ) -> tuple[SignalType, float, list[str]]:
        """Calibrate signal by market regime + similar-case bias.

        Goal: reduce false SELL/HOLD in strong trend regimes.
        """
        notes: list[str] = []

        snap = current.market_snapshot
        indicators = snap.indicators or {}
        candles = list(snap.recent_candles or [])

        close_now = self._safe_float(getattr(snap.ohlcv, "close", None))
        close_3d = self._safe_float(getattr(candles[-4], "close", None)) if len(candles) >= 4 else None
        close_14d = self._safe_float(getattr(candles[-15], "close", None)) if len(candles) >= 15 else None

        ret_3d = self._pct_change(close_3d, close_now)
        ret_14d = self._pct_change(close_14d, close_now)
        macd = self._safe_float(indicators.get("macd_hist"))
        rsi = self._safe_float(indicators.get("rsi"))

        sim7: list[float] = []
        sim30: list[float] = []
        for case in similar[:5]:
            r7 = self._safe_float(case.record.future_return_7d)
            r30 = self._safe_float(case.record.future_return_30d)
            if r7 is not None:
                sim7.append(r7)
            if r30 is not None:
                sim30.append(r30)
        avg7 = (sum(sim7) / len(sim7)) if sim7 else 0.0
        avg30 = (sum(sim30) / len(sim30)) if sim30 else 0.0
        knn_has_data = len(sim7) > 0

        bull_regime = (
            (ret_14d is not None and ret_14d >= 6.0)
            or (avg30 >= 3.0)
            or (macd is not None and macd > 0 and avg7 >= 1.0)
        )
        bear_regime = (
            (ret_14d is not None and ret_14d <= -6.0)
            or (avg30 <= -3.0)
            or (macd is not None and macd < 0 and avg7 <= -1.0)
        )
        short_up = ret_3d is not None and ret_3d >= 1.8
        short_down = ret_3d is not None and ret_3d <= -1.8

        # Guardrail 1: suppress SELL in strong bullish regime without clear breakdown.
        if signal == SignalType.SELL and bull_regime and not (short_down and avg7 <= -1.0):
            signal = SignalType.HOLD
            notes.append("policy:block_sell_in_bull_regime")

        # Guardrail 2: suppress BUY in strong bearish regime without clear reversal.
        if signal == SignalType.BUY and bear_regime and not (short_up and avg7 >= 1.0):
            signal = SignalType.HOLD
            notes.append("policy:block_buy_in_bear_regime")

        # Guardrail 3: avoid indecisive HOLD when regime and short momentum align.
        if signal == SignalType.HOLD:
            if bull_regime and short_up and avg7 >= 0:
                signal = SignalType.BUY
                notes.append("policy:hold_to_buy")
            elif bear_regime and short_down and avg7 <= 0:
                signal = SignalType.SELL
                notes.append("policy:hold_to_sell")

        # kNN confirmation: if LLM signal and similar-case outcomes contradict, override to HOLD.
        # Empirically: SELL accuracy is 40.8% when kNN disagrees — below random.
        # Also suppress SELL when no kNN return data is available (can't confirm).
        if self._knn_confirm_threshold > 0.0:
            if signal == SignalType.SELL and not knn_has_data:
                signal = SignalType.HOLD
                notes.append("policy:knn_no_data_suppress_sell")
            elif signal == SignalType.SELL and avg7 > self._knn_confirm_threshold:
                signal = SignalType.HOLD
                notes.append("policy:knn_veto_sell")
            elif signal == SignalType.BUY and avg7 < -self._knn_confirm_threshold:
                signal = SignalType.HOLD
                notes.append("policy:knn_veto_buy")

        # Guardrail 6: RSI extremes signal price exhaustion — momentum has already reversed
        # even though kNN regime still looks directional. Block at turning points.
        # BUY + overbought RSI + negative 3d momentum = top formation.
        # SELL + oversold RSI + positive 3d momentum = bottom formation.
        if signal == SignalType.BUY and rsi is not None and rsi > 70 and short_down:
            signal = SignalType.HOLD
            notes.append("policy:block_buy_rsi_exhaustion")
        elif signal == SignalType.SELL and rsi is not None and rsi < 30 and short_up:
            signal = SignalType.HOLD
            notes.append("policy:block_sell_rsi_oversold")

        # Guardrail 7: dual-timeframe momentum confirmation — fires earlier than Guardrail 2
        # (bear_regime needs 14d <= -6%; this needs 14d <= -4% + 3d already negative).
        # Catches the first 2 weeks of a correction before the full regime flag triggers.
        # Symmetric: blocks SELL when both timeframes confirm bullish recovery.
        if signal == SignalType.BUY and ret_14d is not None and ret_14d <= -4.0 and short_down:
            signal = SignalType.HOLD
            notes.append("policy:block_buy_dual_momentum_bear")
        elif signal == SignalType.SELL and ret_14d is not None and ret_14d >= 4.0 and short_up:
            signal = SignalType.HOLD
            notes.append("policy:block_sell_dual_momentum_bull")

        directional_bias = 0.0
        if bull_regime:
            directional_bias += 1.2
        if bear_regime:
            directional_bias -= 1.2
        if short_up:
            directional_bias += 0.8
        if short_down:
            directional_bias -= 0.8
        if avg7 >= 1.0:
            directional_bias += 0.8
        elif avg7 <= -1.0:
            directional_bias -= 0.8
        if avg30 >= 1.5:
            directional_bias += 0.6
        elif avg30 <= -1.5:
            directional_bias -= 0.6
        if macd is not None and macd > 0:
            directional_bias += 0.3
        elif macd is not None and macd < 0:
            directional_bias -= 0.3

        # Confidence calibration from evidence consistency.
        evidence = 0.0
        if signal == SignalType.BUY:
            evidence += 1.8 if bull_regime else -1.8
            evidence += 1.2 if short_up else (-1.0 if short_down else 0.0)
            evidence += 0.9 if avg7 >= 0 else -0.9
            evidence += 0.6 if avg30 >= 0 else -0.6
            evidence += 0.4 if (rsi is not None and rsi < 68) else -0.2
        elif signal == SignalType.SELL:
            evidence += 1.8 if bear_regime else -1.8
            evidence += 1.2 if short_down else (-1.0 if short_up else 0.0)
            evidence += 0.9 if avg7 <= 0 else -0.9
            evidence += 0.6 if avg30 <= 0 else -0.6
            evidence += 0.4 if (rsi is not None and rsi > 32) else -0.2
        else:
            # HOLD should remain lower confidence unless evidence is strongly mixed.
            mixed = (bull_regime and bear_regime) or (abs(avg7) < 1.0 and abs(avg30) < 1.5)
            evidence += 0.4 if mixed else -0.8

        confidence = 0.5 + max(-2.0, min(2.0, evidence)) * 0.12
        if signal == SignalType.HOLD:
            confidence = min(confidence, 0.62)
        confidence = max(0.35, min(0.9, confidence))

        # Guardrail 4: confidence gate for directional signals (trade-off: lower coverage, higher precision).
        if signal in {SignalType.BUY, SignalType.SELL} and confidence < self._min_directional_confidence:
            signal = SignalType.HOLD
            confidence = min(confidence, 0.55)
            notes.append("policy:confidence_gate_to_hold")

        # Guardrail 5: release HOLD when directional bias is clearly one-sided.
        if signal == SignalType.HOLD and abs(directional_bias) >= self._hold_release_bias:
            signal = SignalType.BUY if directional_bias > 0 else SignalType.SELL
            confidence = max(confidence, self._min_directional_confidence)
            notes.append("policy:hold_release_by_bias")

        return signal, round(confidence, 4), notes
