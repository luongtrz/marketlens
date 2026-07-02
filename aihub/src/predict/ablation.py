"""Naive current-context AI evaluation helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aihub.src.llm.base import LLMClient
from stockmem.scripts.ndjson_eval_common import (
    HeadConfig,
    PredictionMetrics,
    actual_signal,
    configured_head_signal,
    knn_returns_signal,
    load_head_config,
    load_historical_rows,
    load_knn_weights,
    matured_pool,
    retrieve_fixed_knn,
    summarize_predictions,
)

logger = logging.getLogger(__name__)

LEGACY_CURRENT_CONTEXT_SYSTEM_PROMPT = """
You are a crypto trading analyst.
Decide one signal: BUY, SELL, or HOLD.
Use only the current-day market context and current-day news context provided below.
Do not use historical analogies.

Rules:
- BUY when the current market context and news context support upward direction over the next 7 days.
- SELL when they support downward direction over the next 7 days.
- HOLD when the evidence is mixed, weak, or near-neutral.
- Keep confidence conservative when the evidence conflicts.

You MUST respond with a JSON object:
{
  "reasoning_steps": [
    "step 1: summarize market context",
    "step 2: summarize news context",
    "step 3: resolve conflicts and choose signal"
  ],
  "signal": "BUY",
  "confidence": 0.64,
  "explanation": "2 concise sentences"
}
"""

COMPACT_CURRENT_CONTEXT_SYSTEM_PROMPT = """
You are a crypto trading analyst.
Use only the provided current-day market and news context.
Predict the next-7-day direction as BUY, SELL, or HOLD.
Return JSON only: {"signal":"BUY","confidence":0.64}
Confidence must be between 0 and 1.
No explanation, no reasoning, no markdown, no extra text.
"""

LEGACY_MAX_HEADLINE_CHARS = 400
COMPACT_MAX_HEADLINE_CHARS = 160
_PREFERRED_INDICATOR_KEYS = (
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "price_change_pct",
    "msi",
)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _pct_change(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / old * 100.0


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _scalar_indicator_pairs(indicators: dict[str, Any]) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for key in _PREFERRED_INDICATOR_KEYS:
        value = indicators.get(key)
        if isinstance(value, (int, float)):
            pairs.append((key, float(value)))
    if pairs:
        return pairs
    for key, value in indicators.items():
        if isinstance(value, (int, float)):
            pairs.append((str(key), float(value)))
        if len(pairs) >= 6:
            break
    return pairs


def render_current_context(record: Any, *, prompt_style: str = "compact") -> str:
    snapshot = record.market_snapshot
    ohlcv = getattr(snapshot, "ohlcv", None)
    recent = list(getattr(snapshot, "recent_candles", None) or [])
    close_now = float(_field(ohlcv, "close", 0.0)) if ohlcv is not None else None
    close_1d = float(_field(recent[-2], "close", 0.0)) if len(recent) >= 2 else None
    close_3d = float(_field(recent[-4], "close", 0.0)) if len(recent) >= 4 else None
    ret_1d = _pct_change(close_1d, close_now)
    ret_3d = _pct_change(close_3d, close_now)
    indicators = getattr(snapshot, "indicators", None) or {}

    if prompt_style == "legacy":
        lines = [
            f"Symbol: {record.symbol}",
            f"Date: {record.date}",
        ]
        if ohlcv is not None:
            lines.append(
                f"Latest Candle: open={float(_field(ohlcv, 'open', 0.0)):.4f} "
                f"high={float(_field(ohlcv, 'high', 0.0)):.4f} "
                f"low={float(_field(ohlcv, 'low', 0.0)):.4f} "
                f"close={float(_field(ohlcv, 'close', 0.0)):.4f} "
                f"volume={float(_field(ohlcv, 'volume', 0.0)):.2f}"
            )
        if ret_1d is not None:
            lines.append(f"Price Change 1d: {ret_1d:+.2f}%")
        if ret_3d is not None:
            lines.append(f"Price Change 3d: {ret_3d:+.2f}%")
        if indicators:
            lines.append(
                "Indicators: "
                + "  ".join(
                    f"{key}={float(value):.4f}" if isinstance(value, (int, float)) else f"{key}={value}"
                    for key, value in indicators.items()
                )
            )
        lines.append(
            f"News Sentiment (1d aggregate): {record.sentiment_label} "
            f"(score={record.sentiment_score:+.2f})"
        )
        if record.summary:
            lines.append(f"News Titles: {_clip(str(record.summary), LEGACY_MAX_HEADLINE_CHARS)}")
        return "\n".join(lines)

    scalar_indicators = _scalar_indicator_pairs(indicators)

    lines = [
        f"Symbol: {record.symbol}",
        f"Date: {record.date}",
    ]
    if ohlcv is not None:
        lines.append(f"Close: {float(_field(ohlcv, 'close', 0.0)):.4f}")
    if ret_1d is not None:
        lines.append(f"Change 1d: {ret_1d:+.2f}%")
    if ret_3d is not None:
        lines.append(f"Change 3d: {ret_3d:+.2f}%")
    if scalar_indicators:
        lines.append(
            "Indicators: "
            + " ".join(f"{key}={value:.4f}" for key, value in scalar_indicators)
        )
    lines.append(
        f"Sentiment 1d: {record.sentiment_label} "
        f"(score={record.sentiment_score:+.2f})"
    )
    if record.summary:
        lines.append(f"Titles: {_clip(str(record.summary), COMPACT_MAX_HEADLINE_CHARS)}")
    return "\n".join(lines)


def _parse_signal(value: object) -> str:
    text = str(value or "HOLD").upper().strip()
    return text if text in {"BUY", "SELL", "HOLD"} else "HOLD"


def _parse_confidence(value: object) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = 0.0
    return max(0.0, min(1.0, conf))


def _clean_prediction_text(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = _clean_prediction_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start != -1:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
        raise


def _extract_prediction_fields(raw: str) -> dict[str, Any]:
    text = _clean_prediction_text(raw)

    signal_match = re.search(r"\b(BUY|SELL|HOLD)\b", text, flags=re.IGNORECASE)
    conf_match = re.search(
        r"confidence[^0-9\-]*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    explanation_match = re.search(
        r"explanation[^A-Za-z0-9\"]*\"?([^\n\"]+)\"?",
        text,
        flags=re.IGNORECASE,
    )
    if not signal_match:
        raise ValueError(f"Could not extract signal from model output: {text[:400]}")

    return {
        "signal": signal_match.group(1).upper(),
        "confidence": float(conf_match.group(1)) if conf_match else 0.5,
        "explanation": explanation_match.group(1).strip() if explanation_match else text[:300],
        "reasoning_steps": [],
    }


def _parse_prediction_payload(raw: str) -> dict[str, Any]:
    try:
        payload = _extract_json_object(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return _extract_prediction_fields(raw)


async def predict_current_context(
    llm: LLMClient,
    record: Any,
    *,
    prompt_style: str = "compact",
    max_attempts: int = 4,
    retry_delay_seconds: float = 8.0,
) -> dict[str, Any]:
    prompt = render_current_context(record, prompt_style=prompt_style)
    last_error: Exception | None = None
    last_raw: str = ""

    for attempt in range(1, max_attempts + 1):
        system_prompt = (
            LEGACY_CURRENT_CONTEXT_SYSTEM_PROMPT
            if prompt_style == "legacy"
            else COMPACT_CURRENT_CONTEXT_SYSTEM_PROMPT
        )
        if attempt > 1:
            if prompt_style == "legacy":
                system_prompt += (
                    "\n\nReturn only one valid JSON object with double-quoted keys and values. "
                    "Do not include markdown, comments, or any text before/after the JSON."
                )
            else:
                system_prompt += (
                    "\n\nReturn exactly one JSON object with keys signal and confidence only. "
                    "Do not include markdown, comments, reasoning, or any text before or after the JSON."
                )
        raw = await llm.generate(prompt=prompt, system=system_prompt)
        last_raw = raw
        try:
            payload = _parse_prediction_payload(raw)
            return {
                "predicted_signal": _parse_signal(payload.get("signal")),
                "confidence": _parse_confidence(payload.get("confidence")),
                "explanation": str(payload.get("explanation", "")),
                "reasoning_steps": list(payload.get("reasoning_steps", [])),
                "prompt_chars": len(prompt),
                "raw_response": _clean_prediction_text(raw),
                "attempts_used": attempt,
            }
        except Exception as exc:
            last_error = exc
            logger.warning(
                "[naive_current_ai] parse failure on %s attempt %d/%d: %s | raw=%s",
                record.date,
                attempt,
                max_attempts,
                exc,
                _clean_prediction_text(raw)[:300],
            )
            if attempt < max_attempts:
                await asyncio.sleep(retry_delay_seconds * attempt)

    raise RuntimeError(
        f"Could not recover a valid prediction after {max_attempts} attempts. "
        f"Last error: {last_error}. Last raw: {_clean_prediction_text(last_raw)[:500]}"
    )


def write_summary_markdown(
    metrics: list[PredictionMetrics],
    out_path: Path,
    *,
    data_path: Path,
    k: int,
    label_threshold: float,
) -> None:
    lines = [
        "# Current-Context AI vs Structured Baselines",
        "",
        f"- Data source: `{data_path}`",
        f"- Test split: `2025-07-01` to `2026-05-01`",
        f"- Label threshold: `±{label_threshold:.2f}%` on `future_return_7d`",
        f"- Fixed-kNN retrieval depth: `k={k}`",
        "",
        "| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate | Avg conf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        lines.append(
            "| "
            f"{item.name} | {item.n} | {item.overall_acc:.4f} | {item.active_acc:.4f} | "
            f"{item.coverage:.4f} | {item.hit_at_5_same_sign:.4f} | {item.buy_rate:.4f} | "
            f"{item.hold_rate:.4f} | {item.sell_rate:.4f} | {item.avg_confidence:.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_current_context_evaluation(
    *,
    llm: LLMClient,
    data_path: Path,
    out_dir: Path,
    weights_path: Path,
    fixed_head_path: Path,
    k: int = 5,
    label_threshold: float = 2.0,
    max_queries: int | None = None,
    eval_date_from: str | None = None,
    eval_date_to: str | None = None,
    progress_every: int = 10,
    resume: bool = True,
    prediction_max_attempts: int = 4,
    prediction_retry_delay_seconds: float = 8.0,
    throttle_seconds: float = 1.5,
    prompt_style: str = "compact",
    replace_existing_dates: bool = False,
) -> dict[str, Any]:
    rows = load_historical_rows(data_path)
    all_test_rows = [row for row in rows if row.split == "test"]
    process_rows = list(all_test_rows)
    if eval_date_from is not None:
        process_rows = [row for row in process_rows if row.record.date.isoformat() >= eval_date_from]
    if eval_date_to is not None:
        process_rows = [row for row in process_rows if row.record.date.isoformat() <= eval_date_to]
    if max_queries is not None:
        process_rows = process_rows[:max_queries]

    out_dir.mkdir(parents=True, exist_ok=True)
    weights = load_knn_weights(weights_path)
    fixed_head: HeadConfig = load_head_config(fixed_head_path)
    effective_k = fixed_head.k
    naive_path = out_dir / "naive_current_ai_test.jsonl"

    existing: list[dict[str, Any]] = []
    existing_dates: set[str] = set()
    replace_dates = {row.record.date.isoformat() for row in process_rows} if replace_existing_dates else set()
    if not resume and naive_path.exists():
        naive_path.unlink()
    if resume and naive_path.exists():
        with naive_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row_date = str(row.get("date"))
                if row_date in replace_dates:
                    continue
                existing.append(row)
                existing_dates.add(row_date)
        if replace_existing_dates:
            with naive_path.open("w", encoding="utf-8") as handle:
                for row in existing:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    naive_rows = list(existing)
    open_mode = "a" if resume else "w"
    with naive_path.open(open_mode, encoding="utf-8") as handle:
        for index, query in enumerate(process_rows, start=1):
            query_date = query.record.date.isoformat()
            pool = matured_pool(rows, query)
            similar = retrieve_fixed_knn(query, pool, weights=weights, k=effective_k)
            top5_same = any(
                actual_signal(item.record.future_return_7d, label_threshold)
                == actual_signal(query.record.future_return_7d, label_threshold)
                for item in similar[:5]
            )
            if query_date in existing_dates:
                continue
            model_out = await predict_current_context(
                llm,
                query.record,
                prompt_style=prompt_style,
                max_attempts=prediction_max_attempts,
                retry_delay_seconds=prediction_retry_delay_seconds,
            )
            row = {
                "date": query_date,
                "model": "naive_current_ai",
                "predicted_signal": model_out["predicted_signal"],
                "actual_signal": actual_signal(query.record.future_return_7d, label_threshold),
                "actual_return_7d": query.record.future_return_7d,
                "confidence": model_out["confidence"],
                "top5_same_sign": top5_same,
                "retrieval_count_reference": len(similar),
                "prompt_chars": model_out["prompt_chars"],
                "explanation": model_out["explanation"],
                "reasoning_steps": model_out["reasoning_steps"],
                "raw_response": model_out.get("raw_response"),
                "attempts_used": model_out.get("attempts_used"),
                "prompt_style": prompt_style,
            }
            naive_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if throttle_seconds > 0:
                await asyncio.sleep(throttle_seconds)
            if progress_every > 0 and index % progress_every == 0:
                partial = summarize_predictions(
                    "naive_current_ai",
                    naive_rows,
                    label_threshold=label_threshold,
                )
                logger.info(
                    "[naive_current_ai] %d/%d overall_acc=%.4f active_acc=%.4f coverage=%.4f",
                    index,
                    len(process_rows),
                    partial.overall_acc,
                    partial.active_acc,
                    partial.coverage,
                )

    fixed_rows: list[dict[str, Any]] = []
    knn_returns_rows: list[dict[str, Any]] = []
    for query in all_test_rows:
        pool = matured_pool(rows, query)
        similar = retrieve_fixed_knn(query, pool, weights=weights, k=effective_k)
        top5_same = any(
            actual_signal(item.record.future_return_7d, label_threshold)
            == actual_signal(query.record.future_return_7d, label_threshold)
            for item in similar[:5]
        )
        fixed_signal, fixed_conf = configured_head_signal(similar, head=fixed_head)
        knn_signal, knn_conf = knn_returns_signal(similar, threshold=label_threshold)
        base = {
            "date": query.record.date.isoformat(),
            "actual_signal": actual_signal(query.record.future_return_7d, label_threshold),
            "actual_return_7d": query.record.future_return_7d,
            "top5_same_sign": top5_same,
            "retrieval_count_reference": len(similar),
        }
        fixed_rows.append({
            **base,
            "model": "fixed_knn_rolling_stable",
            "predicted_signal": fixed_signal,
            "confidence": fixed_conf,
        })
        knn_returns_rows.append({
            **base,
            "model": "knn_returns",
            "predicted_signal": knn_signal,
            "confidence": knn_conf,
        })

    for name, rows_out in (
        ("fixed_knn_test.jsonl", fixed_rows),
        ("knn_returns_test.jsonl", knn_returns_rows),
    ):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = [
        summarize_predictions("naive_current_ai", naive_rows, label_threshold=label_threshold),
        summarize_predictions("fixed_knn_rolling_stable", fixed_rows, label_threshold=label_threshold),
        summarize_predictions("knn_returns", knn_returns_rows, label_threshold=label_threshold),
    ]
    summary_json = {
        "data_path": str(data_path),
        "weights_path": str(weights_path),
        "fixed_head_path": str(fixed_head_path),
        "k": effective_k,
        "label_threshold": label_threshold,
        "models": [asdict(item) for item in metrics],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(
        metrics,
        out_dir / "summary.md",
        data_path=data_path,
        k=effective_k,
        label_threshold=label_threshold,
    )
    return {
        "metrics": metrics,
        "summary_path": out_dir / "summary.json",
        "markdown_path": out_dir / "summary.md",
    }
