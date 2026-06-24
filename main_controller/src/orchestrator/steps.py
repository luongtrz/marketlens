"""Individual pipeline step functions.

Data-flow summary:
  Step 1 COLLECT   (parallel)  : crawler.get_latest [REQUIRED] + market.get_snapshot [REQUIRED]
  Step 2 AI SCORE  (sequential): aggregate pre-computed sentiment from articles;
                                  factors from Supabase daily_factor_snapshots (or AIHub fallback);
                                  push to factorledge.update_ledger + classify_vector [graceful]
  Step 3 STOCKMEM  (sequential, REQUIRED): stockmem.save → stockmem.search(k=5)
  Step 4 PREDICT   (REQUIRED)  : aihub.predict(current, similar)

Note — Step 2 no longer calls aihub.sentiment separately.
  Each IngestionRecord already carries sentiment_score + sentiment_label produced
  by the crawler enrichment pipeline (CryptoBERT / FinBERT).  We average those
  pre-computed scores across all fetched articles.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

from main_controller.src.orchestrator.context import PipelineContext
from main_controller.src.orchestrator.exceptions import PipelineError
from shared.models.factor import Factor
from shared.models.market import MarketSnapshot
from shared.models.memory import StockMemRecord
from shared.models.prediction import PredictResponse


def _best_text(article) -> str:
    """Return the richest available text for an article.

    Priority: summary → raw_text (truncated) → headline slug.
    """
    summary = (getattr(article, "summary", None) or "").strip()
    if summary:
        return summary

    raw = (getattr(article, "raw_text", None) or "").strip()
    if raw:
        return raw[:800]  # keep token budget sane per-article

    name = (getattr(article, "article_name", "") or "").strip()
    if name.startswith("http"):
        from urllib.parse import urlparse
        path = urlparse(name).path.rstrip("/")
        slug = path.rsplit("/", 1)[-1] if path else ""
        return slug.replace("-", " ").replace("_", " ")
    return name

logger = logging.getLogger(__name__)


def _parse_supabase_factors(raw: object) -> list[Factor]:
    """Parse daily_factor_snapshots.factors_json, tolerant to stringified JSON."""
    items = raw
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except Exception:
            return []
    if not isinstance(items, list):
        return []

    out: list[Factor] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            out.append(Factor(
                name=str(item.get("name", "")),
                type=str(item.get("type", "macro")),
                polarity=float(item.get("polarity", 0.0)),
                confidence=float(item.get("weight", item.get("confidence", 0.8))),
            ))
        except (ValueError, TypeError):
            continue
    return out


def _parse_supabase_factor_vector(raw: object) -> list[float]:
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for v in raw:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return []
    return out if len(out) == 75 else []


class ModuleClients:
    """Container for all module client instances (injected via DI)."""

    def __init__(
        self,
        crawler=None,
        aihub=None,
        market=None,
        stockmem=None,
        factorledge=None,
        llm_gateway=None,
    ) -> None:
        self.crawler = crawler
        self.aihub = aihub
        self.market = market
        self.stockmem = stockmem
        self.factorledge = factorledge
        self.llm_gateway = llm_gateway


async def step_collect(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 1: Collect articles and market snapshot in parallel — both REQUIRED.

    In historical mode (ctx.as_of_date set), fetches only data available on that
    date — no look-ahead. Articles are filtered to that day; OHLCV is fetched via
    /history with end_time so future candles are excluded.
    """
    if ctx.as_of_date is not None:
        target = ctx.as_of_date
        # Cut-off = 1ms before midnight of target date so Binance excludes that day's candle
        cutoff = datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
        end_ts = int(cutoff.timestamp() * 1000) - 1

        articles_task = clients.crawler.get_latest(
            ctx.symbol,
            publish_gte=cutoff,
            publish_lte=cutoff + timedelta(days=1),
        )
        history_task = clients.market.get_history(
            ctx.symbol, interval="1d", limit=50, end_time=str(end_ts)
        )
        results = await asyncio.gather(articles_task, history_task, return_exceptions=True)

        if isinstance(results[0], Exception):
            raise PipelineError(f"Crawler failed: {results[0]}")
        ctx.latest_articles = results[0]

        if isinstance(results[1], Exception):
            raise PipelineError(f"MarketData history failed: {results[1]}")
        candles = results[1]
        # Pick the candle whose date matches target_date (last one ≤ target)
        match = next(
            (c for c in reversed(candles) if c.timestamp.date() <= target),
            candles[-1] if candles else None,
        )
        if match is None:
            raise PipelineError(f"No OHLCV candle found for {target}")

        try:
            indicators = await clients.market.get_indicators(candles)
        except Exception as exc:
            logger.warning("Failed to compute historical indicators: %s", exc)
            indicators = {}

        ctx.market_snapshot = MarketSnapshot(
            symbol=ctx.symbol,
            timestamp=match.timestamp,
            ohlcv=match,
            recent_candles=candles[-21:],
            indicators=indicators,
            source="binance",
        )
    else:
        results = await asyncio.gather(
            clients.crawler.get_latest(ctx.symbol),
            clients.market.get_snapshot(ctx.symbol),
            return_exceptions=True,
        )
        if isinstance(results[0], Exception):
            raise PipelineError(f"Crawler failed: {results[0]}")
        ctx.latest_articles = results[0]

        if isinstance(results[1], Exception):
            raise PipelineError(f"MarketData failed: {results[1]}")
        ctx.market_snapshot = results[1]


async def step_ai_score(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 2: Aggregate pre-computed sentiment; load factors from Supabase (or AIHub fallback).

    Sentiment — each IngestionRecord already carries sentiment_score (CryptoBERT/FinBERT)
    produced by the crawler enrichment pipeline.  We take the mean across all
    articles so that a single outlier does not dominate.

    Factors — prefer Supabase daily_factor_snapshots (pre-computed by external
    pipeline).  Falls back to AIHub LLM extraction only when Supabase has no
    snapshot for today.
    """
    articles = ctx.latest_articles or []

    # ── Sentiment: aggregate pre-computed scores ──────────────────────────────
    scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
    if scores:
        avg = sum(scores) / len(scores)
        ctx.sentiment_score = round(avg, 4)
        ctx.sentiment_label = (
            "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral"
        )
        logger.info(
            "[step_ai_score] sentiment aggregated from %d articles: %.4f (%s)",
            len(scores), avg, ctx.sentiment_label,
        )
    else:
        ctx.sentiment_score = 0.0
        ctx.sentiment_label = "neutral"
        logger.warning("[step_ai_score] no pre-computed sentiment found, defaulting to 0.0")

    # ── Factors: prefer Supabase daily_factor_snapshots ─────────────────────
    raw_factors: list[Factor] = []
    as_of = ctx.as_of_date or date.today()

    try:
        from shared.supabase_service import SupabaseReadService  # noqa: E402
        snap_svc = SupabaseReadService.from_env(default_table="daily_factor_snapshots")
        if snap_svc is not None:
            snap_rows = await snap_svc.select_rows(
                order="snapshot_date.desc",
                limit=1,
                columns="snapshot_date,factors_json,factor_vector",
                extra_duplicate_params=[
                    ("snapshot_date", f"eq.{as_of.isoformat()}"),
                    ("symbol", f"eq.{ctx.symbol.upper()}"),
                ],
            )
            if snap_rows:
                raw_factors = _parse_supabase_factors(snap_rows[0].get("factors_json"))
                snap_factor_vector = _parse_supabase_factor_vector(snap_rows[0].get("factor_vector"))
                if snap_factor_vector:
                    ctx.factor_vector = snap_factor_vector
                logger.info(
                    "[step_ai_score] loaded %d factors from Supabase daily_factor_snapshots",
                    len(raw_factors),
                )
    except Exception as exc:
        logger.warning("[step_ai_score] Supabase factor fetch failed: %s", exc)

    # ── Fallback to AIHub LLM extraction ───────────────────────────────────
    if not raw_factors:
        combined_text = "\n\n".join(
            t for a in articles if (t := _best_text(a))
        )[:6000]

        factors_result = await asyncio.gather(
            clients.aihub.factors(combined_text, ticker=ctx.symbol),
            return_exceptions=True,
        )
        factors_result = factors_result[0]

        if isinstance(factors_result, Exception):
            ctx.errors.append(f"AIHub factors failed: {factors_result}")
            logger.warning("Factors degraded to []: %s", factors_result)
        else:
            raw_factors = factors_result
            logger.info("[step_ai_score] extracted %d factors from AIHub", len(raw_factors))

    # ── Factor Ledge: push factors → build/update rolling ledger ─────────────
    ctx.raw_factors = raw_factors
    try:
        ctx.factors = await clients.factorledge.update_ledger(
            records=[
                {
                    "date": str(date.today()),
                    "factors": [f.name for f in raw_factors],
                }
            ],
            window_days=7,
        )
    except Exception as exc:
        ctx.errors.append(f"FactorLedge failed: {exc}")
        logger.warning("FactorLedge unavailable, factors=[]: %s", exc)
        ctx.factors = []

    # ── Factor Vector: get 75d binary vector from classify-service ──────────
    try:
        factor_names = [f.name for f in raw_factors]
        if not ctx.factor_vector and factor_names:
            ctx.factor_vector = await clients.factorledge.classify_vector(factor_names)
            active_count = sum(1 for v in ctx.factor_vector if v == 1.0)
            logger.info(
                "[step_ai_score] factor vector: %d/75 bits active", active_count,
            )
        elif not ctx.factor_vector:
            ctx.factor_vector = []
    except Exception as exc:
        ctx.errors.append(f"FactorLedge classify_vector failed: {exc}")
        logger.warning("FactorLedge classify_vector unavailable: %s", exc)
        ctx.factor_vector = []


def _headline(article) -> str:
    """Return the article name/title, truncating URLs."""
    name = (getattr(article, "article_name", "") or "").strip()
    if name.startswith("http"):
        path = urlparse(name).path.rstrip("/")
        slug = path.rsplit("/", 1)[-1] if path else ""
        return slug.replace("-", " ").replace("_", " ")
    return name


def _short_headlines_for_summary(articles: list) -> str:
    """Avoid multi-kB URL titles in StockMem summaries (bloats Groq input)."""
    parts: list[str] = []
    for a in articles[:3]:
        h = _headline(a)
        if not h:
            continue
        if len(h) > 220:
            h = h[:219].rstrip() + "…"
        parts.append(h)
    return " ".join(parts)


async def step_stockmem(
    ctx: PipelineContext, clients: ModuleClients, k_similar: int = 3
) -> None:
    """STEP 3: Save current record to StockMem, retrieve k nearest past records."""
    if ctx.market_snapshot is None:
        raise PipelineError("market_snapshot is None — cannot build StockMem record")

    current_record = StockMemRecord(
        date=ctx.as_of_date or date.today(),
        symbol=ctx.symbol,
        sentiment_score=ctx.sentiment_score or 0.0,
        sentiment_label=ctx.sentiment_label or "neutral",
        factors=[f.name for f in ctx.raw_factors],
        normalized_factors=ctx.factors or ctx.raw_factors,
        market_snapshot=ctx.market_snapshot,
        factor_vector=ctx.factor_vector,
        summary=_short_headlines_for_summary(ctx.latest_articles),
        article_ids=[a.id for a in ctx.latest_articles],
        article_sources=[a.source for a in ctx.latest_articles],
        article_published_at=[a.date_published for a in ctx.latest_articles],
        run_id=str(ctx.run_id),
    )
    ctx.current_record = current_record

    try:
        ctx.current_record_id = await clients.stockmem.save(current_record)
        results = await clients.stockmem.search(
            query=current_record,
            k=k_similar + 1,
            before_date=ctx.as_of_date,  # Exclude records on/after backtest date
        )
        # Exclude the record we just saved so it doesn't appear as its own similar case
        ctx.similar_records = [
            r for r in results if r.record.id != ctx.current_record_id
        ][:k_similar]
    except Exception as exc:
        raise PipelineError(f"StockMem failed: {exc}") from exc


_DEFAULT_KNN_WEIGHTS: dict[str, float] = {
    "1d": 0.15, "3d": 0.25, "7d": 0.35, "15d": 0.15, "30d": 0.10
}


def _knn_returns_signal(
    similar_records: list,
    weights: dict[str, float],
    buy_threshold: float,
    sell_threshold: float,
) -> "PredictResponse":
    """Derive signal from weighted average future returns of k similar records.

    Weights are normalized per-record so missing horizons don't bias the result.
    """
    from shared.models.prediction import PredictResponse, SignalType

    per_record_avgs: list[float] = []
    for sr in similar_records:
        rec = sr.record
        total_w = total_v = 0.0
        for h, w in weights.items():
            val = getattr(rec, f"future_return_{h}", None)
            if val is not None:
                total_v += val * w
                total_w += w
        if total_w > 0:
            per_record_avgs.append(total_v / total_w)

    if not per_record_avgs:
        return PredictResponse(
            signal=SignalType.HOLD,
            confidence=0.50,
            explanation="kNN-returns: no future return data in similar records; defaulting to HOLD.",
            reasoning_steps=["knn_returns: no data available"],
        )

    overall_avg = sum(per_record_avgs) / len(per_record_avgs)

    if overall_avg > buy_threshold:
        signal = SignalType.BUY
        distance = overall_avg - buy_threshold
    elif overall_avg < sell_threshold:
        signal = SignalType.SELL
        distance = sell_threshold - overall_avg  # positive
    else:
        signal = SignalType.HOLD
        distance = 0.0

    # Consensus: fraction of individual records that agree with signal
    def _rec_signal(v: float) -> str:
        if v > buy_threshold:
            return "BUY"
        if v < sell_threshold:
            return "SELL"
        return "HOLD"

    consensus = sum(1 for v in per_record_avgs if _rec_signal(v) == signal.value)
    consensus_bonus = round((consensus / len(per_record_avgs) - 0.5) * 0.10, 3)

    confidence = round(min(max(0.55 + min(distance / 15.0, 0.35) + consensus_bonus, 0.50), 0.95), 3)

    w_str = " | ".join(f"{h}={w:.0%}" for h, w in weights.items())
    explanation = (
        f"kNN-returns: {len(per_record_avgs)} similar days → "
        f"weighted avg = {overall_avg:+.2f}% "
        f"(weights: {w_str}). "
        f"Consensus {consensus}/{len(per_record_avgs)} → {signal.value}."
    )
    reasoning_steps = [
        f"Similar records with return data: {len(per_record_avgs)}/{len(similar_records)}",
        f"Per-record averages: {[f'{v:+.2f}%' for v in per_record_avgs]}",
        f"Overall weighted avg: {overall_avg:+.2f}%",
        f"Thresholds: BUY > {buy_threshold}%, SELL < {sell_threshold}%",
        f"Signal: {signal.value}  |  distance from threshold: {distance:.2f}%",
        f"Consensus: {consensus}/{len(per_record_avgs)}  |  confidence: {confidence:.3f}",
    ]

    return PredictResponse(
        signal=signal,
        confidence=confidence,
        explanation=explanation,
        reasoning_steps=reasoning_steps,
    )


def _cem_rag_signal(
    similar_records: list,
    tau: float = 0.22,
    horizon: str = "7d",
) -> tuple["PredictResponse", "CEMRAGPrediction"]:
    """CEM-RAG: derive p_up/p_down/p_hold from kNN similarity × future return direction.

    Similarity from SimilarRecord is in [-1, 1]; shifted to [0, 1] as weight so even
    low-similarity neighbours contribute a small positive mass.
    Signal rule: BUY if p_up - p_down >= tau, SELL if p_down - p_up >= tau, else HOLD.
    tau=0.22 is the policy_v4 value calibrated on val Sharpe with coverage gate.
    """
    from shared.models.prediction import CEMRAGPrediction, SignalType

    ret_attr = f"future_return_{horizon}"
    usable = [
        sr for sr in similar_records
        if getattr(sr.record, ret_attr, None) is not None
    ]

    if not usable:
        cem = CEMRAGPrediction(
            horizon=horizon, p_up=1 / 3, p_down=1 / 3, p_hold=1 / 3,
            signal="HOLD", confidence=0.50, tau=tau,
            explanation="cem_rag: no similar records with future return data — defaulting to HOLD.",
            retrieval_count=0,
        )
        return (
            PredictResponse(
                signal=SignalType.HOLD,
                confidence=0.50,
                explanation=cem.explanation,
                reasoning_steps=["cem_rag: 0 usable records"],
            ),
            cem,
        )

    # Shift similarity [-1,1] → [0,1] so weight is always positive
    weights = [(sr.similarity + 1) / 2 for sr in usable]
    total_w = sum(weights) + 1e-12

    p_up   = sum(w for w, sr in zip(weights, usable) if getattr(sr.record, ret_attr) > 0) / total_w
    p_down = sum(w for w, sr in zip(weights, usable) if getattr(sr.record, ret_attr) < 0) / total_w
    p_hold = max(0.0, 1.0 - p_up - p_down)

    diff_up   = p_up - p_down
    diff_down = p_down - p_up

    if diff_up >= tau:
        signal     = SignalType.BUY
        confidence = round(min(0.50 + diff_up * 0.80, 0.95), 3)
    elif diff_down >= tau:
        signal     = SignalType.SELL
        confidence = round(min(0.50 + diff_down * 0.80, 0.95), 3)
    else:
        signal     = SignalType.HOLD
        confidence = round(0.50 + max(diff_up, diff_down) * 0.30, 3)

    ret_vals = [f"{getattr(sr.record, ret_attr):+.2f}%" for sr in usable]
    explanation = (
        f"cem_rag ({horizon}): {len(usable)} similar records → "
        f"p_up={p_up:.3f} p_down={p_down:.3f} p_hold={p_hold:.3f}  "
        f"τ={tau} → {signal.value}. "
        f"Retrieved {horizon} returns: {', '.join(ret_vals)}."
    )

    cem = CEMRAGPrediction(
        horizon=horizon,
        p_up=round(p_up, 4),
        p_down=round(p_down, 4),
        p_hold=round(p_hold, 4),
        signal=signal.value,
        confidence=confidence,
        tau=tau,
        explanation=explanation,
        retrieval_count=len(usable),
    )

    return (
        PredictResponse(
            signal=signal,
            confidence=confidence,
            explanation=explanation,
            reasoning_steps=[
                f"horizon: {horizon}  tau: {tau}",
                f"usable similar records: {len(usable)}/{len(similar_records)}",
                f"p_up={p_up:.4f}  p_down={p_down:.4f}  p_hold={p_hold:.4f}",
                f"diff_up={diff_up:.4f}  diff_down={diff_down:.4f}",
                f"signal={signal.value}  confidence={confidence:.3f}",
                f"retrieved {horizon} returns: {ret_vals}",
            ],
        ),
        cem,
    )


async def step_predict(
    ctx: PipelineContext,
    clients: ModuleClients,
    predict_provider: str = "knn_returns",
    llm_model: str | None = None,
    knn_weights: dict[str, float] | None = None,
    knn_buy_threshold: float = 3.0,
    knn_sell_threshold: float = -3.0,
    cem_tau: float = 0.22,
    cem_horizon: str = "7d",
) -> None:
    """STEP 4: Predict signal using selected provider."""
    if ctx.current_record is None or ctx.market_snapshot is None:
        raise PipelineError("current_record or market_snapshot is None — cannot predict")

    try:
        provider = (predict_provider or "knn_returns").strip().lower()
        if provider == "cem_rag":
            ctx.prediction, ctx.cem_rag_prediction = _cem_rag_signal(
                similar_records=ctx.similar_records,
                tau=cem_tau,
                horizon=cem_horizon,
            )
        elif provider == "knn_returns":
            ctx.prediction = _knn_returns_signal(
                similar_records=ctx.similar_records,
                weights=knn_weights or _DEFAULT_KNN_WEIGHTS,
                buy_threshold=knn_buy_threshold,
                sell_threshold=knn_sell_threshold,
            )
        elif provider == "llm_gateway":
            if clients.llm_gateway is None:
                raise PipelineError("LLM Gateway client is not configured")
            ctx.prediction = await clients.llm_gateway.predict(
                current=ctx.current_record,
                similar=ctx.similar_records,
                model=llm_model,
            )
        else:
            ctx.prediction = await clients.aihub.predict(
                current=ctx.current_record,
                similar=ctx.similar_records,
            )
    except Exception as exc:
        raise PipelineError(f"predict failed ({predict_provider}): {exc}") from exc
