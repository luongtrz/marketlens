"""MainController FastAPI application — /run, /status, /result, /backfill endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Any, AsyncIterator, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main_controller.src.clients.aihub_client import AIHubClient
from main_controller.src.clients.crawler_client import CrawlerClient
from main_controller.src.clients.factorledge_client import FactorLedgeClient
from main_controller.src.clients.llm_gateway_client import LLMGatewayClient
from main_controller.src.clients.market_client import MarketClient
from main_controller.src.clients.stockmem_client import StockMemClient
from main_controller.src.config import MainControllerConfig
from main_controller.src.auth.repository import AuthRepository
from main_controller.src.auth.routes import router as auth_router
from main_controller.src.auth.service import AuthService
from main_controller.src.orchestrator.pipeline import Pipeline, PipelineConfig
from main_controller.src.orchestrator.steps import ModuleClients
from main_controller.src.ui_routes import router as ui_router
from shared.models.factor import Factor, NormalizedFactor
from shared.models.market import MarketSnapshot
from shared.models.memory import StockMemRecord
from shared.models.prediction import PredictionResult
from shared.supabase_news import fetch_news_articles_from_supabase

logger = logging.getLogger(__name__)


class RunState(BaseModel):
    run_id: str
    symbol: str
    status: Literal["pending", "running", "done", "failed"] = "pending"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    result: PredictionResult | None = None
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = MainControllerConfig()
    auth_repo = AuthRepository(config.db_url)
    await auth_repo.init()
    app.state.auth = AuthService(
        auth_repo,
        jwt_secret=config.jwt_secret,
        jwt_algorithm=config.jwt_algorithm,
        access_ttl_minutes=config.jwt_access_ttl_minutes,
        refresh_ttl_days=config.jwt_refresh_ttl_days,
        issuer=config.jwt_issuer,
    )
    clients = ModuleClients(
        crawler=CrawlerClient(config.crawler_url),
        aihub=AIHubClient(config.aihub_url),
        llm_gateway=LLMGatewayClient(
            config.llm_gateway_url,
            min_directional_confidence=config.llm_min_directional_confidence,
            hold_release_bias=config.llm_hold_release_bias,
            knn_confirm_threshold=config.llm_knn_confirm_threshold,
        ),
        market=MarketClient(config.market_data_url),
        stockmem=StockMemClient(config.stockmem_url),
        factorledge=FactorLedgeClient(config.factorledge_url),
    )
    app.state.pipeline = Pipeline(
        clients,
        PipelineConfig(
            k_similar=config.k_similar,
            predict_provider=config.predict_provider,
            knn_weights={
                "1d": config.knn_return_w1d,
                "3d": config.knn_return_w3d,
                "7d": config.knn_return_w7d,
                "15d": config.knn_return_w15d,
                "30d": config.knn_return_w30d,
            },
            knn_buy_threshold=config.knn_buy_threshold,
            knn_sell_threshold=config.knn_sell_threshold,
        ),
    )
    app.state.clients = clients
    app.state.run_states: dict[str, RunState] = {}
    app.state.background_tasks: set[asyncio.Task] = set()

    cron_task = asyncio.create_task(
        _daily_cron(
            symbols=[s.strip() for s in config.cron_symbols.split(",") if s.strip()],
            cron_hour=config.cron_hour,
            cron_minute=config.cron_minute,
            pipeline=app.state.pipeline,
            run_states=app.state.run_states,
        )
    )
    app.state.background_tasks.add(cron_task)
    cron_task.add_done_callback(app.state.background_tasks.discard)

    yield

    cron_task.cancel()
    if app.state.background_tasks:
        await asyncio.gather(*app.state.background_tasks, return_exceptions=True)
    await auth_repo.aclose()


app = FastAPI(title="MainController", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ui_router)
app.include_router(auth_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
async def run(
    symbol: str,
    trigger: str = "manual",
    as_of_date: date | None = Query(default=None, alias="date", description="Historical date (YYYY-MM-DD). Omit for live mode."),
    llm_model: str | None = Query(default=None, alias="model", description="Optional LLM model override for llm_gateway provider."),
) -> dict:
    run_id = str(uuid4())
    state = RunState(run_id=run_id, symbol=symbol, started_at=datetime.now(timezone.utc))
    app.state.run_states[run_id] = state

    task = asyncio.create_task(
        _execute_run(
            run_id,
            symbol,
            app.state.pipeline,
            app.state.run_states,
            as_of_date=as_of_date,
            llm_model=llm_model,
        )
    )
    app.state.background_tasks.add(task)
    task.add_done_callback(app.state.background_tasks.discard)

    return {
        "run_id": run_id,
        "status": "pending",
        "as_of_date": str(as_of_date) if as_of_date else None,
        "model": llm_model,
    }


@app.get("/status/{run_id}")
async def status(run_id: str) -> dict:
    state = app.state.run_states.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": state.run_id,
        "symbol": state.symbol,
        "status": state.status,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "has_result": state.result is not None,
    }


@app.get("/result/{run_id}", response_model=PredictionResult)
async def result(run_id: str) -> PredictionResult:
    state = app.state.run_states.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    if state.status != "done":
        raise HTTPException(status_code=425, detail=f"run status is '{state.status}'")
    return state.result  # type: ignore[return-value]


async def _execute_run(
    run_id: str,
    symbol: str,
    pipeline: Pipeline,
    run_states: dict[str, RunState],
    as_of_date: date | None = None,
    llm_model: str | None = None,
) -> None:
    state = run_states[run_id]
    state.status = "running"
    try:
        result = await pipeline.run(
            symbol,
            run_id=UUID(run_id),
            as_of_date=as_of_date,
            llm_model=llm_model,
        )
        state.result = result
        state.status = "done"
    except Exception as exc:
        logger.exception("Pipeline run %s failed: %s", run_id, exc)
        state.error = str(exc)
        state.status = "failed"
    finally:
        state.finished_at = datetime.now(timezone.utc)


async def _daily_cron(
    symbols: list[str],
    cron_hour: int,
    pipeline: Pipeline,
    run_states: dict[str, RunState],
    cron_minute: int = 0,
) -> None:
    """Wake up daily at cron_hour:cron_minute UTC and run the pipeline for each symbol."""
    logger.info("Daily cron started — symbols=%s time=%02d:%02d UTC", symbols, cron_hour, cron_minute)
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=cron_hour, minute=cron_minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        wait_secs = (next_run - now).total_seconds()
        logger.info("Cron: next run at %s (%.0fs)", next_run.isoformat(), wait_secs)
        try:
            await asyncio.sleep(wait_secs)
        except asyncio.CancelledError:
            break
        for symbol in symbols:
            run_id = str(uuid4())
            run_states[run_id] = RunState(
                run_id=run_id, symbol=symbol, started_at=datetime.now(timezone.utc)
            )
            logger.info("Cron: triggering pipeline run_id=%s symbol=%s", run_id, symbol)
            try:
                await _execute_run(run_id, symbol, pipeline, run_states)
            except Exception as exc:
                logger.exception("Cron run failed for %s: %s", symbol, exc)


def _rsi(candles: list[Any], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 50.0
    closes = [c.close for c in candles[-(period + 1):]]
    deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return round(100.0 - 100.0 / (1.0 + avg_gain / avg_loss), 2)


def _macd_hist(candles: list[Any]) -> float:
    """EMA-12 minus EMA-26 of closes (simplified single-bar MACD line, normalised by price)."""
    if len(candles) < 27:
        return 0.0
    closes = [c.close for c in candles]

    def _ema(data: list[float], p: int) -> float:
        k = 2.0 / (p + 1)
        v = data[0]
        for x in data[1:]:
            v = x * k + v * (1 - k)
        return v

    tail = closes[-26:]
    e12 = _ema(tail[-12:], 12)
    e26 = _ema(tail, 26)
    last_price = closes[-1] or 1.0
    return round((e12 - e26) / last_price * 100.0, 4)


@app.post("/backfill")
async def backfill(symbol: str, days: int = 30, offset: int = 0) -> dict:
    """Populate StockMem with historical records from Supabase + market history.

    Call repeatedly with increasing ``offset`` to page through history:
      POST /backfill?symbol=BTC&days=30&offset=0
      POST /backfill?symbol=BTC&days=30&offset=30
      POST /backfill?symbol=BTC&days=30&offset=60  ...
    """
    clients: ModuleClients = app.state.clients
    today = date.today()
    end_date = today - timedelta(days=offset)
    start_date = end_date - timedelta(days=days)

    # Fetch OHLCV with 20 extra days so each record can include recent_candles
    CANDLE_CONTEXT = 22  # 20 preceding candles + buffer
    ohlcv_by_date: dict[date, object] = {}
    all_candles: list = []
    try:
        end_ts = int((datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
                      + timedelta(days=1)).timestamp() * 1000)
        all_candles = await clients.market.get_history(
            symbol=symbol, interval="1d",
            limit=days + CANDLE_CONTEXT,
            end_time=str(end_ts),
        )
        for c in all_candles:
            ohlcv_by_date[c.timestamp.date()] = c
    except Exception as exc:
        logger.warning("backfill: market history failed: %s", exc)


    # Fetch all articles for this window in one call, group by date in Python
    day_start_all = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    day_end_all = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
    try:
        all_articles = await fetch_news_articles_from_supabase(
            limit=days * 50,
            symbol=symbol,
            publish_gte=day_start_all,
            publish_lte=day_end_all,
        )
    except Exception as exc:
        return {"error": f"Supabase fetch failed: {exc}"}

    # Group articles by date
    from collections import defaultdict
    articles_by_date: dict[date, list] = defaultdict(list)
    for a in all_articles:
        articles_by_date[a.date_published.date()].append(a)

    logger.info("backfill: %d articles across %d dates", len(all_articles), len(articles_by_date))

    from collections import defaultdict as _dd
    from urllib.parse import urlparse
    from shared.supabase_service import SupabaseReadService

    # Fetch pre-computed factors from daily_factor_snapshots (one call covers the whole window)
    factor_snapshot_by_date: dict[date, dict[str, Any]] = {}
    try:
        snap_svc = SupabaseReadService.from_env(default_table="daily_factor_snapshots")
        if snap_svc:
            snap_rows = await snap_svc.select_rows(
                order="snapshot_date.desc",
                limit=days + 5,
                columns="snapshot_date,factors_json,factor_vector",
                extra_duplicate_params=[
                    ("snapshot_date", f"gte.{start_date.isoformat()}"),
                    ("snapshot_date", f"lte.{end_date.isoformat()}"),
                    ("symbol", f"eq.{symbol.upper()}"),
                ],
            )
            for row in snap_rows:
                d = date.fromisoformat(str(row["snapshot_date"]))
                factor_snapshot_by_date[d] = {
                    "factors_json": row.get("factors_json"),
                    "factor_vector": row.get("factor_vector"),
                }
            logger.info("backfill: loaded factor snapshots for %d dates from Supabase", len(factor_snapshot_by_date))
    except Exception as exc:
        logger.warning("backfill: daily_factor_snapshots fetch failed: %s", exc)

    # Pre-compute per-date metadata (fast, no I/O)
    valid_dates = []
    skipped = 0
    for target_date, articles in sorted(articles_by_date.items()):
        ohlcv = ohlcv_by_date.get(target_date)
        if ohlcv is None:
            skipped += 1
            continue
        scored = [a.sentiment_score for a in articles if a.sentiment_score != 0.0]
        avg_score = sum(scored) / len(scored) if scored else 0.0
        label = "bullish" if avg_score > 0.15 else ("bearish" if avg_score < -0.15 else "neutral")
        preceding = [c for c in all_candles if c.timestamp.date() < target_date][-20:]
        all_prec  = [c for c in all_candles if c.timestamp.date() < target_date]
        rsi_val  = _rsi(preceding + [ohlcv])
        macd_val = _macd_hist(all_prec + [ohlcv])
        price_chg = round((ohlcv.close - preceding[-1].close) / preceding[-1].close * 100.0, 4) if preceding else 0.0
        msi_val  = round(max(0.0, min(100.0, 50.0 + (rsi_val - 50.0) * 0.6 + avg_score * 15.0)), 2)
        texts = [(a.summary or a.article_name or "")[:300] for a in articles[:8]]
        combined = " ".join(t for t in texts if t.strip())[:2000]
        valid_dates.append((target_date, articles, ohlcv, avg_score, label, preceding, rsi_val, macd_val, price_chg, msi_val, combined))

    # Build factor list per date: prefer Supabase snapshot, fall back to AIHub
    async def _fetch_factors_aihub(combined: str, td: date) -> list[Factor]:
        if not combined.strip():
            return []
        try:
            return await asyncio.wait_for(
                clients.aihub.factors(text=combined, ticker=symbol),
                timeout=25.0,
            )
        except Exception as exc:
            logger.warning("backfill: AIHub factors failed for %s: %s", td, exc)
            return []

    # Dates that have no Supabase snapshot need AIHub fallback
    dates_needing_aihub = [row for row in valid_dates if row[0] not in factor_snapshot_by_date]
    aihub_results = await asyncio.gather(
        *[_fetch_factors_aihub(row[10], row[0]) for row in dates_needing_aihub]
    )
    aihub_by_date: dict[date, list[Factor]] = {
        row[0]: factors for row, factors in zip(dates_needing_aihub, aihub_results)
    }

    def _parse_factors_json(raw: object) -> list[Factor]:
        data = raw
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return []
        if not isinstance(data, list):
            return []
        out: list[Factor] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                out.append(Factor(
                    name=item.get("name", ""),
                    type=item.get("type", "macro"),
                    polarity=float(item.get("polarity", 0.0)),
                    confidence=float(item.get("weight", item.get("confidence", 0.8))),
                ))
            except Exception:
                continue
        return out

    def _factor_list_for_date(td: date, first_article_id: str, day_start: datetime) -> list[Factor]:
        """Return Factor list from Supabase snapshot or AIHub fallback."""
        snap = factor_snapshot_by_date.get(td)
        if snap:
            parsed = _parse_factors_json(snap.get("factors_json"))
            if parsed:
                return parsed
        return aihub_by_date.get(td, [])

    factor_results = [_factor_list_for_date(row[0], str(row[1][0].id) if row[1] else "backfill",
                                             datetime(row[0].year, row[0].month, row[0].day, tzinfo=timezone.utc))
                      for row in valid_dates]

    # Build and save records
    saved = 0
    errors = []

    # ── Compute factor_vector for each date via FactorLedge classify-service ─
    factor_vector_by_date: dict[date, list[float]] = {}
    for td, flist in zip([row[0] for row in valid_dates], factor_results):
        snap = factor_snapshot_by_date.get(td) or {}
        snap_vec = snap.get("factor_vector")
        if isinstance(snap_vec, list) and len(snap_vec) == 75:
            try:
                factor_vector_by_date[td] = [float(v) for v in snap_vec]
                continue
            except Exception:
                pass
        if flist:
            try:
                names = [f.name for f in flist]
                fv = await clients.factorledge.classify_vector(names)
                factor_vector_by_date[td] = fv
                active = sum(1 for v in fv if v == 1.0)
                logger.info("backfill: %s factor_vector %d/75 bits active", td, active)
            except Exception as exc:
                logger.warning("backfill: classify_vector failed for %s: %s", td, exc)
                factor_vector_by_date[td] = []

    for row, factor_list in zip(valid_dates, factor_results):
        target_date, articles, ohlcv, avg_score, label, preceding, rsi_val, macd_val, price_chg, msi_val, _ = row
        day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        first_article_id = str(articles[0].id) if articles else "backfill"

        normalized_factors = [
            NormalizedFactor(
                name=f.name, type=f.type, weight=f.confidence, polarity=f.polarity,
                source_article_id=first_article_id, observed_at=day_start,
            )
            for f in factor_list
        ]

        summary_parts = []
        for a in articles[:3]:
            name = (a.article_name or "").strip()
            if name.startswith("http"):
                slug = urlparse(name).path.rstrip("/").rsplit("/", 1)[-1]
                name = slug.replace("-", " ")
            if name:
                summary_parts.append(name[:120])

        snapshot = MarketSnapshot(
            symbol=symbol, timestamp=day_start, ohlcv=ohlcv, recent_candles=preceding,
            indicators={"rsi": rsi_val, "macd_hist": macd_val, "price_change_pct": price_chg, "msi": msi_val},
            source="binance",
        )
        record = StockMemRecord(
            date=target_date, symbol=symbol,
            sentiment_score=avg_score, sentiment_label=label,
            factors=[f.name for f in factor_list], normalized_factors=normalized_factors,
            factor_vector=factor_vector_by_date.get(target_date, []),
            market_snapshot=snapshot, summary=" ".join(summary_parts),
            article_ids=[a.id for a in articles],
        )
        try:
            await clients.stockmem.save(record)
            saved += 1
        except Exception as exc:
            errors.append(f"{target_date}: stockmem save failed: {exc}")

    return {
        "symbol": symbol,
        "days": days,
        "offset": offset,
        "window": f"{start_date} → {end_date}",
        "articles_fetched": len(all_articles),
        "dates_with_articles": len(articles_by_date),
        "saved": saved,
        "skipped_no_ohlcv": skipped,
        "errors": errors[:10],
    }


@app.post("/fill-returns")
async def fill_returns(symbol: str) -> dict:
    """Backfill future_return_1d/3d/7d/15d/30d for StockMem records that are missing them.

    Fetches close price from Binance for each horizon relative to each record's
    date, then computes % change vs the record's own close price.
    Processes records where any future_return field IS NULL.
    """
    clients: ModuleClients = app.state.clients
    today = date.today()

    records = await clients.stockmem.list_missing_returns(symbol=symbol)
    updated = 0
    skipped = 0
    errors: list[str] = []

    for record in records:
        record_date = record.date
        base_close = record.market_snapshot.ohlcv.close if record.market_snapshot else None
        if base_close is None or base_close == 0:
            skipped += 1
            continue

        returns: dict[str, float | None] = {
            "r1d": None, "r3d": None, "r7d": None, "r15d": None, "r30d": None
        }

        for offset_days, attr in [(1, "r1d"), (3, "r3d"), (7, "r7d"), (15, "r15d"), (30, "r30d")]:
            target = record_date + timedelta(days=offset_days)
            if target > today:
                continue
            try:
                end_ts = int(
                    (datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
                     + timedelta(days=1)).timestamp() * 1000
                )
                candles = await clients.market.get_history(
                    symbol=symbol, interval="1d", limit=3, end_time=str(end_ts)
                )
                match = next(
                    (c for c in reversed(candles) if c.timestamp.date() <= target), None
                )
                if match:
                    returns[attr] = round((match.close - base_close) / base_close * 100.0, 4)
            except Exception as exc:
                errors.append(f"{record_date} D+{offset_days}: {exc}")

        if all(v is None for v in returns.values()):
            skipped += 1
            continue

        try:
            await clients.stockmem.update_future_returns(
                record.id,
                future_return_1d=returns["r1d"],
                future_return_3d=returns["r3d"],
                future_return_7d=returns["r7d"],
                future_return_15d=returns["r15d"],
                future_return_30d=returns["r30d"],
            )
            updated += 1
        except Exception as exc:
            errors.append(f"{record_date}: update failed: {exc}")

    return {
        "symbol": symbol,
        "records_found": len(records),
        "updated": updated,
        "skipped": skipped,
        "errors": errors[:20],
    }
