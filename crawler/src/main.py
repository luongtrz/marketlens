"""Crawler module entry point — starts the RSS polling loop."""

import asyncio
import datetime
import json
import logging
import os
import uuid
from pathlib import Path

import httpx
from crawler.src.config import CrawlerConfig
from crawler.src.db.writer import IngestionDBWriter
from crawler.src.llm.client import LLMClient
from crawler.src.rss.deduplicator import Deduplicator
from crawler.src.rss.fetcher import FeedSource, RSSFetcher
from shared.asset_tags import detect_asset_tags
from shared.config.loader import load_yaml
from shared.models.article import IngestionRecord, RawArticle

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _load_env_files() -> None:
    """Load environment variables from crawler/.env (preferred) and .env."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    module_root = Path(__file__).resolve().parents[1]  # .../crawler
    project_root = module_root.parent  # .../marketlens

    # Keep explicit shell env highest priority.
    load_dotenv(module_root / ".env", override=False)
    load_dotenv(project_root / ".env", override=False)


async def start() -> None:
    """Initialize and start the Crawler polling loop."""
    _load_env_files()
    loaded = load_yaml("crawler/config.yaml")
    config_data = loaded.get("crawler", loaded) if isinstance(loaded, dict) else {}
    # Keep YAML defaults, but let CRAWLER_* environment variables win.
    field_names = set(CrawlerConfig.model_fields)
    merged_config: dict[str, object] = {}
    for key, value in (config_data or {}).items():
        if key not in field_names:
            continue
        env_key = f"CRAWLER_{key.upper()}"
        if env_key in os.environ:
            continue
        merged_config[key] = value
    config = CrawlerConfig(**merged_config)
    logger.info("Starting Crawler with poll_interval=%ds", config.poll_interval_seconds)
    logger.info(
        "Config: db_url=%s, dedup=%s, fetch_content=%s, sitemap_backfill=%s, feeds=%d",
        config.db_url,
        config.dedup_backend,
        config.fetch_content,
        config.enable_sitemap_backfill,
        len(config.feeds),
    )
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    seen_file = Path(config.seen_file)
    articles_file = Path(config.articles_file)
    seen_file.parent.mkdir(parents=True, exist_ok=True)
    articles_file.parent.mkdir(parents=True, exist_ok=True)

    seen_urls = _load_seen(seen_file)
    dedup = Deduplicator(backend=config.dedup_backend, redis_url=config.redis_url)
    for url in seen_urls:
        await dedup.mark_seen(url)

    sources = [FeedSource(**f.model_dump()) for f in config.feeds]
    fetcher = RSSFetcher(sources=sources, poll_interval_seconds=config.poll_interval_seconds)
    llm = LLMClient(aihub_url=config.aihub_url, enable_summary=config.enable_summary)
    writer = IngestionDBWriter(config.db_url)
    supabase_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase_table = os.getenv("SUPABASE_TABLE", "news_articles")
    supabase_enabled = bool(supabase_url and supabase_key)
    logger.info("Supabase sink enabled=%s", supabase_enabled)
    supabase_client = httpx.AsyncClient(timeout=20) if supabase_enabled else None

    while True:
        cycle_count = 0
        logger.info("Cycle started")
        rss_articles = await fetcher.fetch_all()
        logger.info("Fetched RSS articles: %d", len(rss_articles))
        all_articles: list[RawArticle] = list(rss_articles)

        if config.enable_sitemap_backfill:
            now = datetime.datetime.now(datetime.timezone.utc)
            start_date = datetime.datetime(config.min_publish_year, 1, 1, tzinfo=datetime.timezone.utc)
            for source in sources:
                logger.info("Backfilling sitemap for source=%s", source.name)
                before = len(all_articles)
                all_articles.extend(
                    await fetcher.fetch_sitemap_urls(
                        source_url=source.url,
                        source_name=source.name,
                        category=source.category,
                        start_date=start_date,
                        end_date=now,
                        max_urls=config.sitemap_max_urls_per_source,
                        max_sitemap_files=config.sitemap_max_sitemap_files,
                    )
                )
                logger.info(
                    "Sitemap source=%s added=%d (total=%d)",
                    source.name,
                    len(all_articles) - before,
                    len(all_articles),
                )

        logger.info("Processing total candidate articles=%d", len(all_articles))
        processed = 0
        for article in all_articles:
            if not article.url:
                continue
            if await dedup.is_seen(article.url):
                continue

            published = article.published
            if published is None:
                await dedup.mark_seen(article.url)
                seen_urls.add(article.url)
                continue
            if published.year < config.min_publish_year:
                await dedup.mark_seen(article.url)
                seen_urls.add(article.url)
                continue

            content = article.text
            page_headline = ""
            if config.fetch_content:
                content, page_headline = await fetcher.fetch_article_page(article.url)
                content = content or article.text

            asset_tags = detect_asset_tags(article.title, content or "", article.url)
            if not asset_tags and page_headline and page_headline.strip() != (article.title or "").strip():
                alt_tags = detect_asset_tags(page_headline, content or "", article.url)
                if alt_tags:
                    article = article.model_copy(update={"title": page_headline.strip()})
                    asset_tags = alt_tags

            if config.only_btc_eth and not asset_tags:
                await dedup.mark_seen(article.url)
                seen_urls.add(article.url)
                continue

            enriched = await llm.enrich(
                RawArticle(
                    title=article.title,
                    url=article.url,
                    source=article.source,
                    category=article.category,
                    published=published,
                    text=content,
                )
            )

            if enriched.sentiment_score > 0.15:
                sentiment_label = "bullish"
            elif enriched.sentiment_score < -0.15:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"

            record = IngestionRecord(
                id=str(uuid.uuid4()),
                article_name=article.title,
                source=article.source,
                url=article.url,
                date_published=published,
                date_crawled=datetime.datetime.now(datetime.timezone.utc),
                summary=enriched.summary,
                sentiment_score=enriched.sentiment_score,
                sentiment_label=sentiment_label,
                factors=enriched.factors,
                raw_text=content,
                metadata={"category": article.category, "asset_tags": sorted(asset_tags)},
            )
            try:
                await writer.write(record)
            except Exception as exc:
                # Local ingestion DB is optional for crawler runs that only need Supabase.
                logger.warning("Local DB write failed, continue with Supabase: %s", str(exc)[:200])
            if supabase_client is not None:
                await _write_to_supabase(
                    client=supabase_client,
                    supabase_url=supabase_url,
                    supabase_key=supabase_key,
                    supabase_table=supabase_table,
                    record=record,
                )
            _append_article_jsonl(articles_file, record)

            await dedup.mark_seen(article.url)
            seen_urls.add(article.url)
            cycle_count += 1
            processed += 1
            if processed % 20 == 0:
                logger.info("Progress: processed=%d, inserted=%d", processed, cycle_count)

        _save_seen(seen_file, seen_urls)
        logger.info("Crawler cycle finished, new_records=%d", cycle_count)
        await asyncio.sleep(config.poll_interval_seconds)


def _load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _save_seen(path: Path, values: set[str]) -> None:
    path.write_text(json.dumps(sorted(values), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_article_jsonl(path: Path, record: IngestionRecord) -> None:
    row = {
        "id": record.id,
        "title": record.article_name,
        "link": record.url,
        "source": record.source,
        "published": record.date_published.isoformat(),
        "summary": record.summary,
        "content": record.raw_text,
        "factors": record.factors,
        "sentiment": record.sentiment_score,
    }
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_supabase_row(record: IngestionRecord) -> dict[str, object]:
    content = (record.raw_text or record.summary or record.article_name or "").strip()
    asset_tags = detect_asset_tags(record.article_name, content, record.url)
    coin = sorted(asset_tags) if asset_tags else ["General"]
    payload: dict[str, object] = {
        "header": (record.article_name or "Untitled")[:200],
        "content": content[:5000],
        "publish_at": record.date_published.isoformat(),
        "crawled_at": record.date_crawled.isoformat(),
        "source_url": record.url,
        "sentiment_score": record.sentiment_score,
        "coin": coin,
    }
    if record.summary and str(record.summary).strip():
        payload["summary"] = str(record.summary).strip()[:8000]
    return payload


async def _write_to_supabase(
    client: httpx.AsyncClient,
    supabase_url: str,
    supabase_key: str,
    supabase_table: str,
    record: IngestionRecord,
) -> None:
    payload = _build_supabase_row(record)
    endpoint = f"{supabase_url}/rest/v1/{supabase_table}?on_conflict=source_url"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    try:
        resp = await client.post(endpoint, headers=headers, json=[payload])
        if not (200 <= resp.status_code < 300):
            logger.warning(
                "Supabase insert failed status=%s body=%s",
                resp.status_code,
                (resp.text or "")[:200],
            )
    except Exception as exc:
        logger.warning("Supabase insert error: %s", str(exc)[:200])


if __name__ == "__main__":
    asyncio.run(start())
