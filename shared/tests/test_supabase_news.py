"""Tests for Supabase news row mapping."""

from datetime import datetime, timezone

from shared.supabase_news import _row_text_matches_symbol, normalize_news_source_host, supabase_row_to_ingestion


def test_normalize_news_source_host() -> None:
    assert normalize_news_source_host(" CoinTelegraph.COM ") == "cointelegraph.com"
    assert normalize_news_source_host("crypto.news") == "crypto.news"
    assert normalize_news_source_host(None) is None
    assert normalize_news_source_host("") is None
    assert normalize_news_source_host("bad/host") is None
    assert normalize_news_source_host("../x") is None


def test_supabase_row_to_ingestion_maps_fields() -> None:
    row = {
        "header": "Test headline",
        "content": "Full body",
        "source_url": "https://www.example.com/news/1",
        "publish_at": "2026-04-01T12:00:00+00:00",
        "crawled_at": "2026-04-01T12:05:00+00:00",
    }
    rec = supabase_row_to_ingestion(row)
    assert rec.article_name == "Test headline"
    assert rec.url == "https://www.example.com/news/1"
    assert rec.source == "example.com"
    assert rec.summary == "Full body"
    assert rec.raw_text == "Full body"
    assert rec.sentiment_label == "neutral"
    assert rec.sentiment_score == 0.0
    assert rec.date_published == datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)


def test_supabase_row_uses_summary_column_when_present() -> None:
    row = {
        "id": "c",
        "header": "H",
        "content": "Long body content " * 50,
        "summary": "  DB summary line  ",
        "source_url": "https://x.test/c",
        "publish_at": "2026-01-01T00:00:00Z",
    }
    rec = supabase_row_to_ingestion(row)
    assert rec.summary == "DB summary line"


def test_supabase_row_derives_sentiment_label_from_score() -> None:
    row = {
        "id": "a",
        "header": "H",
        "content": "",
        "source_url": "https://x.test/a",
        "publish_at": "2026-01-01T00:00:00Z",
        "sentiment_score": 0.6,
    }
    rec = supabase_row_to_ingestion(row)
    assert rec.sentiment_score == 0.6
    assert rec.sentiment_label == "bullish"


def test_supabase_row_ui_scale_sentiment() -> None:
    row = {
        "id": "b",
        "header": "H",
        "source_url": "https://x.test/b",
        "publish_at": "2026-01-01T00:00:00Z",
        "sentiment_score": 80.0,
    }
    rec = supabase_row_to_ingestion(row)
    assert abs(rec.sentiment_score - 0.6) < 1e-6
    assert rec.sentiment_label == "bullish"
    assert rec.date_crawled == datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_supabase_row_uses_id_when_present() -> None:
    row = {
        "id": "row-uuid",
        "header": "H",
        "content": "",
        "source_url": "https://x.test/a",
        "publish_at": "2026-01-01T00:00:00Z",
    }
    rec = supabase_row_to_ingestion(row)
    assert rec.id == "row-uuid"


def test_symbol_filter_eth_does_not_match_tether_substring() -> None:
    assert not _row_text_matches_symbol(
        "Blockspace: Tether’s USDtb and stablecoins",
        "",
        "ETHUSDT",
    )


def test_symbol_filter_eth_matches_ethereum_terms() -> None:
    assert _row_text_matches_symbol(
        "Ethereum Foundation sells ETH",
        "",
        "ETHUSDT",
    )
    assert _row_text_matches_symbol("ETH staking update", "", "ETH")


def test_symbol_filter_btc_uses_whole_words() -> None:
    assert _row_text_matches_symbol("Bitcoin ETF flows", "", "BTCUSDT")
    assert not _row_text_matches_symbol(
        "Unrelated tether headline without bitcoin",
        "",
        "BTCUSDT",
    )
