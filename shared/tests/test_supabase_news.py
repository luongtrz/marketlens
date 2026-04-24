"""Tests for Supabase news row mapping."""

from datetime import datetime, timezone

from shared.supabase_news import supabase_row_to_ingestion


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
    assert rec.raw_text == "Full body"
    assert rec.sentiment_label == "neutral"
    assert rec.date_published == datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    assert rec.date_crawled == datetime(2026, 4, 1, 12, 5, tzinfo=timezone.utc)


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
