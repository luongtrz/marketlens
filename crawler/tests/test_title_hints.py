"""Tests for RSS/sitemap title derivation."""

from crawler.src.rss.parser import FeedParser
from crawler.src.rss.title_hints import normalize_article_title, title_from_article_url


def test_title_from_url_skips_date_prefix() -> None:
    url = "https://cointelegraph.com/news/crypto-biz-capital-has-no-consensus"
    assert title_from_article_url(url) == "Crypto biz capital has no consensus"


def test_normalize_replaces_url_shaped_title() -> None:
    url = "https://decrypt.co/12345/some-story-here"
    assert normalize_article_title(url, url) == "Some story here"


def test_normalize_keeps_real_title() -> None:
    url = "https://example.com/a/b"
    assert normalize_article_title("Real headline", url) == "Real headline"


def test_parser_uses_slug_when_title_is_url() -> None:
    p = FeedParser()
    entry = {
        "title": "https://www.coindesk.com/markets/2024/01/01/btc/",
        "link": "https://www.coindesk.com/markets/2024/01/01/btc/",
        "published_parsed": (2024, 1, 1, 12, 0, 0),
    }
    a = p.parse(entry, "CoinDesk", "crypto")
    assert a.title == "Btc"
    assert a.url.startswith("https://")
