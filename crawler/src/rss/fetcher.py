"""RSS feed utilities — fetches entries, sitemap URLs and article content."""

import asyncio
import datetime
import re
import xml.etree.ElementTree as et
from typing import Awaitable, Callable
from urllib.parse import urlparse

import feedparser
import httpx
from pydantic import BaseModel

from crawler.src.rss.parser import FeedParser
from crawler.src.rss.title_hints import normalize_article_title
from shared.models.article import RawArticle


class FeedSource(BaseModel):
    """Configuration for a single RSS feed source."""

    name: str
    url: str
    category: str


class RSSFetcher:
    """Continuously polls RSS feeds and yields new articles."""

    def __init__(
        self,
        sources: list[FeedSource],
        poll_interval_seconds: int,
        timeout_seconds: int = 15,
    ) -> None:
        self._sources = sources
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._parser = FeedParser()
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"User-Agent": "rss-crawler/2.0"},
            follow_redirects=True,
        )

    async def poll_forever(
        self,
        on_cycle: Callable[[list[RawArticle]], Awaitable[None]],
    ) -> None:
        """Start infinite polling loop and dispatch each cycle's entries."""
        while True:
            articles = await self.fetch_all()
            await on_cycle(articles)
            await asyncio.sleep(self._poll_interval_seconds)

    async def fetch_all(self) -> list[RawArticle]:
        """Fetch RSS entries from every configured source."""
        results: list[RawArticle] = []
        for source in self._sources:
            results.extend(await self.fetch_one(source))
        return results

    async def fetch_one(self, source: FeedSource) -> list[RawArticle]:
        """Fetch and parse all entries from one RSS source."""
        try:
            response = await self._client.get(source.url)
            response.raise_for_status()
            parsed = feedparser.parse(response.text)
        except Exception:
            return []

        items: list[RawArticle] = []
        for entry in parsed.entries:
            article = self._parser.parse(dict(entry), source.name, source.category)
            if article.url:
                items.append(article)
        return items

    async def fetch_sitemap_urls(
        self,
        source_url: str,
        source_name: str,
        category: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_urls: int = 20000,
        max_sitemap_files: int = 0,
    ) -> list[RawArticle]:
        """Discover article URLs from sitemap indexes for historical backfill."""
        host = (urlparse(source_url).hostname or "").lower()
        if not host:
            return []

        sitemap_candidates = [
            f"https://{host}/sitemap.xml",
            f"https://{host}/sitemap_index.xml",
            f"https://{host}/sitemap-index.xml",
            f"https://{host}/post-sitemap.xml",
            f"https://{host}/news-sitemap.xml",
        ]

        child_sitemaps: list[tuple[str, str | None]] = []
        seen_child: set[str] = set()
        for sm in sitemap_candidates:
            try:
                resp = await self._client.get(sm)
                if resp.status_code >= 400:
                    continue
                rows = _extract_sitemap_rows(resp.content)
                if not rows:
                    continue
                if rows and rows[0][0].endswith(".xml"):
                    for loc, lm in rows:
                        if loc not in seen_child:
                            seen_child.add(loc)
                            child_sitemaps.append((loc, lm))
                elif sm not in seen_child:
                    seen_child.add(sm)
                    child_sitemaps.append((sm, None))
            except Exception:
                continue

        child_sitemaps.sort(
            key=lambda item: _parse_date_like(item[1]) or datetime.datetime.min.replace(
                tzinfo=datetime.timezone.utc
            ),
            reverse=True,
        )
        if max_sitemap_files > 0:
            child_sitemaps = child_sitemaps[:max_sitemap_files]

        discovered: list[RawArticle] = []
        seen_urls: set[str] = set()
        for sm_url, _ in child_sitemaps:
            if len(discovered) >= max_urls:
                break
            try:
                resp = await self._client.get(sm_url)
                if resp.status_code >= 400:
                    continue
                rows = _extract_sitemap_rows(resp.content)
            except Exception:
                continue

            for loc, lastmod in rows:
                if len(discovered) >= max_urls:
                    break
                if loc in seen_urls:
                    continue
                seen_urls.add(loc)
                if (urlparse(loc).hostname or "").lower() != host:
                    continue
                dt = _parse_date_like(lastmod) or _parse_date_from_article_url(loc)
                if dt is None or not (start_date <= dt <= end_date):
                    continue
                discovered.append(
                    RawArticle(
                        title=normalize_article_title(None, loc),
                        url=loc,
                        source=source_name,
                        category=category,
                        published=dt,
                        text=None,
                    )
                )
        return discovered

    async def fetch_article_content(self, url: str) -> str:
        """Fetch and extract article body text. Uses BeautifulSoup when available."""
        content, _ = await self.fetch_article_page(url)
        return content

    async def fetch_article_headline(self, url: str) -> str:
        """Best-effort ``og:title`` / ``<title>`` from article HTML."""
        _, headline = await self.fetch_article_page(url)
        return headline

    async def fetch_article_page(self, url: str) -> tuple[str, str]:
        """Single GET: return ``(body_text, headline)``."""

        try:
            resp = await self._client.get(url)
            if resp.status_code >= 400:
                return "", ""
            html = resp.text
        except Exception:
            return "", ""

        try:
            from bs4 import BeautifulSoup
        except Exception:
            m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
            headline = _clean_title_tag(m.group(1)) if m else ""
            return _clean_text(_strip_tags(html))[:5000], headline

        soup = BeautifulSoup(html, "html.parser")
        headline = _headline_from_soup(soup)
        for bad in soup(["script", "style", "noscript"]):
            bad.decompose()

        host = (urlparse(url).hostname or "").lower()
        if host.endswith("cryptoslate.com"):
            root = soup.select_one(".post-box__content-flow")
            if root:
                return _extract_by_nodes(root), headline
        if host.endswith("theblock.co"):
            root = (
                soup.select_one("#articleContent .dynamic-content")
                or soup.select_one("#articleContent")
                or soup.select_one("article .article-content .dynamic-content")
            )
            if root:
                return _extract_by_nodes(root), headline
        if host.endswith("cointelegraph.com"):
            root = soup.select_one('[data-testid="html-renderer-container"]')
            if root:
                return _extract_by_nodes(root), headline
        if host.endswith("coindesk.com"):
            root = soup.select_one('[data-module-name="article-body"]')
            if root:
                return _extract_by_nodes(root), headline
        if host.endswith("decrypt.co"):
            root = soup.select_one("main .post-content") or soup.select_one(".post-content")
            if root:
                return _extract_by_nodes(root), headline

        for selector in ("article", "main article", "main", ".article-content", ".post-content"):
            for node in soup.select(selector):
                text = _extract_by_nodes(node)
                if len(text) >= 200:
                    return text[:5000], headline
        return _clean_text(soup.get_text(" ", strip=True))[:5000], headline


def _headline_from_soup(soup: object) -> str:
    try:
        og = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "og:title"})  # type: ignore[attr-defined]
        if og and og.get("content"):
            return _clean_title_tag(str(og["content"]))
        title_tag = soup.title  # type: ignore[attr-defined]
        if title_tag and title_tag.string:
            return _clean_title_tag(title_tag.string)
    except Exception:
        pass
    return ""


def _clean_title_tag(raw: str) -> str:
    title = _clean_text(raw)
    for sep in (" | ", " - ", " — ", " – "):
        if sep in title:
            title = title.split(sep, 1)[0].strip()
            break
    return title[:500]


def _extract_by_nodes(root: object) -> str:
    try:
        nodes = root.find_all(["h2", "h3", "p", "li", "blockquote"])  # type: ignore[attr-defined]
        parts = []
        for node in nodes:
            text = node.get_text(" ", strip=True)
            if len(text) >= 12:
                parts.append(text)
        if parts:
            return _clean_text(" ".join(parts))[:5000]
        return _clean_text(root.get_text(" ", strip=True))[:5000]  # type: ignore[attr-defined]
    except Exception:
        return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _parse_date_like(value: str | None) -> datetime.datetime | None:
    if not value:
        return None
    try:
        from dateutil import parser as date_parser

        dt = date_parser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except Exception:
        return None


_URL_DATE_RE = re.compile(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/")


def _parse_date_from_article_url(url: str) -> datetime.datetime | None:
    match = _URL_DATE_RE.search(url)
    if not match:
        return None
    try:
        return datetime.datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            tzinfo=datetime.timezone.utc,
        )
    except ValueError:
        return None


def _extract_sitemap_rows(xml_bytes: bytes) -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = []
    try:
        root = et.fromstring(xml_bytes)
    except Exception:
        return rows
    tag = root.tag.lower()
    if tag.endswith("urlset"):
        for url_node in root.findall(".//{*}url"):
            loc = url_node.findtext("{*}loc")
            lastmod = url_node.findtext("{*}lastmod")
            if loc:
                rows.append((loc.strip(), (lastmod or "").strip() or None))
    elif tag.endswith("sitemapindex"):
        for sm_node in root.findall(".//{*}sitemap"):
            loc = sm_node.findtext("{*}loc")
            lastmod = sm_node.findtext("{*}lastmod")
            if loc:
                rows.append((loc.strip(), (lastmod or "").strip() or None))
    return rows
