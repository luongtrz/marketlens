import feedparser
import requests
from bs4 import BeautifulSoup
from typing import List
import re
from urllib.parse import urlparse
import datetime
import xml.etree.ElementTree as ET
from .models import Article


def _clean_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ").strip()


def _clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace and remove duplicate adjacent lines.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    parts = [p.strip() for p in text.split(". ")]
    deduped = []
    seen = set()
    for p in parts:
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return ". ".join(deduped).strip()


def _is_low_quality_body(text: str, page_title: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if len(t) < 180:
        return True
    normalized_title = (page_title or "").strip().lower()
    normalized_text = t.lower()
    # Reject cases where extracted text is effectively just the title.
    if normalized_title and (normalized_text == normalized_title or normalized_text.startswith(normalized_title)):
        if len(normalized_text) < len(normalized_title) + 60:
            return True
    return False


def _extract_coindesk_article_body(soup: BeautifulSoup, page_title: str) -> str:
    """CoinDesk layout: [data-module-name='article-body'] > .document-body > <p>.

    Skips obvious chrome inside the body (ads, premium placeholders).
    """
    root = soup.select_one('[data-module-name="article-body"]')
    if not root:
        return ""
    doc = root.select_one("div.document-body")
    if not doc:
        doc = root.select_one(".document-body")
    if not doc:
        doc = root

    for noise in doc.select(
        ".article-ad, .premium-hide, .ad-desktop, .ad-mobile, [class*='article-ad']"
    ):
        noise.decompose()

    parts = []
    # Keep structural content blocks, not only paragraphs.
    for node in doc.find_all(["h2", "h3", "p", "li", "blockquote"]):
        t = node.get_text(" ", strip=True)
        if len(t) >= 12:
            parts.append(t)

    text = _clean_text(" ".join(parts))
    if not text or _is_low_quality_body(text, page_title):
        return ""
    return text[:5000]


def _extract_cointelegraph_article_body(soup: BeautifulSoup, page_title: str) -> str:
    """CoinTelegraph: main prose lives in div[data-testid='html-renderer-container'].

    Fallback: .ct-prose inside <article> (wrapper class names are CSS-module hashed).
    """
    root = soup.select_one('[data-testid="html-renderer-container"]')
    if not root:
        article = soup.find("article")
        if article:
            root = article.select_one(".ct-prose")
    if not root:
        return ""

    for noise in root.select(
        "aside, .advert, [class*='advert'], iframe, [data-testid*='ad']"
    ):
        noise.decompose()

    parts = []
    for p in root.find_all("p"):
        t = p.get_text(" ", strip=True)
        if len(t) >= 12:
            parts.append(t)
    if not parts:
        for li in root.find_all("li"):
            t = li.get_text(" ", strip=True)
            if len(t) >= 12:
                parts.append(t)
    text = _clean_text(" ".join(parts)) if parts else _clean_text(root.get_text(" ", strip=True))
    if not text or _is_low_quality_body(text, page_title):
        return ""
    return text[:5000]


def _extract_decrypt_article_body(soup: BeautifulSoup, page_title: str) -> str:
    """Decrypt.co: main copy lives in div.post-content (under <main>).

    Strips widget/ad rows (e.g. myriad widget, post-content-w-full promo rows) then collects <p>.
    """
    root = None
    main = soup.find("main")
    if main:
        root = main.select_one(".post-content")
    if not root:
        root = soup.select_one("main .post-content")
    if not root:
        root = soup.select_one(".post-content")
    if not root:
        if not soup.select_one("p.text-decryptBlack"):
            return ""

    for noise in root.select(
        "#myriad-pm-widget-content, [id*='myriad'], .post-content-w-full, aside, iframe"
    ):
        noise.decompose()

    parts = []
    # Decrypt frequently has a key summary paragraph above article body.
    lead = soup.select_one("p.text-decryptBlack.font-meta-serif-pro.text-xl.mt-2")
    if lead:
        lead_text = lead.get_text(" ", strip=True)
        if len(lead_text) >= 12:
            parts.append(lead_text)

    for p in root.find_all("p"):
        t = p.get_text(" ", strip=True)
        if len(t) >= 12:
            parts.append(t)
    text = _clean_text(" ".join(parts))
    if not text or _is_low_quality_body(text, page_title):
        return ""
    return text[:5000]


def fetch_article_content(url: str, timeout: int = 10) -> str:
    """Fetch the HTML at `url` and attempt to extract the main article text.

    Strategy:
    - Remove common noisy blocks (ads/nav/footer/related/comments)
    - Prefer semantic/article-like selectors
    - Fallback to largest meaningful content block
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "rss-crawler/1.0"})
        resp.raise_for_status()
        html = resp.content
    except Exception:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    page_title = ""
    if soup.title:
        page_title = soup.title.get_text(" ", strip=True)

    # Light cleanup only first so site-specific selectors still see full DOM.
    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()

    host = (urlparse(url).hostname or "").lower()
    if host.endswith("coindesk.com"):
        coindesk_text = _extract_coindesk_article_body(soup, page_title)
        if coindesk_text:
            return coindesk_text

    if host.endswith("cointelegraph.com"):
        ct_text = _extract_cointelegraph_article_body(soup, page_title)
        if ct_text:
            return ct_text

    if host == "decrypt.co" or host.endswith(".decrypt.co"):
        decrypt_text = _extract_decrypt_article_body(soup, page_title)
        if decrypt_text:
            return decrypt_text

    for bad in soup(["header", "footer", "svg", "img", "nav", "aside", "form"]):
        bad.decompose()

    # Remove common noisy sections by class/id hints.
    noisy_hints = (
        "related", "recommend", "footer", "header", "comment", "social",
        "share", "newsletter", "subscribe", "promo", "advert", "ads", "banner",
        "breadcrumb", "sidebar", "tag-list", "most-read",
    )
    for el in soup.find_all(True):
        attrs = " ".join(
            [
                " ".join(el.get("class", [])),
                el.get("id", ""),
                el.get("role", ""),
                el.get("aria-label", ""),
            ]
        ).lower()
        if any(h in attrs for h in noisy_hints):
            el.decompose()

    # Prefer semantic/article-like selectors.
    preferred_selectors = [
        "article",
        "main article",
        "main",
        "[itemprop='articleBody']",
        ".article-content",
        ".post-content",
        ".entry-content",
        ".content-body",
        ".article-body",
    ]
    for selector in preferred_selectors:
        for node in soup.select(selector):
            text = _clean_text(node.get_text(separator=" "))
            if len(text) >= 200 and not _is_low_quality_body(text, page_title):
                return text[:5000]

    candidates = []
    for name in ("main", "section", "div"):
        for el in soup.find_all(name):
            txt = _clean_text(el.get_text(separator=" "))
            if len(txt) < 200:
                continue
            # Penalize blocks that are link-heavy (often nav/related lists).
            links = el.find_all("a")
            text_len = len(txt)
            link_density = (sum(len(a.get_text(" ", strip=True)) for a in links) / text_len) if text_len else 1
            if link_density > 0.35:
                continue
            candidates.append((text_len, txt))

    if candidates:
        # pick the largest block
        candidates.sort(reverse=True, key=lambda x: x[0])
        best = candidates[0][1][:5000]
        if not _is_low_quality_body(best, page_title):
            return best

    # fallback to body
    body = soup.body
    if body:
        fallback = _clean_text(body.get_text(separator=" "))[:5000]
        if not _is_low_quality_body(fallback, page_title):
            return fallback

    return ""


def fetch_feed(url: str, timeout: int = 10) -> List[Article]:
    """Fetch and parse RSS/Atom feed at `url` and return list of Article."""
    try:
        # feedparser can accept the URL directly but using requests gives timeout control
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "rss-crawler/1.0"})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception:
        parsed = feedparser.parse(url)

    articles = []
    for entry in parsed.entries:
        entry_id = entry.get("id") or entry.get("guid") or entry.get("link") or (entry.get("title") or "").strip()
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        published = entry.get("published") or entry.get("updated") or None
        summary = entry.get("summary") or entry.get("description") or None
        content_blocks = entry.get("content") or []
        if content_blocks and isinstance(content_blocks, list):
            first = content_blocks[0]
            if isinstance(first, dict):
                block = first.get("value")
                if block:
                    cleaned_block = _clean_html(block)
                    if cleaned_block and len(cleaned_block) > len(summary or ""):
                        summary = cleaned_block
        if summary:
            summary = _clean_html(summary)

        articles.append(Article(id=str(entry_id), title=title, link=link, published=published, summary=summary))

    return articles


def _parse_date_like(value: str):
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


# e.g. coindesk.com/.../2026/04/09/slug
_URL_DATE_RE = re.compile(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/")


def _parse_date_from_article_url(loc: str):
    if not loc:
        return None
    m = _URL_DATE_RE.search(loc)
    if not m:
        return None
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime.datetime(y, mo, d, tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


def _extract_sitemap_rows(xml_bytes: bytes):
    rows = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return rows

    tag = root.tag.lower()
    # urlset
    if tag.endswith("urlset"):
        for url_node in root.findall(".//{*}url"):
            loc = url_node.findtext("{*}loc")
            lastmod = url_node.findtext("{*}lastmod")
            if loc:
                rows.append((loc.strip(), (lastmod or "").strip() or None))
    # sitemapindex
    elif tag.endswith("sitemapindex"):
        for sm_node in root.findall(".//{*}sitemap"):
            loc = sm_node.findtext("{*}loc")
            lastmod = sm_node.findtext("{*}lastmod")
            if loc:
                rows.append((loc.strip(), (lastmod or "").strip() or None))
    return rows


def fetch_sitemap_urls(
    source_url: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    max_urls: int = 20000,
    timeout: int = 10,
    max_sitemap_files: int = 0,
) -> List[dict]:
    """Discover historical article URLs from sitemap for a source domain.

    max_sitemap_files: max child sitemap XML files to open (after sorting by lastmod, newest first).
    0 means no limit (all discovered child sitemaps are processed).
    """
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

    headers = {"User-Agent": "rss-crawler/1.0"}
    discovered = []
    seen_urls = set()
    min_utc = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    child_sitemaps = []

    def _child_sitemap_sort_key(item):
        lm = item[1]
        dt = _parse_date_like(lm) if lm else None
        return dt if dt is not None else min_utc

    seen_child = set()

    for sm in sitemap_candidates:
        try:
            resp = requests.get(sm, timeout=timeout, headers=headers)
            if resp.status_code >= 400:
                continue
            rows = _extract_sitemap_rows(resp.content)
            if not rows:
                continue
            if rows and rows[0][0].endswith(".xml"):
                for loc, lm in rows:
                    if loc in seen_child:
                        continue
                    seen_child.add(loc)
                    child_sitemaps.append((loc, lm))
            else:
                if sm in seen_child:
                    continue
                seen_child.add(sm)
                child_sitemaps.append((sm, None))
        except Exception:
            continue

    child_sitemaps.sort(key=_child_sitemap_sort_key, reverse=True)
    if max_sitemap_files > 0:
        child_sitemaps = child_sitemaps[:max_sitemap_files]

    for sm_url, _ in child_sitemaps:
        if len(discovered) >= max_urls:
            break
        try:
            resp = requests.get(sm_url, timeout=timeout, headers=headers)
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

            dt = _parse_date_like(lastmod)
            if dt is None:
                dt = _parse_date_from_article_url(loc)
            if dt is None:
                continue
            if not (start_date <= dt <= end_date):
                continue
            published = dt.isoformat()

            discovered.append(
                {
                    "id": loc,
                    "title": "",
                    "link": loc,
                    "published": published,
                    "summary": None,
                }
            )

    return discovered
