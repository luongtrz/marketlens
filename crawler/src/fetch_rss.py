import feedparser
import requests
from bs4 import BeautifulSoup
from typing import List
from .models import Article


def _clean_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ").strip()


def fetch_article_content(url: str, timeout: int = 10) -> str:
    """Fetch the HTML at `url` and attempt to extract the main article text.

    Strategy:
    - Prefer <article> tags
    - Otherwise choose the element among <main>, <div>, <section> with the largest text length
    - Fallback to body text
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "rss-crawler/1.0"})
        resp.raise_for_status()
        html = resp.content
    except Exception:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for bad in soup(["script", "style", "noscript", "header", "footer", "svg", "img"]):
        bad.decompose()

    # Prefer semantic article tag
    article_tag = soup.find("article")
    if article_tag:
        text = article_tag.get_text(separator=" ").strip()
        if len(text) > 50:
            return text

    candidates = []
    for name in ("main", "section", "div"):
        for el in soup.find_all(name):
            txt = el.get_text(separator=" ").strip()
            if len(txt) > 100:
                candidates.append((len(txt), txt))

    if candidates:
        # pick the largest block
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1]

    # fallback to body
    body = soup.body
    if body:
        return body.get_text(separator=" ").strip()

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
        if summary:
            summary = _clean_html(summary)

        articles.append(Article(id=str(entry_id), title=title, link=link, published=published, summary=summary))

    return articles
