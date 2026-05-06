"""Best-effort titles when RSS/sitemap only expose bare URLs."""

from __future__ import annotations

from urllib.parse import urlparse, unquote


def _sentence_case_start(text: str) -> str:
    """Uppercase the first character; leave the rest unchanged (slug → readable title)."""
    s = text.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def title_from_article_url(url: str) -> str | None:
    """Build a readable headline from the last meaningful path segment.

    Handles common news patterns: ``/2024/05/01/some-article-slug/`` → ``some article slug``.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        path = (urlparse(url).path or "").strip("/")
        if not path:
            return None
        segments = [unquote(s) for s in path.split("/") if s]
        if not segments:
            return None
        # Drop leading ``YYYY`` (and optional ``MM``, ``DD``) path segments used by publishers.
        while segments and segments[0].isdigit() and len(segments[0]) == 4:
            segments = segments[1:]
        while segments and segments[0].isdigit():
            segments = segments[1:]
        if not segments:
            return None
        last = segments[-1]
        for ext in (".html", ".htm", ".php", ".aspx", ".jsp", ".shtml"):
            if last.lower().endswith(ext):
                last = last[: -len(ext)]
                break
        if last.isdigit() and len(segments) >= 2:
            last = segments[-2]
            for ext in (".html", ".htm", ".php", ".aspx", ".jsp", ".shtml"):
                if last.lower().endswith(ext):
                    last = last[: -len(ext)]
                    break
        if not last or len(last) < 3 or last.isdigit():
            return None
        readable = last.replace("-", " ").replace("_", " ").strip()
        readable = " ".join(readable.split())
        if len(readable) < 3:
            return None
        return _sentence_case_start(readable)[:500]
    except Exception:
        return None


def normalize_article_title(raw_title: str | None, url: str) -> str:
    """Return a headline: use RSS title unless it is empty or identical to the URL."""
    t = (raw_title or "").strip()
    u = (url or "").strip()
    if not u:
        return t or "Untitled"

    broken = (
        not t
        or t.startswith("http://")
        or t.startswith("https://")
        or t.rstrip("/") == u.rstrip("/")
        or t == u
    )
    if broken:
        hint = title_from_article_url(u)
        if hint:
            return hint
        return t or "Untitled"
    return t
