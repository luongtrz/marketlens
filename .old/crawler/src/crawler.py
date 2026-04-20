import json
import os
import re
import time
import datetime
from typing import Set
from urllib.parse import urlparse
from pathlib import Path
import sys
import requests

# Ensure repository root is on sys.path so absolute imports like `src.config`
# work when this file is executed directly (python src/crawler.py).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

 
# Support running as a package or as a script. When executed directly
# the relative imports may fail with "no known parent package", so fall
# back to absolute imports from the `src` package.
try:
    from .config import DEFAULT_FEEDS, DATA_DIR, SEEN_FILE, ARTICLES_FILE, FETCH_TIMEOUT, SOURCES
    from .fetch_rss import fetch_feed, fetch_article_content, fetch_sitemap_urls
except Exception:
    from src.config import DEFAULT_FEEDS, DATA_DIR, SEEN_FILE, ARTICLES_FILE, FETCH_TIMEOUT, SOURCES
    from src.fetch_rss import fetch_feed, fetch_article_content, fetch_sitemap_urls


class Crawler:
    def __init__(self, feeds=None, seen_file=SEEN_FILE, articles_file=ARTICLES_FILE, timeout=FETCH_TIMEOUT, fetch_content: bool = True, backend_url: str = None, analysis_url: str = None):
        # `feeds` may be a list of strings or dicts with a `url` key.
        self.feeds = feeds or SOURCES
        self.seen_file = os.getenv("CRAWLER_SEEN_FILE", seen_file)
        self.articles_file = os.getenv("CRAWLER_ARTICLES_FILE", articles_file)
        self.timeout = timeout
        self.fetch_content = fetch_content
        self.backend_url = backend_url or os.getenv("BACKEND_URL", "http://localhost:8080")
        self.analysis_url = analysis_url or os.getenv("ANALYSIS_URL", "http://localhost:8000")
        self.enable_analysis = os.getenv("ENABLE_ANALYSIS", "false").lower() == "true"
        self.enable_sitemap_backfill = os.getenv("ENABLE_SITEMAP_BACKFILL", "true").lower() == "true"
        self.sitemap_max_urls_per_source = int(os.getenv("SITEMAP_MAX_URLS_PER_SOURCE", "120000"))
        # 0 = all discovered child sitemap XML files (replaces old hard cap of 30 in fetch_sitemap_urls).
        self.sitemap_max_sitemap_files = int(os.getenv("SITEMAP_MAX_SITEMAP_FILES", "0"))
        self.supabase_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.supabase_table = os.getenv("SUPABASE_TABLE", "news_articles")
        self.min_publish_date = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        self.feed_include = [
            x.strip().lower()
            for x in (os.getenv("CRAWLER_FEED_INCLUDE", "")).split(",")
            if x.strip()
        ]
        self.feed_exclude = [
            x.strip().lower()
            for x in (os.getenv("CRAWLER_FEED_EXCLUDE", "")).split(",")
            if x.strip()
        ]
        os.makedirs(DATA_DIR, exist_ok=True)
        # Ensure custom output locations can be created when running multiple processes.
        seen_parent = os.path.dirname(self.seen_file)
        articles_parent = os.path.dirname(self.articles_file)
        if seen_parent:
            os.makedirs(seen_parent, exist_ok=True)
        if articles_parent:
            os.makedirs(articles_parent, exist_ok=True)
        self.seen: Set[str] = self._load_seen()

    def _feed_allowed(self, feed_url: str, feed_name: str) -> bool:
        key = f"{feed_name} {feed_url}".lower()
        if self.feed_include:
            if not any(token in key for token in self.feed_include):
                return False
        if self.feed_exclude:
            if any(token in key for token in self.feed_exclude):
                return False
        return True

    def _load_seen(self) -> Set[str]:
        if os.path.exists(self.seen_file):
            try:
                with open(self.seen_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data if isinstance(data, list) else [])
            except Exception:
                return set()
        return set()

    def _save_seen(self):
        try:
            with open(self.seen_file, "w", encoding="utf-8") as f:
                json.dump(list(self.seen), f, indent=2)
        except Exception:
            pass

    def _append_article(self, article_dict: dict):
        try:
            with open(self.articles_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(article_dict, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _extract_money(self, text: str):
        if not text:
            return []
        # Match patterns like $12,345.67 or USD 12,345 or 12,345 USD
        money_re = re.compile(r"(\$|USD\s?)\s?([\d,]+(?:\.\d{1,2})?)|([\d,]+(?:\.\d{1,2})?)\s?(USD|\$)", re.IGNORECASE)
        results = []
        for m in money_re.finditer(text):
            groups = [g for g in m.groups() if g]
            results.append(" ".join(groups))
        return results

    def _determine_tag(self, title: str, content: str) -> str:
        text = (title + " " + content).lower()
        if "btc" in text or "bitcoin" in text:
            return "BTC"
        if "eth" in text or "ethereum" in text or "etherium" in text:
            return "ETH"
        return "General"

    def _parse_published_date(self, published: str):
        if not published:
            return None
        try:
            from dateutil import parser as date_parser
            parsed_date = date_parser.parse(published)
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=datetime.timezone.utc)
            return parsed_date.astimezone(datetime.timezone.utc)
        except Exception:
            return None

    def _is_in_target_date_range(self, published: str) -> bool:
        parsed_date = self._parse_published_date(published)
        if parsed_date is None:
            return False
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        return self.min_publish_date <= parsed_date <= now_utc

    def _fetch_title_from_url(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": "rss-crawler/1.0"})
            if resp.status_code >= 400:
                return ""
            m = re.search(r"<title[^>]*>(.*?)</title>", resp.text, flags=re.IGNORECASE | re.DOTALL)
            if not m:
                return ""
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            # Remove common title suffixes after separators.
            for sep in (" | ", " - ", " — "):
                if sep in title:
                    title = title.split(sep)[0].strip()
                    break
            return title[:200]
        except Exception:
            return ""

    def _store_to_supabase(self, row_data: dict) -> bool:
        if not self.supabase_url or not self.supabase_service_role_key:
            print("[crawler] Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY; skipping Supabase insert")
            return False

        headers = {
            "apikey": self.supabase_service_role_key,
            "Authorization": f"Bearer {self.supabase_service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        endpoint = f"{self.supabase_url}/rest/v1/{self.supabase_table}?on_conflict=source_url"

        try:
            resp = requests.post(endpoint, headers=headers, json=[row_data], timeout=15)
            if 200 <= resp.status_code < 300:
                return True
            print(f"[crawler] Supabase insert failed: {resp.status_code}")
            print(f"[crawler] Response: {resp.text[:200]}")
            return False
        except Exception as e:
            print(f"[crawler] Error inserting into Supabase: {str(e)[:200]}")
            return False

    def _build_supabase_row(self, article_data: dict):
        parsed_date = self._parse_published_date(article_data.get("published"))
        article_datetime = parsed_date.isoformat() if parsed_date else None
        title = (article_data.get("title") or "Untitled").strip()
        news_content = (
            article_data.get("content")
            or article_data.get("summary")
            or article_data.get("title")
            or ""
        )
        if not title or not news_content.strip() or not article_data.get("link"):
            return None

        return {
            "header": title[:200],
            "content": news_content[:5000],
            "publish_at": article_datetime,
            "crawled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_url": article_data.get("link"),
        }

    def _analyze_and_store(self, article_data: dict):
        """Analyze article content and store it in Supabase."""
        try:
            # Prepare data for analysis
            analysis_input = f"{article_data['title']}. {article_data['summary']}"
            if article_data.get('content'):
                analysis_input += f" {article_data['content'][:500]}"  # Limit content length
            
            # Call analysis API
            print(f"[crawler] Analyzing article: {article_data['title'][:50]}...")
            analysis_response = requests.post(
                f"{self.analysis_url}/analysis",
                json={"data": analysis_input},
                timeout=30
            )
            
            if analysis_response.status_code == 200:
                analysis_result = analysis_response.json()
                
                # Keep optional enrichment path for future use.
                title = analysis_result.get('title') or article_data.get('title') or "Untitled"
                news_content = article_data.get('content') or article_data.get('summary') or article_data.get('title') or ""
                
                if not title.strip() or not news_content.strip():
                    print(f"[crawler] Skipping article - missing required fields")
                    return False
                
                article_data["title"] = title
                article_data["content"] = news_content
                supabase_data = self._build_supabase_row(article_data)
                if not supabase_data:
                    return False
                
                # Store in Supabase
                stored = self._store_to_supabase(supabase_data)
                if stored:
                    print("[crawler] Stored article to Supabase (news_articles)")
                    return True
            else:
                print(f"[crawler] Analysis failed: {analysis_response.status_code}")
                print(f"[crawler] Response: {analysis_response.text[:200]}")
        except Exception as e:
            print(f"[crawler] Error analyzing/storing article: {str(e)[:200]}")
        
        return False

    def process_once(self):
        new_count = 0
        for feed in self.feeds:
            # support feeds as either a plain URL string or a dict {"url":..., "name":...}
            if isinstance(feed, str):
                feed_url = feed
                feed_name = feed
            elif isinstance(feed, dict):
                feed_url = feed.get("url")
                feed_name = feed.get("name") or feed_url
            else:
                continue
            if not self._feed_allowed(feed_url or "", feed_name or ""):
                continue

            try:
                entries = fetch_feed(feed_url, timeout=self.timeout)
            except Exception:
                entries = []

            if self.enable_sitemap_backfill and feed_url:
                try:
                    sitemap_entries = fetch_sitemap_urls(
                        source_url=feed_url,
                        start_date=self.min_publish_date,
                        end_date=datetime.datetime.now(datetime.timezone.utc),
                        max_urls=self.sitemap_max_urls_per_source,
                        timeout=self.timeout,
                        max_sitemap_files=self.sitemap_max_sitemap_files,
                    )
                    # Keep RSS first for freshness, then historical candidates.
                    for se in sitemap_entries:
                        entries.append(type("Entry", (), se))
                except Exception:
                    pass

            for e in entries:
                if e.id in self.seen:
                    continue
                if not self._is_in_target_date_range(e.published):
                    # Mark skipped entry as seen to avoid repeatedly reprocessing old/out-of-range feed items.
                    self.seen.add(e.id)
                    continue
                # Mark seen immediately to avoid duplicates
                self.seen.add(e.id)
                new_count += 1
                article_data = {
                    "id": e.id,
                    "title": e.title,
                    "link": e.link,
                    "source": feed_name,
                    "source_domain": urlparse(feed_url).hostname if feed_url else None,
                    "published": e.published,
                    "summary": e.summary,
                    "content": None,
                    "extracted_money": self._extract_money((e.title or "") + "\n" + (e.summary or "")),
                }
                if not article_data["title"] and article_data["link"]:
                    article_data["title"] = self._fetch_title_from_url(article_data["link"]) or article_data["link"]
                # Optionally fetch full article HTML and extract main content
                if self.fetch_content and e.link:
                    try:
                        content = fetch_article_content(e.link, timeout=self.timeout)
                        article_data["content"] = content
                        # also extract money from content if present
                        if content:
                            article_data["extracted_money"] = list(
                                set(article_data["extracted_money"]) | set(self._extract_money(content))
                            )
                    except Exception:
                        pass

                # Keep only BTC/ETH-related articles.
                topic_text = article_data.get("content") or article_data.get("summary") or ""
                topic_tag = self._determine_tag(article_data.get("title") or "", topic_text)
                if topic_tag == "General":
                    continue

                # Append to newline-delimited JSON file and print a short summary
                self._append_article(article_data)
                print(f"[crawler] New article: {e.title} -> {e.link}")
                if article_data["extracted_money"]:
                    print(f"[crawler] Extracted money: {article_data['extracted_money']}")
                
                # Crawl-only by default: store raw article directly to Supabase.
                if self.enable_analysis:
                    self._analyze_and_store(article_data)
                else:
                    supabase_data = self._build_supabase_row(article_data)
                    if not supabase_data:
                        print("[crawler] Skipping article - missing required fields")
                        continue
                    stored = self._store_to_supabase(supabase_data)
                    if stored:
                        print("[crawler] Stored article to Supabase (crawl-only)")

        self._save_seen()
        return new_count

    def run(self, interval_seconds: int = 60):
        print("Crawler started with feeds:", self.feeds)
        try:
            while True:
                new = self.process_once()
                print(f"[crawler] Cycle complete, {new} new articles found. Sleeping {interval_seconds}s")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("Crawler stopped by user")


def run_once():
    c = Crawler()
    return c.process_once()


if __name__ == "__main__":
    import sys

    loop = False
    interval = 300
    if "--loop" in sys.argv:
        loop = True
        idx = sys.argv.index("--loop")
        if idx + 1 < len(sys.argv):
            try:
                interval = int(sys.argv[idx + 1])
            except ValueError:
                pass

    if loop:
        c = Crawler()
        print(f"Crawler running in loop mode every {interval}s")
        try:
            c.run(interval)
        except KeyboardInterrupt:
            print("Stopped")
    else:
        print("Running single crawl cycle...")
        new = run_once()
        print(f"Done. {new} new articles found.")
