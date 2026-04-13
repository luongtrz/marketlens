import json
import os
import re
import time
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
    from .fetch_rss import fetch_feed, fetch_article_content
except Exception:
    from src.config import DEFAULT_FEEDS, DATA_DIR, SEEN_FILE, ARTICLES_FILE, FETCH_TIMEOUT, SOURCES
    from src.fetch_rss import fetch_feed, fetch_article_content


class Crawler:
    def __init__(self, feeds=None, seen_file=SEEN_FILE, articles_file=ARTICLES_FILE, timeout=FETCH_TIMEOUT, fetch_content: bool = True, backend_url: str = None, analysis_url: str = None):
        # `feeds` may be a list of strings or dicts with a `url` key.
        self.feeds = feeds or SOURCES
        self.seen_file = seen_file
        self.articles_file = articles_file
        self.timeout = timeout
        self.fetch_content = fetch_content
        self.backend_url = backend_url or os.getenv("BACKEND_URL", "http://localhost:8080")
        self.analysis_url = analysis_url or os.getenv("ANALYSIS_URL", "http://localhost:8000")
        os.makedirs(DATA_DIR, exist_ok=True)
        self.seen: Set[str] = self._load_seen()

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

    def _analyze_and_store(self, article_data: dict):
        """Analyze article content and store it in backend database"""
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
                
                # Parse published date to ISO format
                article_datetime = None
                if article_data.get('published'):
                    try:
                        from dateutil import parser as date_parser
                        parsed_date = date_parser.parse(article_data['published'])
                        article_datetime = parsed_date.isoformat()
                    except Exception:
                        article_datetime = None
                
                # Prepare data for backend storage - ensure required fields are not empty
                title = analysis_result.get('title') or article_data.get('title') or "Untitled"
                news_content = article_data.get('content') or article_data.get('summary') or article_data.get('title') or ""
                
                if not title.strip() or not news_content.strip():
                    print(f"[crawler] Skipping article - missing required fields")
                    return False
                
                # Determine tag
                tag = self._determine_tag(title, news_content)

                backend_data = {
                    "articleDateTime": article_datetime,
                    "title": title[:200],  # Limit title length
                    "url": article_data.get('link'),
                    "news": news_content[:5000],  # Limit content length
                    "summarizeNews": analysis_result.get('summarize_news', article_data.get('summary', ''))[:1000],
                    "tag": tag,
                    # New fields
                    "sentiment": analysis_result.get('sentiment', 'Neutral'),
                    "impactScore": int(analysis_result.get('impact_score', 0)),
                    "keyTakeaways": analysis_result.get('key_takeaways', []),
                    "detailedSummary": analysis_result.get('detailed_summary', ''),
                    "snippet": analysis_result.get('short_summary') or analysis_result.get('summarize_news', '')[:200],
                    "source": article_data.get('source', 'unknown')
                }
                
                # Store in backend database
                backend_response = requests.post(
                    f"{self.backend_url}/api/news",
                    json=backend_data,
                    timeout=10
                )
                
                if backend_response.status_code == 201:
                    print(f"[crawler] Stored article ({tag}) with sentiment: {backend_data['sentiment']}")
                    return True
                elif backend_response.status_code == 500:
                    # Check if it's a duplicate key error
                    response_text = backend_response.text.lower()
                    if 'duplicate' in response_text or 'e11000' in response_text:
                        print(f"[crawler] Article already exists in database (duplicate URL)")
                        return True  # Consider this a success since article is already stored
                    else:
                        print(f"[crawler] Backend storage failed: {backend_response.status_code}")
                        print(f"[crawler] Response: {backend_response.text[:200]}")
                else:
                    print(f"[crawler] Backend storage failed: {backend_response.status_code}")
                    print(f"[crawler] Response: {backend_response.text[:200]}")
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

            try:
                entries = fetch_feed(feed_url, timeout=self.timeout)
            except Exception:
                entries = []

            for e in entries:
                if e.id in self.seen:
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

                # Append to newline-delimited JSON file and print a short summary
                self._append_article(article_data)
                print(f"[crawler] New article: {e.title} -> {e.link}")
                if article_data["extracted_money"]:
                    print(f"[crawler] Extracted money: {article_data['extracted_money']}")
                
                # Analyze and store in backend database
                self._analyze_and_store(article_data)

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
