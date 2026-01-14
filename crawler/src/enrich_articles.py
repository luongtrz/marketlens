import json
import os
import sys
from pathlib import Path

# Ensure repository root is on path so we can import package modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fetch_rss import fetch_article_content
from src.crawler import Crawler


DATA_FILE = os.path.join(ROOT, "data", "articles.jsonl")
TMP_FILE = DATA_FILE + ".tmp"


def enrich_all():
    if not os.path.exists(DATA_FILE):
        print("No articles file found at", DATA_FILE)
        return

    c = Crawler(fetch_content=False)  # reuse helper methods

    total = 0
    updated = 0
    with open(DATA_FILE, "r", encoding="utf-8") as inf, open(TMP_FILE, "w", encoding="utf-8") as outf:
        for line in inf:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            content = obj.get("content")
            if not content and obj.get("link"):
                try:
                    content = fetch_article_content(obj["link"], timeout=10)
                    if content:
                        obj["content"] = content
                        # update extracted money
                        obj["extracted_money"] = list(set(obj.get("extracted_money", []) + c._extract_money(content)))
                        # apply any per-source enrichment rules (e.g., language hints, title normalization)
                        apply_source_rules(obj, c)
                        updated += 1
                        print(f"Enriched: {obj.get('title')}")
                except Exception as e:
                    print("Failed to fetch content for", obj.get("link"), str(e))

            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(TMP_FILE, DATA_FILE)
    print(f"Done. Processed {total} articles, enriched {updated} entries.")


if __name__ == "__main__":
    enrich_all()


def apply_source_rules(obj: dict, crawler: Crawler = None):
    """Apply lightweight, extensible per-source rules to an article object.

    This is a starter implementation — add rules for specific domains as
    needed. Rules may modify title, set a `language` hint, or extract more
    structured metadata.
    """
    src = obj.get("source") or ""
    domain = obj.get("source_domain") or ""

    def rule_coindesk(o):
        t = o.get("title") or ""
        if t.lower().startswith("coindesk"):
            # remove leading site name if present
            parts = t.split("-", 1)
            if len(parts) == 2:
                o["title"] = parts[1].strip()

    def rule_vietnamese(o):
        # set language hint for Vietnamese sites
        o["language"] = "vi"

    # Map domain substrings to rule functions
    rules = [
        ("coindesk.com", rule_coindesk),
        ("cointelegraph.com", lambda o: None),
        ("cryptopanic.com", lambda o: None),
        ("decrypt.co", lambda o: None),
        ("vnexpress.net", rule_vietnamese),
        ("thanhnien.vn", rule_vietnamese),
    ]

    for key, fn in rules:
        if key in (domain or src or ""):
            try:
                fn(obj)
            except Exception:
                pass

    # As an example, if we made changes to content that may affect money
    # extraction, re-run extraction.
    if crawler and obj.get("content"):
        obj["extracted_money"] = list(set(obj.get("extracted_money", []) + crawler._extract_money(obj.get("content"))))
