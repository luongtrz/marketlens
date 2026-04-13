#!/usr/bin/env python3
"""Create a cleaned newline-delimited JSON file with only title and content.

Usage: python src/clean_articles.py
Writes output to `data/articles_clean.jsonl`.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "articles.jsonl"
OUTPUT = ROOT / "data" / "articles_clean.jsonl"


def clean_obj(obj: dict) -> dict:
    title = obj.get("title") or obj.get("headline") or ""
    content = obj.get("content") or obj.get("summary") or ""
    return {
        "id": obj.get("id") or obj.get("link") or None,
        "title": title.strip(),
        "content": content.strip(),
        "source": obj.get("source_domain") or obj.get("source")
    }


def parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        # attempt to salvage a JSON object within the line
        s = line.find("{")
        e = line.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(line[s : e + 1])
            except Exception:
                return None
        return None


def main() -> int:
    if not INPUT.exists():
        print(f"Input not found: {INPUT}", file=sys.stderr)
        return 2

    written = 0
    with INPUT.open("r", encoding="utf-8", errors="ignore") as inf, OUTPUT.open("w", encoding="utf-8") as outf:
        for line in inf:
            obj = parse_line(line)
            if not obj or not isinstance(obj, dict):
                continue
            cleaned = clean_obj(obj)
            outf.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} cleaned articles to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
