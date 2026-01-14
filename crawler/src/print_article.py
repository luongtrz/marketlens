import json
import os
import sys
from argparse import ArgumentParser

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "articles.jsonl")

# Print article content from data/articles.jsonl, to check fetched results.

def load_articles(path):
    if not os.path.exists(path):
        print("No articles file found at:", path)
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def find_by_id(articles, id_):
    for a in articles:
        if a.get("id") == id_:
            return a
    return None


def find_by_title_substr(articles, substr):
    s = substr.lower()
    for a in articles:
        if a.get("title") and s in a.get("title").lower():
            return a
    return None


def main(argv=None):
    p = ArgumentParser(description="Print article content from data/articles.jsonl")
    p.add_argument("--id", help="Article id to print")
    p.add_argument("--index", type=int, help="Zero-based index (0 = first) to print")
    p.add_argument("--title-substr", help="Substring of title to match")
    args = p.parse_args(argv)

    articles = load_articles(DATA_FILE)
    if not articles:
        return 1

    article = None
    if args.id:
        article = find_by_id(articles, args.id)
    elif args.index is not None:
        idx = args.index
        if idx < 0:
            idx = len(articles) + idx
        if 0 <= idx < len(articles):
            article = articles[idx]
    elif args.title_substr:
        article = find_by_title_substr(articles, args.title_substr)
    else:
        # default: most recent (last appended)
        article = articles[-1]

    if not article:
        print("Article not found")
        return 2

    print("--- Article ---")
    print("ID:", article.get("id"))
    print("Title:", article.get("title"))
    print("Link:", article.get("link"))
    print("Published:", article.get("published"))
    print()
    content = article.get("content") or article.get("summary") or "(no content)"
    print(content)
    print("--- End ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
