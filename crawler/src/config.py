import os

DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/", # CoinDesk
    "https://cointelegraph.com/rss",                   # CoinTelegraph
    "https://cryptopanic.com/news/rss/",               # CryptoPanic (Tổng hợp tin)
    "https://decrypt.co/feed",                         # Decrypt
    "https://vnexpress.net/rss/so-hoa.rss",            # VnExpress - Nhịp sống số (Tiếng Việt)
    "https://thanhnien.vn/rss/cong-nghe-game.rss",     # Thanh Niên - Công nghệ
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEEN_FILE = os.path.join(DATA_DIR, "seen.json")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")

FETCH_TIMEOUT = 10

# Backwards-compatible alias. `SOURCES` may contain either string URLs or
# dicts like {"url": "https://...", "name": "Site name"} to allow
# future per-source options.
SOURCES = DEFAULT_FEEDS
