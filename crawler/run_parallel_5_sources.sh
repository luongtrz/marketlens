#!/usr/bin/env bash
set -euo pipefail

# Run 5 crawler workers in parallel (one publisher per process).
# Each worker polls the main RSS feed + an Ethereum-focused feed, with separate seen/articles files.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/crawler/logs"
PID_DIR="$ROOT_DIR/crawler/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

start_worker() {
  local slug="$1"
  local feeds_json="$2"

  local log_file="$LOG_DIR/${slug}.log"
  local pid_file="$PID_DIR/${slug}.pid"

  echo "Starting worker $slug"
  CRAWLER_FEEDS="$feeds_json" \
  CRAWLER_SEEN_FILE="crawler/data/seen_${slug}.json" \
  CRAWLER_ARTICLES_FILE="crawler/data/articles_${slug}.jsonl" \
  CRAWLER_DB_URL="sqlite+aiosqlite:///crawler_${slug}.db" \
  CRAWLER_DEDUP_BACKEND="memory" \
  CRAWLER_MIN_PUBLISH_YEAR="${CRAWLER_MIN_PUBLISH_YEAR:-2018}" \
  CRAWLER_SITEMAP_MAX_URLS_PER_SOURCE="${CRAWLER_SITEMAP_MAX_URLS_PER_SOURCE:-200000}" \
  python3 -m crawler.src.main >"$log_file" 2>&1 &

  local pid=$!
  echo "$pid" > "$pid_file"
  echo "  pid=$pid log=$log_file"
}

start_worker "coindesk" '[{"name":"CoinDesk","url":"https://www.coindesk.com/arc/outboundfeeds/rss/","category":"crypto_news"},{"name":"CoinDesk Ethereum","url":"https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tag=ethereum","category":"crypto_news"}]'

start_worker "cointelegraph" '[{"name":"CoinTelegraph","url":"https://cointelegraph.com/rss","category":"crypto_news"},{"name":"CoinTelegraph Ethereum","url":"https://cointelegraph.com/rss/tag/ethereum","category":"crypto_news"}]'

start_worker "decrypt" '[{"name":"Decrypt","url":"https://decrypt.co/feed","category":"crypto_news"},{"name":"Decrypt Ethereum","url":"https://decrypt.co/feed/ethereum","category":"crypto_news"}]'

start_worker "cryptoslate" '[{"name":"CryptoSlate","url":"https://cryptoslate.com/feed/","category":"crypto_news"},{"name":"CryptoSlate Ethereum","url":"https://cryptoslate.com/category/ethereum/feed/","category":"crypto_news"}]'

start_worker "theblock" '[{"name":"The Block","url":"https://www.theblock.co/rss.xml","category":"crypto_news"}]'

echo ""
echo "All workers started (main + Ethereum RSS where available)."
echo "Logs: $LOG_DIR"
echo "PIDs: $PID_DIR"
echo "Tail a worker log, e.g.: tail -f \"$LOG_DIR/cointelegraph.log\""
