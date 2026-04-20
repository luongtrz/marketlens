#!/usr/bin/env bash
set -euo pipefail

# Run 5 crawler workers in parallel (one source per process).
# Each worker uses separate seen/articles files to avoid overlap.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/crawler/logs"
PID_DIR="$ROOT_DIR/crawler/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

start_worker() {
  local slug="$1"
  local name="$2"
  local url="$3"

  local log_file="$LOG_DIR/${slug}.log"
  local pid_file="$PID_DIR/${slug}.pid"

  echo "Starting $name -> $url"
  CRAWLER_FEEDS="[{\"name\":\"$name\",\"url\":\"$url\",\"category\":\"crypto_news\"}]" \
  CRAWLER_SEEN_FILE="crawler/data/seen_${slug}.json" \
  CRAWLER_ARTICLES_FILE="crawler/data/articles_${slug}.jsonl" \
  CRAWLER_DB_URL="sqlite+aiosqlite:///crawler_${slug}.db" \
  CRAWLER_DEDUP_BACKEND="memory" \
  python3 -m crawler.src.main >"$log_file" 2>&1 &

  local pid=$!
  echo "$pid" > "$pid_file"
  echo "  pid=$pid log=$log_file"
}

start_worker "coindesk" "CoinDesk" "https://www.coindesk.com/arc/outboundfeeds/rss/"
start_worker "cointelegraph" "CoinTelegraph" "https://cointelegraph.com/rss"
start_worker "decrypt" "Decrypt" "https://decrypt.co/feed"
start_worker "cryptoslate" "CryptoSlate" "https://cryptoslate.com/feed/"
start_worker "theblock" "The Block" "https://www.theblock.co/rss.xml"

echo ""
echo "All workers started."
echo "Logs: $LOG_DIR"
echo "PIDs: $PID_DIR"
echo "Tail a worker log, e.g.: tail -f \"$LOG_DIR/coindesk.log\""
