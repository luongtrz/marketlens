#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT_DIR/crawler/pids"

if [[ ! -d "$PID_DIR" ]]; then
  echo "No pid directory found: $PID_DIR"
  exit 0
fi

for pid_file in "$PID_DIR"/*.pid; do
  [[ -e "$pid_file" ]] || continue
  slug="$(basename "$pid_file" .pid)"
  pid="$(cat "$pid_file" || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $slug (pid=$pid)"
    kill "$pid" || true
  else
    echo "Skipping $slug (not running)"
  fi
  rm -f "$pid_file"
done

echo "Done."
