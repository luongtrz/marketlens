#!/bin/bash
set -e

echo "[start.sh] Starting factor_ledge services..."

# Start TypeScript classify service
if [ -f /app/factor_ledge/src/classify/src/index.cjs ]; then
    node /app/factor_ledge/src/classify/src/index.cjs &
    echo "[start.sh] classify-service started (port ${CLASSIFY_PORT:-3001})"
else
    # Fall back to ts-node if esbuild didn't produce output
    cd /app/factor_ledge/src/classify/src && npx ts-node index.ts &
    echo "[start.sh] classify-service started via ts-node (port ${CLASSIFY_PORT:-3001})"
fi

# Start TypeScript ledger service
if [ -f /app/factor_ledge/src/ledger/index.cjs ]; then
    node /app/factor_ledge/src/ledger/index.cjs &
    echo "[start.sh] ledger-service started (port ${LEDGER_PORT:-3002})"
fi

# Start TypeScript query service
if [ -f /app/factor_ledge/src/query/index.cjs ]; then
    node /app/factor_ledge/src/query/index.cjs &
    echo "[start.sh] query-service started (port ${QUERY_PORT:-3003})"
fi

# Wait briefly for TS services to initialise
sleep 2

# Start Python gateway (foreground)
echo "[start.sh] Starting Python gateway (port 8004)..."
exec uvicorn factor_ledge.src.api:app --host 0.0.0.0 --port 8004
