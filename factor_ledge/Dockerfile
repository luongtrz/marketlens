FROM python:3.12-slim

# Install Node.js 20
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY shared /app/shared
COPY factor_ledge /app/factor_ledge

# Install Python deps
RUN pip install --no-cache-dir -r /app/factor_ledge/requirements.txt

# Build TypeScript classify service (ESM → CJS bundle, .cjs ext bypasses "type":"module")
WORKDIR /app/factor_ledge/src/classify/src
RUN npm install && \
    npx esbuild index.ts --bundle --platform=node --format=cjs --outfile=index.cjs \
      --external:node-fetch 2>&1 | tail -3

# Build TypeScript ledger service
WORKDIR /app/factor_ledge/src/ledger
RUN npm install && \
    npx esbuild index.ts --bundle --platform=node --format=cjs --outfile=index.cjs 2>&1 | tail -3

# Build TypeScript query service
WORKDIR /app/factor_ledge/src/query
RUN npm install && \
    npx esbuild index.ts --bundle --platform=node --format=cjs --outfile=index.cjs 2>&1 | tail -3

WORKDIR /app

ENV PYTHONPATH=/app
ENV CLASSIFY_PORT=3001
ENV LEDGER_PORT=3002
ENV QUERY_PORT=3003
ENV CLASSIFY_URL=http://localhost:3001
ENV LEDGER_URL=http://localhost:3002
ENV QUERY_URL=http://localhost:3003

# Startup script: launch TS services then Python gateway
COPY factor_ledge/start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
