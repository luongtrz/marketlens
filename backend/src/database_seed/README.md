# Database Seeding

Script to fetch historical cryptocurrency data from Binance API and store in PostgreSQL.

## Prerequisites

1. **PostgreSQL running** (via Docker Compose):
   ```bash
   # From project root
   docker-compose up -d
   ```

2. **Environment variables** configured in `backend/.env`:
   ```env
   DB_HOST=localhost
   DB_PORT=5433
   DB_USER=postgres
   DB_PASSWORD=postgres
   DB_NAME=marketlens
   ```

## Usage

```bash
# Navigate to backend folder
cd backend

# Seed 1-minute data (all coins)
npm run seed

# Seed specific coin only
npm run seed BTC
npm run seed ETH

# Seed 1-hour and 1-day data (can run in parallel with 1m)
npm run seed:aggregated

# Seed aggregated for specific coin
npm run seed:aggregated BTC
```

## Supported Coins

| Symbol | Binance Pair | Data Available From |
|--------|--------------|---------------------|
| BTC    | BTCUSDT      | Aug 17, 2017        |
| ETH    | ETHUSDT      | Aug 17, 2017        |

## Features

- **Resume Support**: If interrupted, running again will continue from where it stopped
- **Progress Display**: Shows percentage, request count, candle count, and ETA
- **Batch Insert**: Inserts 5000 candles at a time for efficiency
- **Rate Limiting**: 100ms delay between requests (safe for Binance API)

## Data Details

- **Resolution**: 1-minute candles
- **Fields**: timestamp, open, high, low, close, volume
- **Estimated Time**: ~1-2 hours per coin for full history
- **Data Size**: ~4 million candles per coin (~8 years)

## Database Schema

Table: `market_candles`

| Column     | Type    | Description              |
|------------|---------|--------------------------|
| symbol     | VARCHAR | Coin symbol (BTC, ETH)   |
| resolution | VARCHAR | Candle interval (1m)     |
| timestamp  | BIGINT  | Unix timestamp (ms)      |
| open       | DECIMAL | Opening price            |
| high       | DECIMAL | Highest price            |
| low        | DECIMAL | Lowest price             |
| close      | DECIMAL | Closing price            |
| volume     | DECIMAL | Trading volume           |

## Troubleshooting

### Connection refused
Make sure PostgreSQL is running:
```bash
docker ps | grep marketlens-db
```

### Slow seeding
This is normal. Fetching 8 years of minute data takes time. The script shows ETA.

### Want to start fresh
```bash
# Connect to database and truncate
docker exec marketlens-db psql -U postgres -d marketlens -c "TRUNCATE market_candles;"
```

## Adding New Coins

Edit `seed.ts` and add to the `COINS` object:
```typescript
const COINS: Record<string, string> = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',  // Add new coin here
};
```
