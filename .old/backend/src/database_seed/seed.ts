/**
 * Seed script to fetch historical 1-minute candle data from Binance API
 * and store it in the PostgreSQL database.
 * 
 * Usage: 
 *   npm run seed           # Seed all coins (BTC, ETH)
 *   npm run seed BTC       # Seed specific coin
 *   npm run seed BTC ETH   # Seed multiple specific coins
 * 
 * This script will:
 * 1. Connect to the database
 * 2. Fetch 1-minute data from Binance (Aug 2017 to now)
 * 3. Insert data in batches to avoid memory issues
 * 4. Support resume if interrupted
 */

import { DataSource } from 'typeorm';
import { MarketCandle } from '../crypto/entities/market-candle.entity';
import * as dotenv from 'dotenv';

dotenv.config();

const BINANCE_API = 'https://api.binance.com/api/v3/klines';

// Supported coins: symbol -> Binance trading pair
const COINS: Record<string, string> = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
};

const INTERVAL = '1m';
const LIMIT = 1000; // Max per request
const START_TIMESTAMP = 1502928000000; // Aug 17, 2017 (Binance launch)
const BATCH_SIZE = 5000; // Insert batch size

// Rate limiting: Binance allows 1200 requests per minute
const DELAY_MS = 100; // 100ms between requests = 600 req/min (safe margin)

async function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchKlines(binanceSymbol: string, startTime: number): Promise<any[]> {
    const url = `${BINANCE_API}?symbol=${binanceSymbol}&interval=${INTERVAL}&limit=${LIMIT}&startTime=${startTime}`;
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Binance API error: ${response.statusText}`);
    }
    return response.json();
}

function parseKline(symbol: string, kline: any[]): Partial<MarketCandle> {
    return {
        symbol: symbol,
        resolution: '1m',
        timestamp: kline[0],
        open: parseFloat(kline[1]),
        high: parseFloat(kline[2]),
        low: parseFloat(kline[3]),
        close: parseFloat(kline[4]),
        volume: parseFloat(kline[5]),
    };
}

async function seedCoin(
    repository: any,
    symbol: string,
    binanceSymbol: string
): Promise<number> {
    console.log(`\n--- Seeding ${symbol} (${binanceSymbol}) ---`);

    // Check existing data
    const existingCount = await repository.count({ where: { symbol, resolution: '1m' } });
    console.log(`Existing 1m candles for ${symbol}: ${existingCount}`);

    // Find the latest timestamp in DB to resume from
    let startTime = START_TIMESTAMP;
    if (existingCount > 0) {
        const latest = await repository
            .createQueryBuilder('candle')
            .where('candle.symbol = :symbol', { symbol })
            .andWhere('candle.resolution = :resolution', { resolution: '1m' })
            .orderBy('candle.timestamp', 'DESC')
            .getOne();
        if (latest) {
            startTime = Number(latest.timestamp) + 60000; // Next minute
            console.log(`Resuming from: ${new Date(startTime).toISOString()}`);
        }
    }

    const now = Date.now();
    let currentTime = startTime;
    let totalInserted = 0;
    let batch: Partial<MarketCandle>[] = [];
    let requestCount = 0;
    const startedAt = Date.now();

    console.log(`Fetching from ${new Date(startTime).toISOString()} to now...`);

    while (currentTime < now) {
        try {
            const klines = await fetchKlines(binanceSymbol, currentTime);
            requestCount++;

            if (klines.length === 0) {
                console.log('No more data available.');
                break;
            }

            for (const kline of klines) {
                batch.push(parseKline(symbol, kline));
            }

            // Update current time to last candle + 1 minute
            currentTime = klines[klines.length - 1][0] + 60000;

            // Log progress every 10 requests
            if (requestCount % 10 === 0) {
                const progress = ((currentTime - START_TIMESTAMP) / (now - START_TIMESTAMP) * 100);
                const elapsed = (Date.now() - startedAt) / 1000;
                const eta = progress > 0 ? ((100 - progress) / progress) * elapsed : 0;
                const etaMin = Math.floor(eta / 60);
                const etaSec = Math.floor(eta % 60);

                process.stdout.write(`\r[${symbol}] ${progress.toFixed(2)}% | Requests: ${requestCount} | Candles: ${totalInserted + batch.length} | ETA: ${etaMin}m ${etaSec}s    `);
            }

            // Insert batch when full
            if (batch.length >= BATCH_SIZE) {
                await repository.upsert(batch, ['symbol', 'resolution', 'timestamp']);
                totalInserted += batch.length;
                const progress = ((currentTime - START_TIMESTAMP) / (now - START_TIMESTAMP) * 100).toFixed(2);
                console.log(`\n[${symbol}] ${progress}% | Saved ${totalInserted} candles | Date: ${new Date(currentTime).toISOString().split('T')[0]}`);
                batch = [];
            }

            await sleep(DELAY_MS);
        } catch (error) {
            console.error(`\nError at ${new Date(currentTime).toISOString()}:`, error);
            console.log('Retrying in 5 seconds...');
            await sleep(5000);
        }
    }

    // Insert remaining batch
    if (batch.length > 0) {
        await repository.upsert(batch, ['symbol', 'resolution', 'timestamp']);
        totalInserted += batch.length;
    }

    console.log(`\n[${symbol}] Complete! Total inserted: ${totalInserted}`);
    return totalInserted;
}

async function main() {
    console.log('=== MarketLens Data Seeding ===');

    // Parse command line args for specific coins
    const args = process.argv.slice(2);
    const coinsToSeed = args.length > 0
        ? args.filter(arg => COINS[arg.toUpperCase()]).map(arg => arg.toUpperCase())
        : Object.keys(COINS);

    if (coinsToSeed.length === 0) {
        console.error('No valid coins specified. Available coins:', Object.keys(COINS).join(', '));
        process.exit(1);
    }

    console.log(`Coins to seed: ${coinsToSeed.join(', ')}`);
    console.log('Connecting to database...');

    const dataSource = new DataSource({
        type: 'postgres',
        host: process.env.DB_HOST || 'localhost',
        port: parseInt(process.env.DB_PORT || '5433'),
        username: process.env.DB_USER || 'postgres',
        password: process.env.DB_PASSWORD || 'postgres',
        database: process.env.DB_NAME || 'marketlens',
        entities: [MarketCandle],
        synchronize: true,
    });

    await dataSource.initialize();
    console.log('Database connected!');

    const repository = dataSource.getRepository(MarketCandle);
    let grandTotal = 0;

    for (const symbol of coinsToSeed) {
        const binanceSymbol = COINS[symbol];
        const inserted = await seedCoin(repository, symbol, binanceSymbol);
        grandTotal += inserted;
    }

    console.log(`\n=== Seeding Complete ===`);
    console.log(`Grand total candles inserted: ${grandTotal}`);

    // Show final counts
    for (const symbol of coinsToSeed) {
        const count = await repository.count({ where: { symbol, resolution: '1m' } });
        console.log(`${symbol}: ${count} candles`);
    }

    await dataSource.destroy();
    console.log('Database connection closed.');
}

main().catch(console.error);
