/**
 * Seed script to fetch 1-hour and 1-day candle data from Binance API.
 * This runs separately from the 1-minute seed.
 * 
 * Usage: 
 *   npm run seed:aggregated           # Seed 1h and 1d for all coins
 *   npm run seed:aggregated BTC       # Seed specific coin
 */

import { DataSource } from 'typeorm';
import { MarketCandle } from '../crypto/entities/market-candle.entity';
import * as dotenv from 'dotenv';

dotenv.config();

const BINANCE_API = 'https://api.binance.com/api/v3/klines';

const COINS: Record<string, string> = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
};

// Timeframes to seed: interval -> { binanceInterval, startTimestamp }
const TIMEFRAMES = {
    '1h': { binanceInterval: '1h', limit: 1000 },
    '1d': { binanceInterval: '1d', limit: 1000 },
};

const START_TIMESTAMP = 1502928000000; // Aug 17, 2017
const BATCH_SIZE = 1000;
const DELAY_MS = 100;

async function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchKlines(binanceSymbol: string, interval: string, startTime: number, limit: number): Promise<any[]> {
    const url = `${BINANCE_API}?symbol=${binanceSymbol}&interval=${interval}&limit=${limit}&startTime=${startTime}`;
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Binance API error: ${response.statusText}`);
    }
    return response.json();
}

function parseKline(symbol: string, resolution: string, kline: any[]): Partial<MarketCandle> {
    return {
        symbol,
        resolution,
        timestamp: kline[0],
        open: parseFloat(kline[1]),
        high: parseFloat(kline[2]),
        low: parseFloat(kline[3]),
        close: parseFloat(kline[4]),
        volume: parseFloat(kline[5]),
    };
}

async function seedTimeframe(
    repository: any,
    symbol: string,
    binanceSymbol: string,
    resolution: string,
    binanceInterval: string,
    limit: number
): Promise<number> {
    console.log(`\n--- Seeding ${symbol} ${resolution} ---`);

    const existingCount = await repository.count({ where: { symbol, resolution } });
    console.log(`Existing ${resolution} candles for ${symbol}: ${existingCount}`);

    let startTime = START_TIMESTAMP;
    if (existingCount > 0) {
        const latest = await repository
            .createQueryBuilder('candle')
            .where('candle.symbol = :symbol', { symbol })
            .andWhere('candle.resolution = :resolution', { resolution })
            .orderBy('candle.timestamp', 'DESC')
            .getOne();
        if (latest) {
            const interval = resolution === '1h' ? 3600000 : 86400000;
            startTime = Number(latest.timestamp) + interval;
            console.log(`Resuming from: ${new Date(startTime).toISOString()}`);
        }
    }

    const now = Date.now();
    let currentTime = startTime;
    let totalInserted = 0;
    let batch: Partial<MarketCandle>[] = [];
    let requestCount = 0;
    const startedAt = Date.now();

    while (currentTime < now) {
        try {
            const klines = await fetchKlines(binanceSymbol, binanceInterval, currentTime, limit);
            requestCount++;

            if (klines.length === 0) break;

            for (const kline of klines) {
                batch.push(parseKline(symbol, resolution, kline));
            }

            const interval = resolution === '1h' ? 3600000 : 86400000;
            currentTime = klines[klines.length - 1][0] + interval;

            if (requestCount % 5 === 0) {
                const progress = ((currentTime - START_TIMESTAMP) / (now - START_TIMESTAMP) * 100);
                const elapsed = (Date.now() - startedAt) / 1000;
                const eta = progress > 0 ? ((100 - progress) / progress) * elapsed : 0;
                process.stdout.write(`\r[${symbol}/${resolution}] ${progress.toFixed(2)}% | Candles: ${totalInserted + batch.length} | ETA: ${Math.floor(eta)}s    `);
            }

            if (batch.length >= BATCH_SIZE) {
                await repository.upsert(batch, ['symbol', 'resolution', 'timestamp']);
                totalInserted += batch.length;
                console.log(`\n[${symbol}/${resolution}] Saved ${totalInserted} candles`);
                batch = [];
            }

            await sleep(DELAY_MS);
        } catch (error) {
            console.error(`\nError:`, error);
            await sleep(5000);
        }
    }

    if (batch.length > 0) {
        await repository.upsert(batch, ['symbol', 'resolution', 'timestamp']);
        totalInserted += batch.length;
    }

    console.log(`\n[${symbol}/${resolution}] Complete! Total: ${totalInserted}`);
    return totalInserted;
}

async function main() {
    console.log('=== MarketLens Aggregated Data Seeding (1h, 1d) ===');

    const args = process.argv.slice(2);
    const coinsToSeed = args.length > 0
        ? args.filter(arg => COINS[arg.toUpperCase()]).map(arg => arg.toUpperCase())
        : Object.keys(COINS);

    if (coinsToSeed.length === 0) {
        console.error('No valid coins. Available:', Object.keys(COINS).join(', '));
        process.exit(1);
    }

    console.log(`Coins: ${coinsToSeed.join(', ')}`);
    console.log(`Timeframes: 1h, 1d`);

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

        for (const [resolution, config] of Object.entries(TIMEFRAMES)) {
            const inserted = await seedTimeframe(
                repository, symbol, binanceSymbol,
                resolution, config.binanceInterval, config.limit
            );
            grandTotal += inserted;
        }
    }

    console.log(`\n=== Complete! Total: ${grandTotal} candles ===`);

    await dataSource.destroy();
}

main().catch(console.error);
