/**
 * Seed script to fetch historical 1-minute candle data from Binance API
 * and store it in the PostgreSQL database.
 * 
 * Usage: npm run seed
 * 
 * This script will:
 * 1. Connect to the database
 * 2. Fetch 1-minute BTCUSDT data from Binance (Aug 2017 to now)
 * 3. Insert data in batches to avoid memory issues
 */

import { DataSource } from 'typeorm';
import { MarketCandle } from '../entities/market-candle.entity';
import * as dotenv from 'dotenv';

dotenv.config();

const BINANCE_API = 'https://api.binance.com/api/v3/klines';
const SYMBOL = 'BTCUSDT';
const INTERVAL = '1m';
const LIMIT = 1000; // Max per request
const START_TIMESTAMP = 1502928000000; // Aug 17, 2017 (Binance launch)
const BATCH_SIZE = 5000; // Insert batch size

// Rate limiting: Binance allows 1200 requests per minute
const DELAY_MS = 100; // 100ms between requests = 600 req/min (safe margin)

async function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchKlines(startTime: number): Promise<any[]> {
    const url = `${BINANCE_API}?symbol=${SYMBOL}&interval=${INTERVAL}&limit=${LIMIT}&startTime=${startTime}`;
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Binance API error: ${response.statusText}`);
    }
    return response.json();
}

function parseKline(kline: any[]): Partial<MarketCandle> {
    return {
        symbol: 'BTC',
        resolution: '1m',
        timestamp: kline[0],
        open: parseFloat(kline[1]),
        high: parseFloat(kline[2]),
        low: parseFloat(kline[3]),
        close: parseFloat(kline[4]),
        volume: parseFloat(kline[5]),
    };
}

async function main() {
    console.log('=== MarketLens Data Seeding ===');
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

    // Check existing data
    const existingCount = await repository.count({ where: { symbol: 'BTC', resolution: '1m' } });
    console.log(`Existing 1m candles in DB: ${existingCount}`);

    // Find the latest timestamp in DB to resume from
    let startTime = START_TIMESTAMP;
    if (existingCount > 0) {
        const latest = await repository
            .createQueryBuilder('candle')
            .where('candle.symbol = :symbol', { symbol: 'BTC' })
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

    console.log(`Fetching data from ${new Date(startTime).toISOString()} to now...`);
    console.log('This may take a while (1-2 hours for full history)...\n');

    while (currentTime < now) {
        try {
            const klines = await fetchKlines(currentTime);

            if (klines.length === 0) {
                console.log('No more data available.');
                break;
            }

            for (const kline of klines) {
                batch.push(parseKline(kline));
            }

            // Update current time to last candle + 1 minute
            currentTime = klines[klines.length - 1][0] + 60000;

            // Insert batch when full
            if (batch.length >= BATCH_SIZE) {
                await repository.upsert(batch, ['symbol', 'resolution', 'timestamp']);
                totalInserted += batch.length;
                const progress = ((currentTime - START_TIMESTAMP) / (now - START_TIMESTAMP) * 100).toFixed(2);
                console.log(`[${progress}%] Inserted ${totalInserted} candles. Current: ${new Date(currentTime).toISOString()}`);
                batch = [];
            }

            await sleep(DELAY_MS);
        } catch (error) {
            console.error(`Error at ${new Date(currentTime).toISOString()}:`, error);
            console.log('Retrying in 5 seconds...');
            await sleep(5000);
        }
    }

    // Insert remaining batch
    if (batch.length > 0) {
        await repository.upsert(batch, ['symbol', 'resolution', 'timestamp']);
        totalInserted += batch.length;
    }

    console.log(`\n=== Seeding Complete ===`);
    console.log(`Total candles inserted: ${totalInserted}`);

    const finalCount = await repository.count({ where: { symbol: 'BTC', resolution: '1m' } });
    console.log(`Total 1m candles in DB: ${finalCount}`);

    await dataSource.destroy();
    console.log('Database connection closed.');
}

main().catch(console.error);
