import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, MoreThanOrEqual, LessThanOrEqual, Between } from 'typeorm';
import { MarketCandle } from './entities/market-candle.entity';

export interface CoinData {
    symbol: string;
    name: string;
    price: number;
    change24h: number;
    volume: string;
    marketCap: string;
}

export interface HistoryPoint {
    ts: number;
    price: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

@Injectable()
export class CryptoService {
    private readonly apiKey: string;
    private readonly baseUrl = 'https://min-api.cryptocompare.com/data';

    constructor(
        private configService: ConfigService,
        @InjectRepository(MarketCandle)
        private candleRepository: Repository<MarketCandle>,
    ) {
        const apiKey = this.configService.get<string>('CRYPTOCOMPARE_API_KEY');
        if (!apiKey || apiKey.trim() === '') {
            throw new Error(
                'CRYPTOCOMPARE_API_KEY is not configured. Please set this environment variable to a valid API key.',
            );
        }
        this.apiKey = apiKey;
    }

    private formatVolume(num: number): string {
        if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
        return num.toLocaleString();
    }

    async getTopCoins(): Promise<CoinData[]> {
        const url = `${this.baseUrl}/top/mktcapfull?limit=10&tsym=USD`;

        try {
            const response = await fetch(url, {
                headers: { authorization: `Apikey ${this.apiKey}` },
            });

            if (!response.ok) {
                throw new Error(`CryptoCompare API Error: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.Response === 'Error') {
                console.error('CryptoCompare Error:', data.Message);
                return [];
            }

            return data.Data.map((item: any) => {
                const coin = item.CoinInfo;
                const raw = item.RAW ? item.RAW.USD : {};

                return {
                    symbol: coin.Name,
                    name: coin.FullName,
                    price: raw.PRICE || 0,
                    change24h: raw.CHANGE24HOUR || 0,
                    volume: this.formatVolume(raw.VOLUME24HOUR || 0),
                    marketCap: this.formatVolume(raw.MKTCAP || 0),
                };
            });
        } catch (error) {
            console.error('Failed to fetch market data:', error);
            return [];
        }
    }

    /**
     * Get historical data from local PostgreSQL database.
     * Falls back to CryptoCompare API if data not found in DB.
     */
    async getHistoricalData(
        symbol: string,
        limit: number = 144,
        aggregate: number = 1,
        type: 'minute' | 'hour' | 'day' = 'minute',
        toTs?: number,
    ): Promise<HistoryPoint[]> {
        // Map type to resolution
        const resolutionMap: Record<string, string> = {
            'minute': '1m',
            'hour': '1h',
            'day': '1d',
        };
        const resolution = resolutionMap[type];

        // Calculate time range
        const endTime = toTs ? toTs * 1000 : Date.now();
        const intervalMs = type === 'minute' ? 60000 : type === 'hour' ? 3600000 : 86400000;
        const startTime = endTime - (limit * aggregate * intervalMs);

        try {
            // Query from local database
            const candles = await this.candleRepository.find({
                where: {
                    symbol: symbol.toUpperCase(),
                    resolution,
                    timestamp: Between(startTime, endTime),
                },
                order: { timestamp: 'ASC' },
                take: limit,
            });

            if (candles.length > 0) {
                console.log(`[DB] Found ${candles.length} ${resolution} candles for ${symbol}`);

                // Aggregate if needed (e.g., 5m from 1m)
                if (aggregate > 1 && resolution === '1m') {
                    return this.aggregateCandles(candles, aggregate);
                }

                return candles.map(c => ({
                    ts: Number(c.timestamp),
                    price: Number(c.close),
                    open: Number(c.open),
                    high: Number(c.high),
                    low: Number(c.low),
                    close: Number(c.close),
                    volume: Number(c.volume),
                }));
            }

            // Fallback to CryptoCompare API if no data in DB
            console.log(`[API Fallback] No DB data for ${symbol} ${resolution}, using CryptoCompare`);
            return this.fetchFromCryptoCompare(symbol, limit, aggregate, type, toTs);

        } catch (error) {
            console.error('DB query failed, falling back to API:', error);
            return this.fetchFromCryptoCompare(symbol, limit, aggregate, type, toTs);
        }
    }

    /**
     * Aggregate 1-minute candles into larger timeframes (e.g., 5m, 15m, 30m)
     */
    private aggregateCandles(candles: MarketCandle[], aggregate: number): HistoryPoint[] {
        const result: HistoryPoint[] = [];

        for (let i = 0; i < candles.length; i += aggregate) {
            const chunk = candles.slice(i, i + aggregate);
            if (chunk.length === 0) break;

            const open = Number(chunk[0].open);
            const close = Number(chunk[chunk.length - 1].close);
            const high = Math.max(...chunk.map(c => Number(c.high)));
            const low = Math.min(...chunk.map(c => Number(c.low)));
            const volume = chunk.reduce((sum, c) => sum + Number(c.volume), 0);

            result.push({
                ts: Number(chunk[0].timestamp),
                price: close,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        return result;
    }

    /**
     * Fallback: Fetch from CryptoCompare API
     */
    private async fetchFromCryptoCompare(
        symbol: string,
        limit: number,
        aggregate: number,
        type: 'minute' | 'hour' | 'day',
        toTs?: number,
    ): Promise<HistoryPoint[]> {
        let endpoint = 'histominute';
        if (type === 'hour') endpoint = 'histohour';
        if (type === 'day') endpoint = 'histoday';

        let url = `${this.baseUrl}/${endpoint}?fsym=${symbol}&tsym=USD&limit=${limit}&aggregate=${aggregate}`;

        if (toTs) {
            url += `&toTs=${toTs}`;
        }

        try {
            const response = await fetch(url, {
                headers: { authorization: `Apikey ${this.apiKey}` },
            });

            if (!response.ok) {
                throw new Error(`CryptoCompare API Error: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.Response === 'Error') {
                console.error('CryptoCompare Error:', data.Message);
                return [];
            }

            return data.Data.map((item: any) => ({
                ts: item.time * 1000,
                price: item.close,
                open: item.open,
                high: item.high,
                low: item.low,
                close: item.close,
                volume: item.volumeto,
            }));
        } catch (error) {
            console.error('Failed to fetch from CryptoCompare:', error);
            return [];
        }
    }
}
