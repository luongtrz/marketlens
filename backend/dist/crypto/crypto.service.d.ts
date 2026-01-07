import { ConfigService } from '@nestjs/config';
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
export declare class CryptoService {
    private configService;
    private readonly apiKey;
    private readonly baseUrl;
    constructor(configService: ConfigService);
    private formatVolume;
    getTopCoins(): Promise<CoinData[]>;
    getHistoricalData(symbol: string, limit?: number, aggregate?: number, type?: 'minute' | 'hour' | 'day'): Promise<HistoryPoint[]>;
}
