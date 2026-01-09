import { Injectable } from '@nestjs/common';
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

@Injectable()
export class CryptoService {
    private readonly apiKey: string;
    private readonly baseUrl = 'https://min-api.cryptocompare.com/data';

    constructor(private configService: ConfigService) {
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

    async getHistoricalData(
        symbol: string,
        limit: number = 144,
        aggregate: number = 1,
        type: 'minute' | 'hour' | 'day' = 'minute',
    ): Promise<HistoryPoint[]> {
        let endpoint = 'histominute';
        if (type === 'hour') endpoint = 'histohour';
        if (type === 'day') endpoint = 'histoday';

        const url = `${this.baseUrl}/${endpoint}?fsym=${symbol}&tsym=USD&limit=${limit}&aggregate=${aggregate}`;

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
            console.error('Failed to fetch historical data:', error);
            return [];
        }
    }
}
