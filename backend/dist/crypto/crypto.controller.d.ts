import { CryptoService, CoinData, HistoryPoint } from './crypto.service';
export declare class CryptoController {
    private readonly cryptoService;
    constructor(cryptoService: CryptoService);
    getTopCoins(): Promise<CoinData[]>;
    getHistoricalData(symbol: string, limit?: string, aggregate?: string, type?: 'minute' | 'hour' | 'day'): Promise<HistoryPoint[]>;
}
