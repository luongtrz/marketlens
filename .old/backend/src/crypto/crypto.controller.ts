import { Controller, Get, Query } from '@nestjs/common';
import { CryptoService, CoinData, HistoryPoint } from './crypto.service';

@Controller('api/crypto')
export class CryptoController {
    constructor(private readonly cryptoService: CryptoService) { }

    @Get('top-coins')
    async getTopCoins(): Promise<CoinData[]> {
        return this.cryptoService.getTopCoins();
    }

    @Get('historical')
    async getHistoricalData(
        @Query('symbol') symbol: string,
        @Query('limit') limit?: string,
        @Query('aggregate') aggregate?: string,
        @Query('type') type?: 'minute' | 'hour' | 'day',
        @Query('toTs') toTs?: string, // Unix timestamp (seconds) for end time
    ): Promise<HistoryPoint[]> {
        return this.cryptoService.getHistoricalData(
            symbol,
            limit ? parseInt(limit) : 144,
            aggregate ? parseInt(aggregate) : 1,
            type || 'minute',
            toTs ? parseInt(toTs) : undefined,
        );
    }
}
