"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CryptoService = void 0;
const common_1 = require("@nestjs/common");
const config_1 = require("@nestjs/config");
let CryptoService = class CryptoService {
    configService;
    apiKey;
    baseUrl = 'https://min-api.cryptocompare.com/data';
    constructor(configService) {
        this.configService = configService;
        this.apiKey = this.configService.get('CRYPTOCOMPARE_API_KEY') || '';
    }
    formatVolume(num) {
        if (num >= 1e9)
            return (num / 1e9).toFixed(1) + 'B';
        if (num >= 1e6)
            return (num / 1e6).toFixed(1) + 'M';
        return num.toLocaleString();
    }
    async getTopCoins() {
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
            return data.Data.map((item) => {
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
        }
        catch (error) {
            console.error('Failed to fetch market data:', error);
            return [];
        }
    }
    async getHistoricalData(symbol, limit = 144, aggregate = 1, type = 'minute') {
        let endpoint = 'histominute';
        if (type === 'hour')
            endpoint = 'histohour';
        if (type === 'day')
            endpoint = 'histoday';
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
            return data.Data.map((item) => ({
                ts: item.time * 1000,
                price: item.close,
                open: item.open,
                high: item.high,
                low: item.low,
                close: item.close,
                volume: item.volumeto,
            }));
        }
        catch (error) {
            console.error('Failed to fetch historical data:', error);
            return [];
        }
    }
};
exports.CryptoService = CryptoService;
exports.CryptoService = CryptoService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [config_1.ConfigService])
], CryptoService);
//# sourceMappingURL=crypto.service.js.map