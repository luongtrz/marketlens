import { CoinData, HistoryPoint } from '../types';

const API_KEY = process.env.CRYPTOCOMPARE_API_KEY;
const BASE_URL = 'https://min-api.cryptocompare.com/data';

const options = {
    method: 'GET',
    headers: {
        'authorization': `Apikey ${API_KEY || ''}`
    }
};

const formatVolume = (num: number): string => {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    return num.toLocaleString();
};

export const getTopCoins = async (): Promise<CoinData[]> => {
    // Fetch top 10 coins by market cap
    const url = `${BASE_URL}/top/mktcapfull?limit=10&tsym=USD`;

    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`CryptoCompare API Error: ${response.statusText}`);
        }
        const data = await response.json();

        if (data.Response === 'Error') {
            console.error("CryptoCompare Error:", data.Message);
            return [];
        }

        return data.Data.map((item: any) => {
            const coin = item.CoinInfo;
            const raw = item.RAW ? item.RAW.USD : {};
            const display = item.DISPLAY ? item.DISPLAY.USD : {};

            return {
                symbol: coin.Name,
                name: coin.FullName,
                price: raw.PRICE || 0,
                change24h: raw.CHANGE24HOUR || 0,
                volume: formatVolume(raw.VOLUME24HOUR || 0),
                marketCap: formatVolume(raw.MKTCAP || 0),
                history: []
            };
        });
    } catch (error) {
        console.error("Failed to fetch market data:", error);
        return [];
    }
};

export const getHistoricalData = async (symbol: string, limit: number = 144, aggregate: number = 1, type: 'minute' | 'hour' | 'day' = 'minute'): Promise<HistoryPoint[]> => {
    let endpoint = 'histominute';
    if (type === 'hour') endpoint = 'histohour';
    if (type === 'day') endpoint = 'histoday';

    const url = `${BASE_URL}/v2/${endpoint}?fsym=${symbol}&tsym=USD&limit=${limit}&aggregate=${aggregate}`;

    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            console.error(`Error fetching history for ${symbol}: ${response.status}`);
            return [];
        }
        const data = await response.json();

        if (data.Response === 'Error') {
            return [];
        }

        // Data.Data.Data contains array of { time, high, low, open, volumefrom, volumeto, close }
        const points = data.Data.Data || [];

        // CryptoCompare returns data in chronological order (oldest first)
        // Recharts expects chronological order, so no reverse needed
        return points.map((p: any) => ({
            time: new Date(p.time * 1000).toLocaleString(),
            ts: p.time * 1000,
            price: p.close,
            open: p.open,
            high: p.high,
            low: p.low,
            volume: p.volumeto
        }));

    } catch (e) {
        console.error(e);
        return [];
    }
};
