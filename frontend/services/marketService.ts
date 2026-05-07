/**
 * Market Data Service — connects to the market_data backend (FastAPI).
 *
 * REST:  GET /symbols, GET /history, GET /snapshot
 * WS:    /ws (native WebSocket, not Socket.IO)
 *
 * All data is normalized to the frontend's existing types so Dashboard
 * and other components don't need to change.
 */

import { CoinData, HistoryPoint } from '../types';

// --- Configuration ---

const MARKET_BASE_URL =
    import.meta.env.VITE_MARKET_DATA_URL || '/market';

const MOCK_MODE = String(import.meta.env.VITE_MOCK_MODE || '').toLowerCase() === 'true';
const MOCK_SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP'];

const buildMockHistory = (symbol: string, limit: number): HistoryPoint[] => {
    const now = Date.now();
    const base = symbol === 'BTC' ? 62000 : symbol === 'ETH' ? 3200 : 180;
    const direction = Math.random() > 0.5 ? 1 : -1;
    const points: HistoryPoint[] = [];
    for (let i = limit - 1; i >= 0; i--) {
        const ts = now - i * 60 * 60 * 1000;
        const drift = Math.sin((limit - i) / 6) * 0.015 * direction;
        const price = base * (1 + drift);
        points.push({
            ts,
            time: new Date(ts).toISOString(),
            price: Number(price.toFixed(2)),
            open: Number((price * 0.995).toFixed(2)),
            high: Number((price * 1.01).toFixed(2)),
            low: Number((price * 0.99).toFixed(2)),
            close: Number(price.toFixed(2)),
            volume: Number((Math.random() * 1000 + 500).toFixed(2)),
        });
    }
    return points;
};

// WebSocket URL — handle both absolute URLs and same-origin paths
function getWsUrl(): string {
    if (MARKET_BASE_URL.startsWith('http')) {
        return MARKET_BASE_URL.replace(/^http/, 'ws') + '/ws';
    }
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${window.location.host}${MARKET_BASE_URL}/ws`;
}

// --- REST API ---

/**
 * Fetch available symbols for real-time streaming.
 * Returns e.g. ["BTC", "ETH", "SOL", "BNB", "XRP"]
 */
export const fetchSymbols = async (): Promise<string[]> => {
    if (MOCK_MODE) {
        return [...MOCK_SYMBOLS];
    }
    try {
        const res = await fetch(`${MARKET_BASE_URL}/symbols`);
        if (!res.ok) throw new Error('Failed to fetch symbols');
        const data = await res.json();
        return data.symbols || [];
    } catch (e) {
        console.error('fetchSymbols failed', e);
        return [];
    }
};

/**
 * Fetch historical OHLCV candles from Binance via market_data.
 *
 * Converts the backend OHLCV format to the frontend's HistoryPoint format.
 */
export const fetchHistoricalData = async (
    symbol: string,
    interval: string = '1h',
    limit: number = 200,
    endTime?: number,
): Promise<HistoryPoint[]> => {
    if (MOCK_MODE) {
        return buildMockHistory(symbol.toUpperCase(), Math.min(limit, 200));
    }
    try {
        let url = `${MARKET_BASE_URL}/history?symbol=${symbol}&interval=${interval}&limit=${limit}`;
        if (endTime) {
            url += `&end_time=${endTime}`;
        }
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to fetch history');
        const candles = await res.json();

        return candles.map((c: any) => ({
            ts: new Date(c.timestamp).getTime(),
            time: c.timestamp, // Will be formatted by Dashboard
            price: c.close,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
            volume: c.volume,
        }));
    } catch (e) {
        console.error('fetchHistoricalData failed', e);
        return [];
    }
};

/**
 * Fetch current market snapshot with indicators.
 */
export const fetchSnapshot = async (
    symbol: string,
    interval: string = '1h',
): Promise<{
    price: number;
    change24h: number;
    indicators: Record<string, any>;
} | null> => {
    if (MOCK_MODE) {
        const base = symbol === 'BTC' ? 62000 : symbol === 'ETH' ? 3200 : 180;
        const price = base * (1 + Math.sin(Date.now() / 1_000_000) * 0.01);
        const changeSign = Math.random() > 0.5 ? 1 : -1;
        return { price: Number(price.toFixed(2)), change24h: Number((price * 0.02 * changeSign).toFixed(2)), indicators: {} };
    }
    try {
        const res = await fetch(
            `${MARKET_BASE_URL}/snapshot?symbol=${symbol}&interval=${interval}`,
        );
        if (!res.ok) return null;
        const data = await res.json();
        return {
            price: data.ohlcv.close,
            change24h: 0, // Snapshot doesn't include 24h change
            indicators: data.indicators || {},
        };
    } catch (e) {
        console.error('fetchSnapshot failed', e);
        return null;
    }
};

/**
 * Fetch top coins data. Builds CoinData[] from multiple snapshot + history calls.
 *
 * In production, you might want a dedicated `/top-coins` endpoint on market_data.
 * For now, this fetches symbols and their latest price.
 */
export const fetchTopCoins = async (): Promise<CoinData[]> => {
    if (MOCK_MODE) {
        return MOCK_SYMBOLS.map((symbol) => {
            const history = buildMockHistory(symbol, 24);
            const currentPrice = history[history.length - 1].price || 0;
            const firstClose = history[0]?.price || currentPrice;
            const change24h = currentPrice - firstClose;
            return {
                symbol,
                name: symbol,
                price: currentPrice,
                change24h,
                volume: '-',
                marketCap: '-',
                history,
            };
        });
    }
    const symbols = await fetchSymbols();
    const coins: CoinData[] = [];

    for (const symbol of symbols) {
        try {
            const snapshot = await fetchSnapshot(symbol, '1h');
            const history = await fetchHistoricalData(symbol, '1h', 24);

            // Calculate 24h change from history
            const firstClose = history.length > 0 ? history[0].price! : snapshot?.price || 0;
            const currentPrice = snapshot?.price || 0;
            const change24h = currentPrice - firstClose;

            coins.push({
                symbol,
                name: symbol, // Could be enhanced with full names
                price: currentPrice,
                change24h,
                volume: '-',
                marketCap: '-',
                history,
            });
        } catch {
            console.warn(`Failed to fetch data for ${symbol}`);
        }
    }

    return coins;
};

// --- WebSocket Manager ---

export interface MarketWebSocketOptions {
    /** Called on every trade update with raw price */
    onTrade?: (symbol: string, price: number, message: any) => void;
    /** Called on every kline (candlestick) update */
    onKline?: (symbol: string, data: {
        time: number; open: number; high: number;
        low: number; close: number; volume: number; isFinal: boolean;
    }) => void;
    /** Called when connection status changes */
    onStatusChange?: (status: 'connected' | 'disconnected' | 'error') => void;
}

export class MarketWebSocket {
    private ws: WebSocket | null = null;
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    private mockTimer: ReturnType<typeof setInterval> | null = null;
    private mockPrices = new Map<string, number>();
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 10;
    private reconnectDelay = 3000; // ms
    private subscriptions = new Set<string>();
    private options: MarketWebSocketOptions;
    private isClosing = false;

    constructor(options: MarketWebSocketOptions = {}) {
        this.options = options;
    }

    connect(): void {
        this.isClosing = false;
        if (MOCK_MODE) {
            this.options.onStatusChange?.('connected');
            this._startMockFeed();
            return;
        }
        try {
            this.ws = new WebSocket(getWsUrl());

            this.ws.onopen = () => {
                console.log('[MarketWS] Connected');
                this.reconnectAttempts = 0;
                this.options.onStatusChange?.('connected');

                // Resubscribe on reconnect
                for (const sub of this.subscriptions) {
                    const [type, symbol] = sub.split(':');
                    this._sendSubscribe(symbol, type as 'kline' | 'trade');
                }
            };

            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this._handleMessage(message);
                } catch (e) {
                    console.error('[MarketWS] Parse error', e);
                }
            };

            this.ws.onclose = () => {
                console.log('[MarketWS] Disconnected');
                this.options.onStatusChange?.('disconnected');
                if (!this.isClosing) {
                    this._scheduleReconnect();
                }
            };

            this.ws.onerror = (error) => {
                console.error('[MarketWS] Error', error);
                this.options.onStatusChange?.('error');
            };
        } catch (e) {
            console.error('[MarketWS] Connection error', e);
            this.options.onStatusChange?.('error');
            this._scheduleReconnect();
        }
    }

    subscribe(symbol: string, type: 'kline' | 'trade'): void {
        const key = `${type}:${symbol.toUpperCase()}`;
        this.subscriptions.add(key);
        if (this.ws?.readyState === WebSocket.OPEN) {
            this._sendSubscribe(symbol, type);
        }
    }

    unsubscribe(symbol: string, type: 'kline' | 'trade'): void {
        const key = `${type}:${symbol.toUpperCase()}`;
        this.subscriptions.delete(key);
        if (this.ws?.readyState === WebSocket.OPEN) {
            this._sendUnsubscribe(symbol, type);
        }
    }

    disconnect(): void {
        this.isClosing = true;
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.mockTimer) {
            clearInterval(this.mockTimer);
            this.mockTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.subscriptions.clear();
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    // --- Private ---

    private _sendSubscribe(symbol: string, type: 'kline' | 'trade'): void {
        this.ws?.send(JSON.stringify({
            action: 'subscribe',
            symbol: symbol.toUpperCase(),
            type,
        }));
    }

    private _sendUnsubscribe(symbol: string, type: 'kline' | 'trade'): void {
        this.ws?.send(JSON.stringify({
            action: 'unsubscribe',
            symbol: symbol.toUpperCase(),
            type,
        }));
    }

    private _handleMessage(message: any): void {
        if (message.type === 'trade') {
            const data = message.data;
            const symbol = (data.s || '').replace('USDT', '');
            const price = parseFloat(data.p);
            this.options.onTrade?.(symbol, price, data);
        } else if (message.type === 'kline') {
            const data = message.data;
            this.options.onKline?.(data.symbol, {
                time: data.time,
                open: data.open,
                high: data.high,
                low: data.low,
                close: data.close,
                volume: data.volume,
                isFinal: data.isFinal,
            });
        } else if (message.error) {
            console.warn('[MarketWS] Server error:', message.error);
        }
    }

    private _scheduleReconnect(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('[MarketWS] Max reconnect attempts reached');
            return;
        }
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);
        console.log(`[MarketWS] Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);
        this.reconnectTimer = setTimeout(() => this.connect(), delay);
    }

    private _startMockFeed(): void {
        if (this.mockTimer) return;
        this.mockTimer = setInterval(() => {
            const now = Date.now();
            for (const sub of this.subscriptions) {
                const [type, symbol] = sub.split(':');
                const key = symbol.toUpperCase();
                const prev = this.mockPrices.get(key) || (key === 'BTC' ? 62000 : key === 'ETH' ? 3200 : 180);
                const next = prev * (1 + (Math.random() - 0.5) * 0.002);
                this.mockPrices.set(key, next);

                if (type === 'trade') {
                    this.options.onTrade?.(key, Number(next.toFixed(2)), { p: next, s: `${key}USDT` });
                } else if (type === 'kline') {
                    this.options.onKline?.(key, {
                        time: now,
                        open: prev,
                        high: Math.max(prev, next),
                        low: Math.min(prev, next),
                        close: next,
                        volume: Math.random() * 1000,
                        isFinal: true,
                    });
                }
            }
        }, 1000);
    }
}
