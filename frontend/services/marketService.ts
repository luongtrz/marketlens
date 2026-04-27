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
    import.meta.env.VITE_MARKET_DATA_URL || 'http://localhost:8002';

// WebSocket URL: replace http(s) with ws(s)
function getWsUrl(): string {
    return MARKET_BASE_URL.replace(/^http/, 'ws') + '/ws';
}

// --- REST API ---

/**
 * Fetch available symbols for real-time streaming.
 * Returns e.g. ["BTC", "ETH", "SOL", "BNB", "XRP"]
 */
export const fetchSymbols = async (): Promise<string[]> => {
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
}
