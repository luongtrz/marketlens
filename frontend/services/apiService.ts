import { CoinData, HistoryPoint, NewsArticle, ForecastResult, ChatMessage } from '../types';

/** Same-origin ``/api`` when using Vite proxy (dev) or nginx (Docker). Override with VITE_API_URL if needed. */
const API_BASE_URL =
    (typeof import.meta.env.VITE_API_URL === 'string' && import.meta.env.VITE_API_URL.trim() !== ''
        ? import.meta.env.VITE_API_URL
        : '/api');

const MOCK_MODE = String(import.meta.env.VITE_MOCK_MODE || '').toLowerCase() === 'true';

const mockNowIso = () => new Date().toISOString();
const AUTH_BASE = `${API_BASE_URL.replace(/\/$/, '')}/auth`;

type AuthResponse = {
    access_token: string;
    refresh_token: string;
    token_type: string;
    expires_in: number;
    user: { id: string; email: string };
};

const mockNews = (tag?: string): NewsArticle[] => {
    const sentiments: Array<'Positive' | 'Neutral' | 'Negative'> = ['Positive', 'Neutral', 'Negative'];
    const base = [
        {
            id: 'mock-1',
            title: 'Bitcoin rebounds as ETF inflows rise',
            source: 'mockwire',
            snippet: 'BTC recovered after strong inflows and improving risk sentiment across crypto markets.',
        },
        {
            id: 'mock-2',
            title: 'Ethereum devs signal upgrade timeline',
            source: 'chainpulse',
            snippet: 'Core developers outlined a tentative schedule for the next network upgrade.',
        },
        {
            id: 'mock-3',
            title: 'Market cools as traders lock in gains',
            source: 'cryptodaily',
            snippet: 'Profit-taking pressured majors despite steady on-chain activity.',
        },
    ];
    const tagValue = tag ? tag.toUpperCase() : undefined;
    return base.map((item, index) => ({
        ...item,
        sentiment: sentiments[Math.floor(Math.random() * sentiments.length)],
        sentimentScore: Math.floor(30 + Math.random() * 60),
        timestamp: new Date(Date.now() - index * 3600_000).toISOString(),
        url: 'https://example.com/mock-article',
        tag: tagValue,
    }));
};

const mockAuthResponse = (email: string): AuthResponse => ({
    access_token: `mock-access-${Math.random().toString(36).slice(2)}`,
    refresh_token: `mock-refresh-${Math.random().toString(36).slice(2)}`,
    token_type: 'bearer',
    expires_in: 3600,
    user: { id: 'mock-user', email },
});

const readAuthError = async (res: Response): Promise<string> => {
    try {
        const data = await res.json();
        if (Array.isArray(data?.detail)) {
            const parts = data.detail.map((item: any) => {
                const loc = Array.isArray(item?.loc) ? item.loc.join('.') : 'field';
                const msg = item?.msg || 'Invalid value';
                return `${loc}: ${msg}`;
            });
            return parts.join(' | ');
        }
        if (data?.detail) return String(data.detail);
    } catch {
        // ignore
    }
    return `Auth request failed (${res.status})`;
};

// Re-export market data functions from marketService
export {
    fetchSymbols,
    fetchHistoricalData,
    fetchSnapshot,
    fetchTopCoins,
    MarketWebSocket,
} from './marketService';
export type { MarketWebSocketOptions } from './marketService';

// Backward-compatible wrappers so existing code keeps working
export const getTopCoins = async (): Promise<CoinData[]> => {
    const { fetchTopCoins } = await import('./marketService');
    return fetchTopCoins();
};

export const getHistoricalData = async (
    symbol: string,
    limit: number = 144,
    aggregate: number = 1,
    type: 'minute' | 'hour' | 'day' = 'minute',
    endTime?: number,
): Promise<HistoryPoint[]> => {
    const { fetchHistoricalData } = await import('./marketService');
    // Map old aggregate/type params to new interval format
    const intervalMap: Record<string, string> = {
        minute: '1m',
        hour: '1h',
        day: '1d',
    };
    const baseInterval = intervalMap[type] || '1h';
    // Adjust limit for the new API
    const adjustedLimit = Math.min(limit * aggregate, 1000);
    return fetchHistoricalData(symbol, baseInterval, adjustedLimit, endTime);
};

// --- AI API ---

export const analyzeArticle = async (article: { title: string; snippet: string; source: string }): Promise<Partial<NewsArticle>> => {
    if (MOCK_MODE) {
        return {
            sentiment: 'Neutral',
            summary: `Mock summary for ${article.title || 'article'}.`,
            sentimentScore: 55,
            detailedSummary: 'This is a mock AI summary used for UI testing.',
            keyTakeaways: ['Mock takeaway 1', 'Mock takeaway 2', 'Mock takeaway 3'],
        };
    }
    try {
        const res = await fetch(`${API_BASE_URL}/ai/analyze-article`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(article)
        });
        if (!res.ok) throw new Error('API Error');
        return await res.json();
    } catch (e) {
        console.error('analyzeArticle failed', e);
        return { sentiment: 'Neutral', summary: 'Analysis unavailable.', sentimentScore: 0 };
    }
};

// --- Auth API ---

export const signUp = async (email: string, password: string): Promise<AuthResponse> => {
    if (MOCK_MODE) return mockAuthResponse(email);
    const res = await fetch(`${AUTH_BASE}/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
    });
    if (!res.ok) throw new Error(await readAuthError(res));
    return await res.json();
};

export const signIn = async (email: string, password: string): Promise<AuthResponse> => {
    if (MOCK_MODE) return mockAuthResponse(email);
    const res = await fetch(`${AUTH_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
    });
    if (!res.ok) throw new Error(await readAuthError(res));
    return await res.json();
};

export const refreshAuth = async (refreshToken: string): Promise<AuthResponse> => {
    if (MOCK_MODE) return mockAuthResponse('mock@marketlens.ai');
    const res = await fetch(`${AUTH_BASE}/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) throw new Error(await readAuthError(res));
    return await res.json();
};

export const generateMarketForecast = async (coinName: string, recentTrend: string, currentPrice: number): Promise<ForecastResult> => {
    if (MOCK_MODE) {
        const base = currentPrice || 100;
        const isBull = Math.random() > 0.5;
        return {
            predictedPrices: [
                base,
                base * (isBull ? 1.01 : 0.99),
                base * (isBull ? 1.015 : 0.985),
                base * (isBull ? 1.02 : 0.98),
                base * (isBull ? 1.03 : 0.97),
            ].map((v) => Number(v.toFixed(2))),
            confidenceScore: Math.floor(40 + Math.random() * 50),
            reasoning: `Mock forecast for ${coinName} (${isBull ? 'Upward' : 'Downward'}).`,
            trend: isBull ? 'Bullish' : 'Bearish',
            recommendation: { action: isBull ? 'Buy' : 'Hold', entryZone: '-', targetPrice: '-', stopLoss: '-' }
        };
    }
    try {
        const res = await fetch(`${API_BASE_URL}/ai/forecast`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coinName, recentTrend, currentPrice })
        });
        if (!res.ok) throw new Error('API Error');
        return await res.json();
    } catch (e) {
        console.error('generateMarketForecast failed', e);
        return {
            predictedPrices: [currentPrice, currentPrice, currentPrice, currentPrice, currentPrice],
            confidenceScore: 0,
            reasoning: "Unable to generate forecast due to network or API limits.",
            trend: "Neutral",
            recommendation: { action: "Hold", entryZone: "-", targetPrice: "-", stopLoss: "-" }
        };
    }
};

export const askChartAnalyst = async (coinSymbol: string, chartData: any[], question: string): Promise<string> => {
    if (MOCK_MODE) {
        return `Mock chart insight for ${coinSymbol}: ${question}`;
    }
    try {
        const res = await fetch(`${API_BASE_URL}/ai/ask-chart`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coinSymbol, chartData, question })
        });
        const text = await res.text();
        return text || "Analysis unavailable.";
    } catch (e) {
        console.error(e);
        return "Analysis unavailable.";
    }
};

export const askNewsContext = async (contextText: string, question: string): Promise<string> => {
    if (MOCK_MODE) {
        return `Mock news answer: ${question}`;
    }
    try {
        const res = await fetch(`${API_BASE_URL}/ai/ask-news`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contextText, question })
        });
        const text = await res.text();
        return text || "Analysis unavailable.";
    } catch (e) {
        return "Analysis unavailable.";
    }
};

export const getHistoricalNews = async (coinName: string, dateStr: string): Promise<NewsArticle[]> => {
    if (MOCK_MODE) {
        return mockNews(coinName).map((item) => ({
            ...item,
            timestamp: new Date(dateStr).toISOString(),
        }));
    }
    try {
        const res = await fetch(`${API_BASE_URL}/ai/historical-news?coinName=${encodeURIComponent(coinName)}&date=${encodeURIComponent(dateStr)}`);
        if (!res.ok) return [];
        return await res.json();
    } catch (e) {
        return [];
    }
};

export const fetchLatestNews = async (start?: string, end?: string, tag?: string): Promise<NewsArticle[]> => {
    if (MOCK_MODE) {
        return mockNews(tag).filter((item) => {
            const ts = new Date(item.timestamp).getTime();
            if (start && ts < new Date(start).getTime()) return false;
            if (end && ts > new Date(end).getTime()) return false;
            return true;
        });
    }
    try {
        const params = new URLSearchParams();
        if (start) params.append('start', start);
        if (end) params.append('end', end);
        if (tag) params.append('tag', tag);
        const query = params.toString();
        const base = API_BASE_URL.replace(/\/$/, '');
        const url = `${base}/ai/latest-news${query ? `?${query}` : ''}`;
        const res = await fetch(url);
        if (!res.ok) {
            console.error('fetchLatestNews failed with status:', res.status);
            return [];
        }
        return await res.json();
    } catch (e) {
        console.error('fetchLatestNews failed', e);
        return [];
    }
};

// Chat Session Adapter
export const createChatSession = () => {
    let history: { role: 'user' | 'model'; content: string }[] = [];

    return {
        sendMessage: async (message: string) => {
            if (MOCK_MODE) {
                const reply = `Mock reply: ${message}`;
                history.push({ role: 'user', content: message });
                history.push({ role: 'model', content: reply });
                return {
                    response: {
                        text: () => reply,
                        candidates: []
                    }
                };
            }
            // Call backend chat API with history
            try {
                const res = await fetch(`${API_BASE_URL}/ai/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, history })
                });

                if (!res.ok) throw new Error('API Error');

                const data = await res.json();

                // Update local history
                history.push({ role: 'user', content: message });
                history.push({ role: 'model', content: data.text });

                // Return compatible response object
                return {
                    response: {
                        text: () => data.text,
                        candidates: [{ groundingMetadata: data.groundingMetadata }]
                    }
                };
            } catch (e) {
                console.error('Chat error', e);
                // Return fallback so UI doesn't crash
                return {
                    response: {
                        text: () => "I'm having trouble connecting to the server.",
                        candidates: []
                    }
                };
            }
        }
    };
};

// Chat session adapter is the last remaining piece using the old API backend.
// AI endpoints (/ai/*) still live on the main_controller or old backend.
