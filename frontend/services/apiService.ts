import { CoinData, HistoryPoint, NewsArticle, ForecastResult, ChatMessage } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';

// --- Crypto API ---

export const getTopCoins = async (): Promise<CoinData[]> => {
    try {
        const res = await fetch(`${API_BASE_URL}/crypto/top-coins`);
        if (!res.ok) throw new Error('API Error');
        return await res.json();
    } catch (e) {
        console.error('getTopCoins failed', e);
        return [];
    }
};

export const getHistoricalData = async (
    symbol: string,
    limit: number = 144,
    aggregate: number = 1,
    type: 'minute' | 'hour' | 'day' = 'minute'
): Promise<HistoryPoint[]> => {
    try {
        const res = await fetch(`${API_BASE_URL}/crypto/historical?symbol=${symbol}&limit=${limit}&aggregate=${aggregate}&type=${type}`);
        if (!res.ok) throw new Error('API Error');
        return await res.json();
    } catch (e) {
        console.error('getHistoricalData failed', e);
        return [];
    }
};

// --- AI API ---

export const analyzeArticle = async (article: { title: string; snippet: string; source: string }): Promise<Partial<NewsArticle>> => {
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
        return { sentiment: 'Neutral', summary: 'Analysis unavailable.', impactScore: 0 };
    }
};

export const generateMarketForecast = async (coinName: string, recentTrend: string, currentPrice: number): Promise<ForecastResult> => {
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
    try {
        const res = await fetch(`${API_BASE_URL}/ai/historical-news?coinName=${encodeURIComponent(coinName)}&date=${encodeURIComponent(dateStr)}`);
        if (!res.ok) return [];
        return await res.json();
    } catch (e) {
        return [];
    }
};

export const fetchLatestNews = async (): Promise<NewsArticle[]> => {
    try {
        const res = await fetch(`${API_BASE_URL}/ai/latest-news`);
        if (!res.ok) return [];
        return await res.json();
    } catch (e) {
        return [];
    }
};

// Chat Session Adapter
export const createChatSession = () => {
    let history: { role: 'user' | 'model'; content: string }[] = [];

    return {
        sendMessage: async (message: string) => {
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

// Real-time WebSocket Stub
import { io, Socket } from 'socket.io-client';

export const createSocketConnection = (namespace: string = 'realtime'): Socket => { // Default to realtime namespace
    const socketUrl = import.meta.env.VITE_API_URL?.replace('/api', '') || 'http://localhost:3001';
    return io(`${socketUrl}/${namespace}`, {
        transports: ['websocket'],
        reconnection: true,
    });
};
