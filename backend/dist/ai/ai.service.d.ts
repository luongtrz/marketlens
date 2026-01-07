import { OnModuleInit } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
export interface NewsArticle {
    id: string;
    title: string;
    source: string;
    timestamp: string;
    snippet: string;
    url: string;
    sentiment: 'Positive' | 'Negative' | 'Neutral';
    summary?: string;
    impactScore: number;
}
export interface ForecastResult {
    predictedPrices: number[];
    confidenceScore: number;
    reasoning: string;
    trend: 'Bullish' | 'Bearish' | 'Neutral';
    sources?: {
        title: string;
        url: string;
    }[];
    recommendation?: {
        action: 'Buy' | 'Sell' | 'Hold';
        entryZone: string;
        targetPrice: string;
        stopLoss: string;
    };
}
export declare class AiService implements OnModuleInit {
    private configService;
    private ai;
    private readonly ANALYSIS_MODEL;
    private readonly CHAT_MODEL;
    private readonly PREMIUM_MODEL;
    constructor(configService: ConfigService);
    onModuleInit(): void;
    analyzeArticle(title: string, snippet: string, source: string): Promise<Partial<NewsArticle>>;
    generateMarketForecast(coinName: string, recentTrend: string, currentPrice: number): Promise<ForecastResult>;
    askChartAnalyst(coinSymbol: string, chartData: any[], question: string): Promise<string>;
    askNewsContext(contextText: string, question: string): Promise<string>;
    getHistoricalNews(coinName: string, dateStr: string): Promise<NewsArticle[]>;
    fetchLatestNews(): Promise<NewsArticle[]>;
    chat(message: string, history: {
        role: 'user' | 'model';
        content: string;
    }[]): Promise<{
        text: string;
        groundingMetadata: any;
    }>;
}
