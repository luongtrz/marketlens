import { AiService } from './ai.service';
export declare class AiController {
    private readonly aiService;
    constructor(aiService: AiService);
    analyzeArticle(body: {
        title: string;
        snippet: string;
        source: string;
    }): Promise<Partial<import("./ai.service").NewsArticle>>;
    generateMarketForecast(body: {
        coinName: string;
        recentTrend: string;
        currentPrice: number;
    }): Promise<import("./ai.service").ForecastResult>;
    askChartAnalyst(body: {
        coinSymbol: string;
        chartData: any[];
        question: string;
    }): Promise<string>;
    askNewsContext(body: {
        contextText: string;
        question: string;
    }): Promise<string>;
    chat(body: {
        message: string;
        history: any[];
    }): Promise<{
        text: string;
        groundingMetadata: any;
    }>;
    getLatestNews(): Promise<import("./ai.service").NewsArticle[]>;
    getHistoricalNews(coinName: string, date: string): Promise<import("./ai.service").NewsArticle[]>;
}
