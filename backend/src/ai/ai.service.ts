import { Injectable, OnModuleInit } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { GoogleGenAI, Type, FunctionDeclaration, Chat, GenerateContentResponse } from '@google/genai';

// Interfaces
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
    sources?: { title: string; url: string }[];
    recommendation?: {
        action: 'Buy' | 'Sell' | 'Hold';
        entryZone: string;
        targetPrice: string;
        stopLoss: string;
    };
}

@Injectable()
export class AiService implements OnModuleInit {
    private ai: GoogleGenAI;

    // Model Distribution Strategy
    private readonly ANALYSIS_MODEL = "gemma-3-4b";
    private readonly CHAT_MODEL = "gemini-2.5-flash-lite";
    private readonly PREMIUM_MODEL = "gemini-3-flash-preview";

    constructor(private configService: ConfigService) { }

    onModuleInit() {
        const apiKey = this.configService.get<string>('GEMINI_API_KEY');
        if (!apiKey || apiKey.trim() === '') {
            throw new Error("GEMINI_API_KEY is required but was not provided. Please set this environment variable.");
        }
        this.ai = new GoogleGenAI({ apiKey });
    }

    async analyzeArticle(title: string, snippet: string, source: string): Promise<Partial<NewsArticle>> {
        const model = this.ANALYSIS_MODEL;
        const prompt = `
      Analyze the following crypto news article snippet for financial sentiment.
      
      Title: ${title}
      Snippet: ${snippet}
      Source: ${source}
  
      Return a JSON object with:
      - sentiment: "Positive", "Negative", or "Neutral"
      - detailedSummary: A 2-3 sentence professional summary of the event.
      - keyTakeaways: An array of 3 short, bullet-point style takeaways for an investor.
      - impactScore: An integer from 0 (no impact) to 100 (high market moving potential).
    `;

        try {
            const response = await this.ai.models.generateContent({
                model,
                contents: prompt,
                config: {
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            sentiment: { type: Type.STRING, enum: ["Positive", "Negative", "Neutral"] },
                            detailedSummary: { type: Type.STRING },
                            keyTakeaways: { type: Type.ARRAY, items: { type: Type.STRING } },
                            impactScore: { type: Type.INTEGER },
                        },
                        required: ["sentiment", "detailedSummary", "keyTakeaways", "impactScore"],
                    },
                },
            });

            const jsonStr = response.text?.trim();
            if (jsonStr) {
                const data = JSON.parse(jsonStr);
                return {
                    ...data,
                    summary: data.detailedSummary
                };
            }
            throw new Error("Empty response");
        } catch (error) {
            console.error("Analysis failed", error);
            return {
                sentiment: 'Neutral',
                summary: 'Analysis unavailable.',
                impactScore: 0
            };
        }
    }

    async generateMarketForecast(coinName: string, recentTrend: string, currentPrice: number): Promise<ForecastResult> {
        const model = this.PREMIUM_MODEL;
        const prompt = `
      Act as a senior technical analyst at a top quantitative trading firm.
      The user is asking about ${coinName}.
      Current Price: $${currentPrice}.
      Recent Chart Trend: ${recentTrend}.
      
      1. Search for the latest news in the last 24 hours regarding ${coinName}.
      2. Generate a forecast for the next 5 time periods.
      3. Generate a concrete trading recommendation.
      4. Return ONLY a valid JSON object (no markdown, no code blocks, just raw JSON).
      
      CRITICAL: In the 'reasoning' field, you MUST include specific technical analysis references:
      - Estimate current RSI levels (e.g., "RSI is hovering at 65").
      - Mention Moving Averages (e.g., "Trading above the 50-period EMA").
      - Mention Volume profile.
      - Explicitly cite 1-2 specific news headlines found during search that support the technical view.
  
      JSON Schema:
         - predictedPrices: An array of 5 numbers representing the predicted price curve. First number close to ${currentPrice}.
         - confidenceScore: A number between 0 and 100.
         - reasoning: A detailed paragraph (approx 100 words) blending technicals and news.
         - trend: "Bullish", "Bearish", or "Neutral".
         - recommendation: Object containing:
              - action: "Buy", "Sell", or "Hold"
              - entryZone: A short string
              - targetPrice: A short string
              - stopLoss: A short string
    `;

        try {
            const response = await this.ai.models.generateContent({
                model,
                contents: prompt,
                config: {
                    tools: [{ googleSearch: {} }],
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            predictedPrices: { type: Type.ARRAY, items: { type: Type.NUMBER } },
                            confidenceScore: { type: Type.NUMBER },
                            reasoning: { type: Type.STRING },
                            trend: { type: Type.STRING, enum: ["Bullish", "Bearish", "Neutral"] },
                            recommendation: {
                                type: Type.OBJECT,
                                properties: {
                                    action: { type: Type.STRING, enum: ["Buy", "Sell", "Hold"] },
                                    entryZone: { type: Type.STRING },
                                    targetPrice: { type: Type.STRING },
                                    stopLoss: { type: Type.STRING }
                                },
                                required: ["action", "entryZone", "targetPrice", "stopLoss"]
                            }
                        },
                        required: ["predictedPrices", "confidenceScore", "reasoning", "trend", "recommendation"],
                    },
                },
            });

            const jsonStr = response.text?.trim();
            const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
            const sources = groundingChunks
                .map((chunk: any) => chunk.web ? { title: chunk.web.title, url: chunk.web.uri } : null)
                .filter((source): source is { title: string; url: string } => source !== null);

            if (jsonStr) {
                const result = JSON.parse(jsonStr);
                return { ...result, sources };
            }
            throw new Error("Empty response");
        } catch (error) {
            console.error("Forecast failed", error);
            return {
                predictedPrices: [currentPrice, currentPrice, currentPrice, currentPrice, currentPrice],
                confidenceScore: 0,
                reasoning: "Unable to generate forecast due to network or API limits.",
                trend: "Neutral"
            };
        }
    }

    async askChartAnalyst(coinSymbol: string, chartData: any[], question: string): Promise<string> {
        const model = this.ANALYSIS_MODEL;
        // Simplify data
        const recentData = chartData.slice(-20).map(p => ({
            t: p.time, p: p.price, o: p.open, h: p.high, l: p.low
        }));

        const prompt = `
      You are an expert technical analyst analyzing the chart for ${coinSymbol}.
      Context Data (Last 20 points): ${JSON.stringify(recentData)}
      User Question: "${question}"
      Answer concisely (max 2 sentences). Focus on technical patterns.
    `;

        try {
            const response = await this.ai.models.generateContent({ model, contents: prompt });
            return response.text || "Analysis unavailable.";
        } catch (e) {
            return "Analysis unavailable.";
        }
    }

    async askNewsContext(contextText: string, question: string): Promise<string> {
        const model = this.CHAT_MODEL;
        const prompt = `
      You are an AI news analyst.
      Context: ${contextText}
      User Question: "${question}"
      Answer based strictly on the context provided above. Keep it brief.
    `;

        try {
            const response = await this.ai.models.generateContent({ model, contents: prompt });
            return response.text || "Analysis unavailable.";
        } catch (e) {
            return "Analysis unavailable.";
        }
    }

    async getHistoricalNews(coinName: string, dateStr: string): Promise<NewsArticle[]> {
        const model = this.PREMIUM_MODEL;
        const prompt = `
      Search for major crypto news headlines specifically for ${coinName} that happened on or around ${dateStr}.
      Focus on market-moving events.
      Return a JSON object with 'articles' array.
    `;

        try {
            const response = await this.ai.models.generateContent({
                model,
                contents: prompt,
                config: {
                    tools: [{ googleSearch: {} }],
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            articles: {
                                type: Type.ARRAY,
                                items: {
                                    type: Type.OBJECT,
                                    properties: {
                                        title: { type: Type.STRING },
                                        source: { type: Type.STRING },
                                        snippet: { type: Type.STRING },
                                        sentiment: { type: Type.STRING, enum: ["Positive", "Negative", "Neutral"] },
                                        impactScore: { type: Type.INTEGER }
                                    },
                                    required: ["title", "source", "snippet", "sentiment", "impactScore"]
                                }
                            }
                        }
                    }
                },
            });

            const jsonStr = response.text?.trim();
            if (jsonStr) {
                const data = JSON.parse(jsonStr);
                if (data.articles && Array.isArray(data.articles)) {
                    return data.articles.map((item: any, index: number) => ({
                        id: `hist-${index}-${Date.now()}`,
                        title: item.title,
                        source: item.source,
                        snippet: item.snippet,
                        timestamp: dateStr,
                        url: '#',
                        sentiment: item.sentiment,
                        impactScore: item.impactScore,
                        summary: item.snippet
                    }));
                }
            }
            return [];
        } catch (e) {
            console.error(e);
            return [];
        }
    }
    async fetchLatestNews(): Promise<NewsArticle[]> {
        return [
            {
                id: '1',
                title: 'Bitcoin breaks $95k resistance as institutional inflow surges',
                source: 'CoinTelegraph',
                timestamp: '2 hours ago',
                snippet: 'ETF volumes have reached a new all-time high, pushing BTC price beyond key resistance levels.',
                url: '#',
                sentiment: 'Positive',
                summary: 'Institutional demand via ETFs is driving Bitcoin price action to new local highs.',
                impactScore: 85
            },
            {
                id: '2',
                title: 'SEC delays decision on Ethereum Spot ETF options',
                source: 'CoinDesk',
                timestamp: '4 hours ago',
                snippet: 'Regulators have asked for more public comment periods, causing a slight dip in ETH prices.',
                url: '#',
                sentiment: 'Negative',
                summary: 'Regulatory delays are causing short-term uncertainty for Ethereum investment products.',
                impactScore: 65
            },
            {
                id: '3',
                title: 'Solana network congestion issues resolved after update',
                source: 'The Block',
                timestamp: '5 hours ago',
                snippet: 'Validators have successfully deployed patch 1.18, restoring sub-second finality.',
                url: '#',
                sentiment: 'Positive',
                summary: 'Technical upgrades have stabilized the Solana network, restoring user confidence.',
                impactScore: 45
            },
            {
                id: '4',
                title: 'Macro Analysis: Inflation data suggests Fed pivot incoming',
                source: 'Bloomberg Crypto',
                timestamp: '6 hours ago',
                snippet: 'CPI data lower than expected. Risk-on assets likely to benefit in Q4.',
                url: '#',
                sentiment: 'Positive',
                summary: 'Favorable macroeconomic indicators suggest a bullish environment for crypto assets.',
                impactScore: 75
            },
            {
                id: '5',
                title: 'Ripple wins key procedural victory in ongoing lawsuit',
                source: 'CryptoSlate',
                timestamp: '8 hours ago',
                snippet: 'Court denies SEC motion to seal documents, XRP rallies 5% on the news.',
                url: '#',
                sentiment: 'Positive',
                summary: 'Legal wins for Ripple create positive momentum for XRP and regulatory clarity.',
                impactScore: 70
            },
            {
                id: '6',
                title: 'Major exchange experiences flash crash in Asia trading hours',
                source: 'The Block',
                timestamp: '9 hours ago',
                snippet: 'Thin liquidity led to a 10% wicking on several altcoin pairs before stabilization.',
                url: '#',
                sentiment: 'Negative',
                summary: 'Liquidity issues on centralized exchanges highlight risks during off-peak hours.',
                impactScore: 55
            },
            {
                id: '7',
                title: 'DeFi TVL reaches 12-month high led by liquid staking',
                source: 'DefiLlama News',
                timestamp: '11 hours ago',
                snippet: 'Lido and RocketPool lead the charge as users seek yield in a sideways market.',
                url: '#',
                sentiment: 'Positive',
                summary: 'Growth in Decentralized Finance Total Value Locked indicates healthy ecosystem engagement.',
                impactScore: 60
            },
            {
                id: '8',
                title: 'New EU crypto regulations MiCA enter final phase',
                source: 'CoinDesk',
                timestamp: '12 hours ago',
                snippet: 'Stablecoin issuers must comply with strict reserve requirements starting next month.',
                url: '#',
                sentiment: 'Neutral',
                summary: 'Regulatory clarity is good for long-term growth but imposes short-term compliance costs.',
                impactScore: 80
            },
            {
                id: '9',
                title: 'Avalanche partners with major gaming studio for subnet launch',
                source: 'Decrypt',
                timestamp: '14 hours ago',
                snippet: 'Triple-A game title to be built exclusively on an Avalanche subnet.',
                url: '#',
                sentiment: 'Positive',
                summary: 'Web3 Gaming partnerships demonstrate real-world utility for blockchain infrastructure.',
                impactScore: 50
            },
            {
                id: '10',
                title: 'Bitcoin miner difficulty adjustment hits all-time high',
                source: 'CoinTelegraph',
                timestamp: '15 hours ago',
                snippet: 'Competition among miners intensifies as hashrate continues to climb.',
                url: '#',
                sentiment: 'Neutral',
                summary: 'High network security is positive, but miner profitability is being squeezed.',
                impactScore: 40
            }
        ];
    }

    async chat(message: string, history: { role: 'user' | 'model'; content: string }[]): Promise<{ text: string; groundingMetadata: any }> {
        const model = this.CHAT_MODEL;
        const chat = this.ai.chats.create({
            model: model,
            config: {
                systemInstruction: "You are Sibyl, an AI assistant specialized in Cryptocurrency and Financial Markets. You provide data-backed answers, explain technical concepts clearly, and always warn about risks. Use Google Search to find real-time info. Do not give financial advice (NFA). At the end of your response, if relevant, briefly suggest 1-2 related topics or questions the user might want to explore next.",
                tools: [{ googleSearch: {} }],
            },
            history: history.map(msg => ({
                role: msg.role,
                parts: [{ text: msg.content }]
            }))
        });

        const result = await (chat as any).sendMessage(message);
        return {
            text: result.text || "",
            groundingMetadata: result.candidates?.[0]?.groundingMetadata
        };
    }
}
