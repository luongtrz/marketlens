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
exports.AiService = void 0;
const common_1 = require("@nestjs/common");
const config_1 = require("@nestjs/config");
const genai_1 = require("@google/genai");
let AiService = class AiService {
    configService;
    ai;
    ANALYSIS_MODEL = "gemma-3-4b";
    CHAT_MODEL = "gemini-2.5-flash-lite";
    PREMIUM_MODEL = "gemini-3-flash-preview";
    constructor(configService) {
        this.configService = configService;
    }
    onModuleInit() {
        const apiKey = this.configService.get('GEMINI_API_KEY');
        if (!apiKey) {
            console.error("GEMINI_API_KEY is missing");
        }
        this.ai = new genai_1.GoogleGenAI({ apiKey: apiKey || 'dummy-key' });
    }
    async analyzeArticle(title, snippet, source) {
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
                        type: genai_1.Type.OBJECT,
                        properties: {
                            sentiment: { type: genai_1.Type.STRING, enum: ["Positive", "Negative", "Neutral"] },
                            detailedSummary: { type: genai_1.Type.STRING },
                            keyTakeaways: { type: genai_1.Type.ARRAY, items: { type: genai_1.Type.STRING } },
                            impactScore: { type: genai_1.Type.INTEGER },
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
        }
        catch (error) {
            console.error("Analysis failed", error);
            return {
                sentiment: 'Neutral',
                summary: 'Analysis unavailable.',
                impactScore: 0
            };
        }
    }
    async generateMarketForecast(coinName, recentTrend, currentPrice) {
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
                        type: genai_1.Type.OBJECT,
                        properties: {
                            predictedPrices: { type: genai_1.Type.ARRAY, items: { type: genai_1.Type.NUMBER } },
                            confidenceScore: { type: genai_1.Type.NUMBER },
                            reasoning: { type: genai_1.Type.STRING },
                            trend: { type: genai_1.Type.STRING, enum: ["Bullish", "Bearish", "Neutral"] },
                            recommendation: {
                                type: genai_1.Type.OBJECT,
                                properties: {
                                    action: { type: genai_1.Type.STRING, enum: ["Buy", "Sell", "Hold"] },
                                    entryZone: { type: genai_1.Type.STRING },
                                    targetPrice: { type: genai_1.Type.STRING },
                                    stopLoss: { type: genai_1.Type.STRING }
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
                .map((chunk) => chunk.web ? { title: chunk.web.title, url: chunk.web.uri } : null)
                .filter((source) => source !== null);
            if (jsonStr) {
                const result = JSON.parse(jsonStr);
                return { ...result, sources };
            }
            throw new Error("Empty response");
        }
        catch (error) {
            console.error("Forecast failed", error);
            return {
                predictedPrices: [currentPrice, currentPrice, currentPrice, currentPrice, currentPrice],
                confidenceScore: 0,
                reasoning: "Unable to generate forecast due to network or API limits.",
                trend: "Neutral"
            };
        }
    }
    async askChartAnalyst(coinSymbol, chartData, question) {
        const model = this.ANALYSIS_MODEL;
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
        }
        catch (e) {
            return "Analysis unavailable.";
        }
    }
    async askNewsContext(contextText, question) {
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
        }
        catch (e) {
            return "Analysis unavailable.";
        }
    }
    async getHistoricalNews(coinName, dateStr) {
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
                        type: genai_1.Type.OBJECT,
                        properties: {
                            articles: {
                                type: genai_1.Type.ARRAY,
                                items: {
                                    type: genai_1.Type.OBJECT,
                                    properties: {
                                        title: { type: genai_1.Type.STRING },
                                        source: { type: genai_1.Type.STRING },
                                        snippet: { type: genai_1.Type.STRING },
                                        sentiment: { type: genai_1.Type.STRING, enum: ["Positive", "Negative", "Neutral"] },
                                        impactScore: { type: genai_1.Type.INTEGER }
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
                    return data.articles.map((item, index) => ({
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
        }
        catch (e) {
            console.error(e);
            return [];
        }
    }
    async fetchLatestNews() {
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
    async chat(message, history) {
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
        const result = await chat.sendMessage(message);
        return {
            text: result.text || "",
            groundingMetadata: result.candidates?.[0]?.groundingMetadata
        };
    }
};
exports.AiService = AiService;
exports.AiService = AiService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [config_1.ConfigService])
], AiService);
//# sourceMappingURL=ai.service.js.map