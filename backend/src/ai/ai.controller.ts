import { Controller, Post, Body, Get, Query } from '@nestjs/common';
import { AiService } from './ai.service';

@Controller('api/ai')
export class AiController {
    constructor(private readonly aiService: AiService) { }

    @Post('analyze-article')
    async analyzeArticle(@Body() body: { title: string; snippet: string; source: string }) {
        return this.aiService.analyzeArticle(body.title, body.snippet, body.source);
    }

    @Post('forecast')
    async generateMarketForecast(@Body() body: { coinName: string; recentTrend: string; currentPrice: number }) {
        return this.aiService.generateMarketForecast(body.coinName, body.recentTrend, body.currentPrice);
    }

    @Post('ask-chart')
    async askChartAnalyst(@Body() body: { coinSymbol: string; chartData: any[]; question: string }) {
        return this.aiService.askChartAnalyst(body.coinSymbol, body.chartData, body.question);
    }

    @Post('ask-news')
    async askNewsContext(@Body() body: { contextText: string; question: string }) {
        return this.aiService.askNewsContext(body.contextText, body.question);
    }

    @Post('chat')
    async chat(@Body() body: { message: string; history: any[] }) {
        return this.aiService.chat(body.message, body.history);
    }

    @Get('latest-news')
    async getLatestNews(@Query('start') start?: string, @Query('end') end?: string, @Query('tag') tag?: string) {
        return this.aiService.fetchLatestNews(start, end, tag);
    }

    @Get('historical-news')
    async getHistoricalNews(@Query('coinName') coinName: string, @Query('date') date: string) {
        return this.aiService.getHistoricalNews(coinName, date);
    }
}
