
export interface NewsArticle {
  id: string;
  title: string;
  source: string;
  timestamp: string;
  snippet: string;
  url: string;
  sentiment?: 'Positive' | 'Negative' | 'Neutral';
  summary?: string;
  detailedSummary?: string;
  keyTakeaways?: string[];
  sentimentScore?: number; // model scale -1 (bearish) … +1 (bullish)
  tag?: string;
}

/** Response from ``GET /api/ai/latest-news?page=&page_size=``. */
export interface LatestNewsPage {
  items: NewsArticle[];
  page: number;
  page_size: number;
  total: number;
}

export interface HistoryPoint {
  time: string;
  ts: number; // Raw timestamp for calculations
  price?: number; // Close price (optional for forecast points)
  open?: number;
  high?: number;
  low?: number;
  volume?: number; // Volume traded
  forecast?: number;
  sentimentScore?: number; // optional chart overlay (not normalized to news API)
  newsSummary?: string;
}

export interface CoinData {
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  volume: string;
  marketCap: string;
  history: HistoryPoint[];
}

export interface PriceAlert {
  id: string;
  symbol: string;
  targetPrice: number;
  condition: 'above' | 'below';
  createdAt: number;
}

export interface GroundingSource {
  title: string;
  url: string;
}

export interface Recommendation {
  action: 'Buy' | 'Sell' | 'Hold';
  entryZone: string;
  targetPrice: string;
  stopLoss: string;
}

export interface ForecastResult {
  predictedPrices: number[];
  confidenceScore: number; // 0-100
  reasoning: string;
  trend: 'Bullish' | 'Bearish' | 'Neutral';
  marketSummary?: string;
  recommendation?: Recommendation;
  sources?: GroundingSource[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  content: string;
  timestamp: Date;
  groundingMetadata?: {
    groundingChunks?: {
      web?: {
        uri: string;
        title: string;
      };
    }[];
  };
}

export interface MarketSentiment {
  overallScore: number; // -100 to 100
  label: 'Bearish' | 'Neutral' | 'Bullish';
  summary: string;
}

export enum AnalysisStatus {
  IDLE = 'IDLE',
  LOADING = 'LOADING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}

export type SignalType = 'Strong Sell' | 'Sell' | 'Neutral' | 'Buy' | 'Strong Buy';
