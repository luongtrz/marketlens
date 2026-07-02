# Frontend Forecast Endpoints and XAI Schema

**Date:** 2026-06-27  
**Project:** MarketLens / MainController / Frontend

## 1. Current frontend wiring

Frontend dashboard is currently wired to:

- `POST /api/ai/forecast`

Implementation path:

- frontend client: [frontend/services/apiService.ts](/home/nmtc/projects/marketlens/frontend/services/apiService.ts:187)
- frontend type: [frontend/types.ts](/home/nmtc/projects/marketlens/frontend/types.ts:69)
- backend UI adapter: [main_controller/src/ui_routes.py](/home/nmtc/projects/marketlens/main_controller/src/ui_routes.py:452)

This route is a **legacy adapter**. It converts `PredictionResult` into a lightweight UI payload:

- `predictedPrices`
- `confidenceScore`
- `reasoning`
- `trend`
- `marketSummary`
- `recommendation`

It does **not** expose full retrieval evidence, similar cases, factor details, or diagnostics in a structured xAI format.

## 2. New XAI endpoint

Added:

- `POST /api/ai/forecast-xai`

Backend implementation:

- route + schema: [main_controller/src/ui_routes.py](/home/nmtc/projects/marketlens/main_controller/src/ui_routes.py:141)

Frontend client + types:

- client: [frontend/services/apiService.ts](/home/nmtc/projects/marketlens/frontend/services/apiService.ts:224)
- types: [frontend/types.ts](/home/nmtc/projects/marketlens/frontend/types.ts:77)

Purpose:

- keep `/api/ai/forecast` stable for existing dashboard
- expose a richer schema for:
  - xAI service
  - evidence rendering
  - trust diagnostics
  - downstream explanation composition

## 3. Request schema

Both forecast endpoints use the same request body:

```json
{
  "coinName": "BTC",
  "recentTrend": "Bullish breakout above resistance",
  "currentPrice": 67450.25
}
```

Accepted aliases on backend:

- `coinName` or `coin_name`
- `recentTrend` or `recent_trend`
- `currentPrice` or `current_price`

## 4. Full XAI response schema

```ts
type ForecastXAIResponse = {
  forecast: {
    symbol: string;
    as_of: string;
    signal: "BUY" | "SELL" | "HOLD";
    trend: "Bullish" | "Bearish" | "Neutral";
    confidence: number;
    confidence_score: number;
    current_price: number;
    predicted_prices: number[];
    horizon: string;
  };
  explanation: {
    summary: string;
    reasoning: string;
    reasoning_steps: string[];
    pipeline_notes: string[];
  };
  market_context: {
    sentiment_score: number;
    key_factors: Array<{
      name: string;
      type: string;
      weight: number;
      polarity: number;
      sector?: string | null;
      related_symbols: string[];
      source_article_id: string;
      observed_at: string;
    }>;
    market_snapshot?: {
      symbol: string;
      timestamp: string;
      ohlcv: {
        timestamp: string;
        open: number;
        high: number;
        low: number;
        close: number;
        volume: number;
        interval: string;
      };
      recent_candles: Array<{
        timestamp: string;
        open: number;
        high: number;
        low: number;
        close: number;
        volume: number;
        interval: string;
      }>;
      indicators: Record<string, unknown>;
      source: string;
    } | null;
  };
  decision_support: {
    recommendation: {
      action: string;
      entry_zone: string;
      target_price: string;
      stop_loss: string;
    };
    evidence: Array<{
      record_id?: string | null;
      date: string;
      symbol: string;
      similarity: number;
      retriever_version: string;
      outcome?: string | null;
      summary?: string | null;
      sentiment_score: number;
      finbert_sentiment_score?: number | null;
      event_match: Record<string, number>;
      factors: string[];
      normalized_factors: Array<{
        name: string;
        type: string;
        weight: number;
        polarity: number;
        sector?: string | null;
        related_symbols: string[];
        source_article_id: string;
        observed_at: string;
      }>;
      future_returns: {
        "1d"?: number | null;
        "3d"?: number | null;
        "7d"?: number | null;
        "15d"?: number | null;
        "30d"?: number | null;
      };
      article_ids: string[];
      article_sources: string[];
      market_snapshot?: {
        symbol: string;
        timestamp: string;
        ohlcv: {
          timestamp: string;
          open: number;
          high: number;
          low: number;
          close: number;
          volume: number;
          interval: string;
        };
        recent_candles: Array<{
          timestamp: string;
          open: number;
          high: number;
          low: number;
          close: number;
          volume: number;
          interval: string;
        }>;
        indicators: Record<string, unknown>;
        source: string;
      } | null;
    }>;
    retrieval: {
      retrieved_count: number;
      retriever_versions: string[];
    };
    cem_rag?: {
      horizon: string;
      p_up: number;
      p_down: number;
      p_hold: number;
      signal: string;
      confidence: number;
      tau: number;
      explanation?: string;
      retrieval_count?: number;
    } | null;
  };
  diagnostics: {
    errors: string[];
    schema_version: string;
  };
};
```

## 5. Example response

```json
{
  "forecast": {
    "symbol": "BTCUSDT",
    "as_of": "2026-06-27T09:12:31.114000Z",
    "signal": "BUY",
    "trend": "Bullish",
    "confidence": 0.64,
    "confidence_score": 64.0,
    "current_price": 67450.25,
    "predicted_prices": [67450.25, 67471.83, 67493.41, 67514.99, 67536.57],
    "horizon": "7d"
  },
  "explanation": {
    "summary": "Recent factors and similar historical setups lean bullish.",
    "reasoning": "Recent factors and similar historical setups lean bullish.\n[Chart trend hint: Bullish breakout above resistance]",
    "reasoning_steps": [
      "Market snapshot collected",
      "News sentiment and factors scored",
      "Retrieved top similar historical cases",
      "Decision head aggregated future returns"
    ],
    "pipeline_notes": []
  },
  "market_context": {
    "sentiment_score": 0.22,
    "key_factors": [
      {
        "name": "ETF inflow momentum",
        "type": "macro",
        "weight": 0.74,
        "polarity": 0.61,
        "sector": null,
        "related_symbols": ["BTC"],
        "source_article_id": "news_123",
        "observed_at": "2026-06-27T08:50:00Z"
      }
    ],
    "market_snapshot": {
      "symbol": "BTCUSDT",
      "timestamp": "2026-06-27T09:10:00Z",
      "ohlcv": {
        "timestamp": "2026-06-27T09:10:00Z",
        "open": 67380.0,
        "high": 67540.0,
        "low": 67310.0,
        "close": 67450.25,
        "volume": 1824.33,
        "interval": "1h"
      },
      "recent_candles": [],
      "indicators": {
        "rsi_14": 58.4,
        "macd_hist": 41.7
      },
      "source": "binance"
    }
  },
  "decision_support": {
    "recommendation": {
      "action": "Buy",
      "entry_zone": "-",
      "target_price": "-",
      "stop_loss": "-"
    },
    "evidence": [
      {
        "record_id": "mem_2024_05_14",
        "date": "2024-05-14",
        "symbol": "BTCUSDT",
        "similarity": 0.842113,
        "retriever_version": "learned_finbert_v1",
        "outcome": "UP",
        "summary": "ETF demand and positive macro tone supported upside.",
        "sentiment_score": 0.31,
        "finbert_sentiment_score": 0.44,
        "event_match": {
          "macro": 0.8,
          "sentiment": 0.7
        },
        "factors": ["ETF inflows", "risk-on sentiment"],
        "normalized_factors": [],
        "future_returns": {
          "1d": 1.2,
          "3d": 3.1,
          "7d": 6.4,
          "15d": 4.8,
          "30d": 8.6
        },
        "article_ids": ["news_123"],
        "article_sources": ["cointelegraph"],
        "market_snapshot": null
      }
    ],
    "retrieval": {
      "retrieved_count": 5,
      "retriever_versions": ["learned_finbert_v1"]
    },
    "cem_rag": null
  },
  "diagnostics": {
    "errors": [],
    "schema_version": "forecast_xai.v1"
  }
}
```

## 6. Recommended frontend usage

Use:

- `/api/ai/forecast` for current dashboard chart card only
- `/api/ai/forecast-xai` for:
  - evidence panels
  - “why this decision” UI
  - LLM/xAI explanation service input
  - trust/debug screens

Recommended xAI prompt input sections:

- `forecast`
- `market_context`
- `decision_support.recommendation`
- top `decision_support.evidence`
- `diagnostics`

## 7. Practical note

Right now the frontend is **not yet rendering** the new xAI payload. It only has:

- backend endpoint
- frontend TypeScript types
- frontend client helper

Next UI wiring step would be:

1. call `generateMarketForecastXAI(...)`
2. render top evidence cards
3. pass condensed evidence bundle into the separate explanation service
