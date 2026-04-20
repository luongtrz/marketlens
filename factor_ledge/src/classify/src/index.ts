/**
 * classify-service — port 3001
 *
 * Nhận raw factor string, trả về group name.
 * Thứ tự ưu tiên: exact match → cache → LLM fallback
 *
 * POST /classify        { factor: string }
 * POST /classify/batch  { factors: string[] }
 * GET  /health
 */

import fetch from "node-fetch";
import {
  getFactorGroup,
  GROUPS,
  EVENT_TAXONOMY,
} from "../../../shared/Taxonomy_en.ts";
import type { ClassifyResponse } from "../../../shared/types.ts";

// Google Gemini API key (bạn có thể đặt vào biến môi trường nếu muốn bảo mật hơn)
const GEMINI_API_KEY = "[REDACTED_GOOGLE_API_KEY]";
const PORT = Number(process.env.CLASSIFY_PORT ?? 3001);

// In-memory cache: factor string → group
// Production: thay bằng Redis SET "classify:cache:<factor>" <group>
const classifyCache = new Map<string, string>();

// ----------------------------------------------------------------
// Similarity scoring & fuzzy matching
// ----------------------------------------------------------------

function calculateSimilarity(str1: string, str2: string): number {
  const s1 = str1.toLowerCase();
  const s2 = str2.toLowerCase();
  
  // Exact match = 1.0
  if (s1 === s2) return 1.0;
  
  // Word-based matching
  const words1 = s1.split(/\s+/);
  const words2 = s2.split(/\s+/);
  
  let matches = 0;
  for (const w1 of words1) {
    for (const w2 of words2) {
      if (w1 === w2 || w2.startsWith(w1) || w1.startsWith(w2)) {
        matches++;
        break;
      }
    }
  }
  
  const maxLen = Math.max(words1.length, words2.length);
  return maxLen > 0 ? matches / maxLen : 0;
}

function classifyByFuzzyMatching(factor: string): string | null {
  let bestGroup: string | null = null;
  let bestScore = 0.3; // Threshold
  
  // Compare dengan tất cả event types
  for (const [group, types] of Object.entries(EVENT_TAXONOMY)) {
    for (const eventType of types) {
      const score = calculateSimilarity(factor, eventType);
      if (score > bestScore) {
        bestScore = score;
        bestGroup = group;
      }
    }
  }
  
  return bestGroup;
}

// ----------------------------------------------------------------
// Keyword-based fallback classify (simpler keyword map)
// ----------------------------------------------------------------

function classifyByKeyword(factor: string): string | null {
  const lower = factor.toLowerCase();
  
  const keywordMap: Record<string, string[]> = {
    "Regulation & Legal": ["regulation", "legal", "government", "law", "enforcement", "legislative", "ban", "sanction", "compliance", "sec", "regulatory"],
    "Macroeconomic": ["interest rate", "inflation", "fed", "cpi", "economic", "gdp", "dollar", "quantitative", "monetary", "recession"],
    "Industry Standards & Opinions": ["protocol proposal", "industry", "report", "opinion", "analyst", "influencer"],
    "Protocol & Product": ["protocol", "upgrade", "feature", "launch", "testnet", "mainnet", "adoption", "fee", "gas", "hash rate", "supply", "halving"],
    "Technology & Development": ["technical", "breakthrough", "development", "audit", "certification", "node", "validator", "ecosystem", "integration", "tooling", "update"],
    "Exchange & Trading": ["exchange", "listing", "delisting", "funding", "revenue", "acquisition", "partnership", "custody", "liquidation", "reserve", "trading", "volume"],
    "DeFi & Ecosystem": ["defi", "protocol launch", "migration", "cross-chain", "yield"],
    "Whale & On-chain": ["whale", "accumulation", "distribution", "on-chain", "flow", "miner"],
    "Key Figures": ["executive", "founder", "ceo", "resignation"],
    "Market Performance": ["market cap", "sector", "dominance", "volume", "etf", "institutional", "price", "surge", "rally", "crash"],
    "TradFi Crossover": ["stock", "correlation", "bond", "commodity", "stablecoin", "traditional finance"],
    "Partnership & Adoption": ["partnership", "adoption", "payment", "integration", "alliance", "initiative", "strategic"],
    "Risk & Warning": ["security", "hack", "breach", "rug pull", "scam", "risk", "insolvency", "exploit", "vulnerability"],
  };

  for (const [group, keywords] of Object.entries(keywordMap)) {
    for (const keyword of keywords) {
      if (lower.includes(keyword)) {
        return group;
      }
    }
  }
  
  return null;
}

// ----------------------------------------------------------------
// Core classify logic
// ----------------------------------------------------------------

async function classifyFactor(factor: string): Promise<ClassifyResponse> {
  // 1. Exact match từ taxonomy
  const exact = getFactorGroup(factor);
  if (exact) return { factor, group: exact, source: "exact" };

  // 2. In-memory cache (LLM result từ lần trước)
  const cached = classifyCache.get(factor);
  if (cached) return { factor, group: cached, source: "cache" };

  // 3. LLM fallback (Google Gemini)
  try {
    const prompt = `Which of the following groups best fits this crypto market factor?\nJust return the group name. If none fits, return Unknown.\n\nFactor: "${factor}"\n\nGroups:\n${GROUPS.map((g, i) => `${i + 1}. ${g}`).join("\n")}`;
    const response = await fetch(`https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }]
        })
      }
    );
    
    if (!response.ok) {
      console.error(`[Gemini API error] Status: ${response.status}, Message: ${response.statusText}`);
      const errorBody = await response.text();
      console.error(`[Gemini API response body]:`, errorBody);
      
      // Fallback 1: Fuzzy matching (tốt nhất)
      const fuzzyMatch = classifyByFuzzyMatching(factor);
      if (fuzzyMatch) {
        classifyCache.set(factor, fuzzyMatch);
        return { factor, group: fuzzyMatch, source: "fuzzy-match" };
      }
      
      // Fallback 2: Keyword matching
      const keywordMatch = classifyByKeyword(factor);
      if (keywordMatch) {
        classifyCache.set(factor, keywordMatch);
        return { factor, group: keywordMatch, source: "keyword-match" };
      }
      
      return { factor, group: null, source: "llm-gemini-error" };
    }
    
    const data = await response.json();
    
    // Debug: Log full response để debug
    console.log("[Gemini raw response]", JSON.stringify(data));
    
    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";
    console.log("[Gemini parsed text]", JSON.stringify(raw));
    // So sánh chính xác, không chỉ includes
    const matched = GROUPS.find((g) => raw === g);
    if (matched) {
      classifyCache.set(factor, matched);
      return { factor, group: matched, source: "llm-gemini" };
    }
    return { factor, group: null, source: "llm-gemini-unknown" };
  } catch (e) {
    console.error("[Classify error]", e);
    
    // Fallback 1: Fuzzy matching (tốt nhất)
    const fuzzyMatch = classifyByFuzzyMatching(factor);
    if (fuzzyMatch) {
      classifyCache.set(factor, fuzzyMatch);
      return { factor, group: fuzzyMatch, source: "fuzzy-match" };
    }
    
    // Fallback 2: Keyword matching
    const keywordMatch = classifyByKeyword(factor);
    if (keywordMatch) {
      classifyCache.set(factor, keywordMatch);
      return { factor, group: keywordMatch, source: "keyword-match" };
    }
    
    return { factor, group: null, source: "llm-gemini-error" };
  }
}

// ----------------------------------------------------------------
// HTTP Server (Node built-in, không cần Express)
// ----------------------------------------------------------------

import { createServer, IncomingMessage, ServerResponse } from "http";

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => res(body));
  });
}

function json(res: ServerResponse, code: number, data: unknown) {
  const body = JSON.stringify(data);
  res.writeHead(code, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

const server = createServer(async (req, res) => {
  const url = req.url ?? "";
  const method = req.method ?? "";

  // Health check
  if (method === "GET" && url === "/health") {
    return json(res, 200, { status: "ok", service: "classify-service", port: PORT });
  }

  // POST /classify — single factor
  if (method === "POST" && url === "/classify") {
    const body = JSON.parse(await readBody(req));
    if (!body.factor) return json(res, 400, { error: "factor required" });
    const result = await classifyFactor(body.factor);
    return json(res, 200, result);
  }

  // POST /classify/batch — multiple factors
  if (method === "POST" && url === "/classify/batch") {
    const body = JSON.parse(await readBody(req));
    if (!Array.isArray(body.factors))
      return json(res, 400, { error: "factors[] required" });

    const results = await Promise.all(
      body.factors.map((f: string) => classifyFactor(f))
    );
    return json(res, 200, { results });
  }

  json(res, 404, { error: "not found" });
});

server.listen(PORT, () => {
  console.log(`[classify-service] listening on :${PORT}`);
});
