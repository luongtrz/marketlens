/**
 * classify-service — port 3001
 *
 * Nhận raw factor string, trả về group name và event type.
 * Thứ tự ưu tiên: exact → cache → keyword → fuzzy → LLM
 *
 * POST /classify        { factor: string }
 * POST /classify/batch  { factors: string[] }
 * GET  /health
 */

import "dotenv/config";
import fetch from "node-fetch";
import {
  getFactorGroup,
  getFactorType,
  GROUPS,
  EVENT_TAXONOMY,
} from "../../../shared/Taxonomy_en.ts";
import { createServer, IncomingMessage, ServerResponse } from "http";

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const PORT = Number(process.env.CLASSIFY_PORT ?? 3001);

const classifyCache = new Map<string, { group: string; type: string | null }>();

function calculateSimilarity(str1: string, str2: string): number {
  const s1 = str1.toLowerCase();
  const s2 = str2.toLowerCase();
  if (s1 === s2) return 1.0;

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

function classifyByFuzzyMatching(
  factor: string
): { group: string; type: string | null } | null {
  let bestGroup: string | null = null;
  let bestType: string | null = null;
  let bestScore = 0.3;

  for (const [group, types] of Object.entries(EVENT_TAXONOMY)) {
    for (const eventType of types) {
      const score = calculateSimilarity(factor, eventType);
      if (score > bestScore) {
        bestScore = score;
        bestGroup = group;
        bestType = eventType;
      }
    }

    const groupScore = calculateSimilarity(factor, group);
    if (groupScore > bestScore) {
      bestScore = groupScore;
      bestGroup = group;
      bestType = null;
    }
  }

  return bestGroup ? { group: bestGroup, type: bestType } : null;
}

function classifyByKeyword(
  factor: string
): { group: string; type: string | null } | null {
  const lower = factor.toLowerCase();
  const hasKeyword = (text: string, keyword: string): boolean => {
    const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const phrasePattern = escaped.replace(/\s+/g, "\\s+");
    return new RegExp(`\\b${phrasePattern}\\b`, "i").test(text);
  };

  const keywordMap: Record<string, string[]> = {
    "Regulation & Legal": [
      "regulation",
      "legal",
      "government",
      "law",
      "enforcement",
      "legislative",
      "ban",
      "sanction",
      "compliance",
      "sec",
      "regulatory",
    ],
    Macroeconomic: [
      "interest rate",
      "inflation",
      "fed",
      "cpi",
      "economic",
      "gdp",
      "dollar",
      "quantitative",
      "monetary",
      "recession",
    ],
    "Industry Standards & Opinions": [
      "protocol proposal",
      "industry",
      "report",
      "opinion",
      "analyst",
      "influencer",
    ],
    "Protocol & Product": [
      "protocol",
      "upgrade",
      "feature",
      "launch",
      "testnet",
      "mainnet",
      "adoption",
      "fee",
      "gas",
      "hash rate",
      "supply",
      "halving",
    ],
    "Technology & Development": [
      "technical",
      "breakthrough",
      "development",
      "audit",
      "certification",
      "node",
      "validator",
      "ecosystem",
      "integration",
      "tooling",
      "update",
    ],
    "Exchange & Trading": [
      "exchange",
      "listing",
      "delisting",
      "funding",
      "revenue",
      "acquisition",
      "partnership",
      "custody",
      "liquidation",
      "reserve",
      "trading",
    ],
    "DeFi & Ecosystem": ["defi", "protocol launch", "migration", "cross-chain", "yield"],
    "Whale & On-chain": ["whale", "accumulation", "distribution", "on-chain", "flow", "miner"],
    "Key Figures": ["executive", "founder", "ceo", "resignation"],
    "Market Performance": [
      "market cap",
      "sector",
      "dominance",
      "volume",
      "etf",
      "institutional",
      "price",
      "surge",
      "rally",
      "crash",
    ],
    "TradFi Crossover": ["stock", "correlation", "bond", "commodity", "stablecoin", "traditional finance"],
    "Partnership & Adoption": [
      "partnership",
      "adoption",
      "payment",
      "integration",
      "alliance",
      "initiative",
      "strategic",
    ],
    "Risk & Warning": ["security", "hack", "breach", "rug pull", "scam", "risk", "insolvency", "exploit", "vulnerability"],
  };

  for (const [group, keywords] of Object.entries(keywordMap)) {
    for (const keyword of keywords) {
      if (hasKeyword(lower, keyword)) {
        return { group, type: null };
      }
    }
  }
  return null;
}

export interface ClassifyResponse {
  factor: string;
  group: string | null;
  type: string | null;
  source: string;
}

async function classifyFactor(factor: string): Promise<ClassifyResponse> {
  // const exactType = getFactorType(factor);
  // const exactGroup = getFactorGroup(factor);
  // if (exactGroup && exactType) {
  //   return { factor, group: exactGroup, type: exactType, source: "exact" };
  // }

  // const cached = classifyCache.get(factor);
  // if (cached) {
  //   return { factor, group: cached.group, type: cached.type, source: "cache" };
  // }

  // const keywordMatch = classifyByKeyword(factor);
  // if (keywordMatch) {
  //   classifyCache.set(factor, keywordMatch);
  //   return {
  //     factor,
  //     group: keywordMatch.group,
  //     type: keywordMatch.type,
  //     source: "keyword-match",
  //   };
  // }

  // const fuzzyMatch = classifyByFuzzyMatching(factor);
  // if (fuzzyMatch) {
  //   classifyCache.set(factor, fuzzyMatch);
  //   return {
  //     factor,
  //     group: fuzzyMatch.group,
  //     type: fuzzyMatch.type,
  //     source: "fuzzy-match",
  //   };
  // }

  if (!GROQ_API_KEY) {
    return { factor, group: null, type: null, source: "unknown_no_key" };
  }

  // 5. LLM Fallback (Groq - LLaMA 3)
  try {
    const prompt = `You are a financial data classifier. Classify the following crypto market factor into ONE "group" and ONE "type" based EXACTLY on this taxonomy:
${JSON.stringify(EVENT_TAXONOMY, null, 2)}

Factor: "${factor}"

Return ONLY a valid JSON object with the keys "group" and "type". If it doesn't fit anything perfectly, pick the closest match or return null for both. Do not include any markdown formatting or extra text.`;

    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" },
        temperature: 0.1
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("[Groq API Error]", response.status, errorText);
      return { factor, group: null, type: null, source: "llm-groq-error" };
    }

    const data = (await response.json()) as any;
    const rawText = data.choices[0].message.content.trim();
    const parsed = JSON.parse(rawText);

    if (parsed.group && GROUPS.includes(parsed.group)) {
      return { factor, group: parsed.group, type: parsed.type || null, source: "llm-groq" };
    }

    return { factor, group: null, type: null, source: "llm-groq-unknown" };
  } catch (e) {
    console.error("[Classify Error]", e);
    return { factor, group: null, type: null, source: "llm-groq-error" };
  }
}

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

  if (method === "GET" && url === "/health") {
    return json(res, 200, {
      status: "ok",
      service: "classify-service",
      port: PORT,
    });
  }

  if (method === "POST" && url === "/classify") {
    try {
      const body = JSON.parse(await readBody(req));
      if (!body.factor) return json(res, 400, { error: "factor required" });
      const result = await classifyFactor(body.factor);
      return json(res, 200, result);
    } catch {
      return json(res, 400, { error: "Invalid JSON" });
    }
  }

  if (method === "POST" && url === "/classify/batch") {
    try {
      const body = JSON.parse(await readBody(req));
      if (!Array.isArray(body.factors)) {
        return json(res, 400, { error: "factors[] required" });
      }

      const factors = body.factors as string[];
      const results = [];

      console.log(
        `[classify-service] Dang xu ly batch ${factors.length} factors (Free Tier mode)...`
      );

      // Run sequentially to reduce 429 on free tier.
      for (const f of factors) {
        const result = await classifyFactor(f);
        results.push(result);

        // Delay only when this factor was classified by Groq.
        if (result.source === "llm-groq") {
          await new Promise((resolve) => setTimeout(resolve, 5000));
        }
      }

      return json(res, 200, { results });
    } catch {
      return json(res, 400, { error: "Invalid JSON" });
    }
  }

  json(res, 404, { error: "not found" });
});

server.listen(PORT, () => {
  console.log(`[classify-service] listening on :${PORT}`);
});
