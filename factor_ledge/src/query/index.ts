/**
 * query-service — port 3003
 *
 * Expose Factor Ledger dưới dạng vector và prompt context
 * cho StockMem (similarity query) và LLM (prediction context).
 *
 * GET /query/vector   → 13-dim weight vector cho StockMem
 * GET /query/factor-vector → 75-dim binary factor vector cho StockMem
 * GET /query/context  → formatted string cho LLM prompt
 * GET /query/top?k=5  → top K active groups
 * GET /health
 */

/// <reference path="./node-shim.d.ts" />

import { createServer, ServerResponse } from "http";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import type {
  FactorLedger,
  FactorVectorResponse,
  WeightVectorResponse,
} from "../../../../Services/shared/types.ts";
import { ALL_TYPES, GROUPS } from "../../../../Services/shared/Taxonomy_en.ts";

const TYPE_INDEX = new Map<string, number>(
  ALL_TYPES.map((typeName, index) => [typeName, index])
);

function loadEnvFile(): void {
  const envPath = join(process.cwd(), ".env");

  if (!existsSync(envPath)) {
    console.warn("[query] .env file not found at", envPath);
    return;
  }

  const content = readFileSync(envPath, "utf8");
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const equalsIndex = trimmed.indexOf("=");
    if (equalsIndex <= 0) continue;

    const key = trimmed.slice(0, equalsIndex).trim();
    const value = trimmed.slice(equalsIndex + 1).trim();

    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

loadEnvFile();

const PORT = Number(process.env.QUERY_PORT ?? 3003);
const LEDGER_URL = process.env.LEDGER_URL ?? "http://localhost:3002";

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

async function fetchLedger(): Promise<FactorLedger> {
  const res = await fetch(`${LEDGER_URL}/ledger/current`);
  if (!res.ok) throw new Error("ledger not available");
  return res.json() as Promise<FactorLedger>;
}

function toWeightVector(ledger: FactorLedger): WeightVectorResponse {
  return {
    groups: GROUPS,
    weights: GROUPS.map((g) => ledger[g]?.weight ?? 0),
  };
}

function toFactorVector(ledger: FactorLedger): FactorVectorResponse {
  const typeVector = new Array(ALL_TYPES.length).fill(0);
  const groupVector = new Array(GROUPS.length).fill(0);

  for (const [groupIndex, groupName] of GROUPS.entries()) {
    const entry = ledger[groupName];
    if (!entry || entry.weight <= 0) continue;

    groupVector[groupIndex] = 1;

    for (const typeName of entry.activeTypes) {
      const typeIndex = TYPE_INDEX.get(typeName);
      if (typeIndex !== undefined) {
        typeVector[typeIndex] = 1;
      }
    }
  }

  return {
    types: ALL_TYPES,
    groups: GROUPS,
    typeVector,
    groupVector,
    factorVector: [...typeVector, ...groupVector],
  };
}

function toPromptContext(ledger: FactorLedger, topK: number): string {
  const sorted = Object.values(ledger)
    .filter((e) => e.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, topK);

  if (sorted.length === 0) return "(no active factors)";

  return sorted
    .map(
      (e) =>
        `[${e.group} — ${(e.weight * 100).toFixed(1)}%] ${e.summary || "(no summary yet)"}`
    )
    .join("\n");
}

// ----------------------------------------------------------------
// HTTP Server
// ----------------------------------------------------------------

function json(res: ServerResponse, code: number, data: unknown) {
  const body = JSON.stringify(data, null, 2);
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(body);
}

function text(res: ServerResponse, code: number, content: string) {
  res.writeHead(code, { "Content-Type": "text/plain" });
  res.end(content);
}

const server = createServer(async (req: any, res: any) => {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
  const path = url.pathname;
  const method = req.method ?? "";

  if (method === "GET" && path === "/health") {
    return json(res, 200, { status: "ok", service: "query-service", port: PORT });
  }

  // GET /query/vector — 13-dim weight vector cho StockMem
  if (method === "GET" && path === "/query/vector") {
    try {
      const ledger = await fetchLedger();
      return json(res, 200, toWeightVector(ledger));
    } catch {
      return json(res, 503, { error: "ledger not ready" });
    }
  }

  // GET /query/factor-vector — 75-dim binary vector cho StockMem
  if (method === "GET" && path === "/query/factor-vector") {
    try {
      const ledger = await fetchLedger();
      return json(res, 200, toFactorVector(ledger));
    } catch {
      return json(res, 503, { error: "ledger not ready" });
    }
  }

  // GET /query/context?k=5 — text cho LLM prompt
  if (method === "GET" && path === "/query/context") {
    const k = parseInt(url.searchParams.get("k") ?? "5", 10);
    try {
      const ledger = await fetchLedger();
      return text(res, 200, toPromptContext(ledger, k));
    } catch {
      return json(res, 503, { error: "ledger not ready" });
    }
  }

  // GET /query/top?k=5 — top K active groups as JSON
  if (method === "GET" && path === "/query/top") {
    const k = parseInt(url.searchParams.get("k") ?? "5", 10);
    try {
      const ledger = await fetchLedger();
      const top = Object.values(ledger)
        .filter((e) => e.weight > 0)
        .sort((a, b) => b.weight - a.weight)
        .slice(0, k);
      return json(res, 200, { topK: k, groups: top });
    } catch {
      return json(res, 503, { error: "ledger not ready" });
    }
  }

  json(res, 404, { error: "not found" });
});

server.listen(PORT, () => {
  console.log(`[query-service] listening on :${PORT}`);
});
