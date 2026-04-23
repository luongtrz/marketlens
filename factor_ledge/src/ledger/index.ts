/**
 * ledger-service — port 3002
 *
 * Nhận DailyRecord[], gọi classify-service để map factors (cả group & type),
 * aggregate theo rolling window, gọi LLM summarize (bulk), lưu ledger.
 */

import { createServer, IncomingMessage, ServerResponse } from "http";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import fetch from "node-fetch";
import type {
  DailyRecord,
  FactorLedger,
  LedgerSnapshot,
} from "../../shared/types.ts";
import { GROUPS } from "../../shared/Taxonomy_en.ts";

function loadEnvFile(): void {
  const envPath = join(process.cwd(), ".env"); 
  
  if (!existsSync(envPath)) {
    console.warn("[ledger] Không tìm thấy file .env tại", envPath);
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
  console.log("[ledger] Đã nạp biến môi trường thành công!");
}

loadEnvFile();

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const PORT = Number(process.env.LEDGER_PORT ?? 3002);
const CLASSIFY_URL = process.env.CLASSIFY_URL ?? "http://localhost:3001";
const SUMMARY_THRESHOLD = 0.05;

// ----------------------------------------------------------------
// In-memory store
// Production: Redis HSET "ledger:current" field value
// ----------------------------------------------------------------

let currentLedger: FactorLedger | null = null;
let lastUpdated: string | null = null;
let currentWindowDays: number = 7;

// ----------------------------------------------------------------
// Step 1: Aggregate
// ----------------------------------------------------------------

async function aggregateLedger(
  records: DailyRecord[],
  windowDays: number
): Promise<FactorLedger> {
  const ledger: FactorLedger = {};
  for (const g of GROUPS) {
    ledger[g] = {
      group: g,
      activeTypes: [],
      rawFactors: [],
      hitCount: 0,
      weight: 0,
      summary: "",
      updatedAt: new Date().toISOString(),
    };
  }

  const windowFactors = records
    .slice(-windowDays)
    .flatMap((r) => r.factors);

  if (windowFactors.length === 0) return ledger;

  try {
    const res = await fetch(`${CLASSIFY_URL}/classify/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ factors: windowFactors }),
    });

    if (res.ok) {
      const { results } = (await res.json()) as {
        results: Array<{ factor: string; group: string | null; type: string | null }>;
      };

      for (const { factor, group, type } of results) {
        if (!group || !ledger[group]) continue;

        ledger[group].hitCount++;
        ledger[group].rawFactors.push(factor);

        if (type && !ledger[group].activeTypes.includes(type)) {
          ledger[group].activeTypes.push(type);
        }
      }
    } else {
      console.error(`[ledger] classify-service error: ${res.status}`);
    }
  } catch (e) {
    console.error("[ledger] classify-service call failed:", e);
  }

  const total = Object.values(ledger).reduce((s, e) => s + e.hitCount, 0);
  if (total > 0) {
    for (const entry of Object.values(ledger)) {
      entry.weight = parseFloat((entry.hitCount / total).toFixed(4));
    }
  }

  return ledger;
}

// ----------------------------------------------------------------
// Step 2: Summarize (Bulk Gemini API)
// ----------------------------------------------------------------

async function summarizeLedger(ledger: FactorLedger): Promise<void> {
  const activeEntries = Object.values(ledger).filter(
    (e) => e.weight >= SUMMARY_THRESHOLD
  );

  if (!activeEntries.length) return;

  console.log(`[ledger] bulk summarizing ${activeEntries.length} active groups...`);

  const payloadToSummarize = activeEntries
    .map((e) => `Group: ${e.group}\nFactors: ${e.rawFactors.slice(0, 5).join(", ")}`)
    .join("\n\n");

  const prompt = `Summarize the following crypto market signals. For each group, write 1-2 concise sentences summarizing the factors.
Return ONLY a valid JSON object where the key is the exact Group name and the value is the summary string.

Data:
${payloadToSummarize}`;

  let usedLLM = false;

  if (GROQ_API_KEY) {
    try {
      const response = await fetch(
        "https://api.groq.com/openai/v1/chat/completions",
        {
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
          }),
        }
      );

      if (response.ok) {
        const data = (await response.json()) as any;
        const rawText = data.choices?.[0]?.message?.content?.trim() || "{}";
        const summaries = JSON.parse(rawText) as Record<string, string>;

        for (const entry of activeEntries) {
          if (summaries[entry.group]) {
            entry.summary = summaries[entry.group];
          } else {
            entry.summary = `${entry.hitCount} signals: ${entry.rawFactors
              .slice(0, 3)
              .join(", ")}${entry.rawFactors.length > 3 ? "..." : ""}`;
          }
        }
        usedLLM = true;
      } else {
        const errorBody = await response.text();
        console.error(`[ledger] Groq Error: ${response.status}`, errorBody);
        console.warn(`[ledger] Groq bulk summarize error: ${response.status}`);
      }
    } catch (e) {
      console.error(`[ledger] Groq call failed:`, e);
    }
  }

  if (!usedLLM) {
    for (const entry of activeEntries) {
      entry.summary = `${entry.hitCount} signals in ${entry.group}: ${entry.rawFactors
        .slice(0, 3)
        .join(", ")}${entry.rawFactors.length > 3 ? "..." : ""}`;
    }
  }
}

// ----------------------------------------------------------------
// HTTP Server
// ----------------------------------------------------------------

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => res(body));
  });
}

function json(res: ServerResponse, code: number, data: unknown) {
  const body = JSON.stringify(data, null, 2);
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(body);
}

const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  const url = req.url ?? "";
  const method = req.method ?? "";

  if (method === "GET" && url === "/health") {
    return json(res, 200, {
      status: "ok",
      service: "ledger-service",
      port: PORT,
    });
  }

  if (method === "POST" && url === "/ledger/update") {
    try {
      const bodyStr = await readBody(req);
      const body = JSON.parse(bodyStr);
      const records: DailyRecord[] = body.records;
      const windowDays: number = body.windowDays ?? 7;

      if (!Array.isArray(records)) {
        return json(res, 400, { error: "records[] required" });
      }

      console.log(
        `[ledger-service] aggregating ${records.length} records, window=${windowDays}d`
      );
      const ledger = await aggregateLedger(records, windowDays);

      await summarizeLedger(ledger);

      currentLedger = ledger;
      currentWindowDays = windowDays;
      lastUpdated = new Date().toISOString();

      const activeCount = Object.values(ledger).filter(
        (e) => e.weight >= SUMMARY_THRESHOLD
      ).length;
      console.log(`[ledger-service] done — ${activeCount} active groups`);

      return json(res, 200, {
        message: "ledger updated",
        activeGroups: activeCount,
        lastUpdated,
      });
    } catch (e) {
      console.error("[ledger] POST /ledger/update error:", e);
      return json(res, 400, { error: "invalid request" });
    }
  }

  if (method === "GET" && url === "/ledger/current") {
    if (!currentLedger) {
      return json(res, 404, { error: "ledger not built yet" });
    }
    return json(res, 200, currentLedger);
  }

  if (method === "GET" && url === "/ledger/snapshot") {
    if (!currentLedger) {
      return json(res, 404, { error: "ledger not built yet" });
    }

    const snapshot: LedgerSnapshot = {
      snapshotDate: lastUpdated ?? new Date().toISOString(),
      windowDays: currentWindowDays,
      totalGroups: GROUPS.length,
      activeGroups: Object.values(currentLedger).filter(
        (e) => e.weight >= SUMMARY_THRESHOLD
      ).length,
      entries: currentLedger,
    };
    return json(res, 200, snapshot);
  }

  json(res, 404, { error: "not found" });
});

server.listen(PORT, () => {
  console.log(`[ledger-service] listening on :${PORT}`);
});
