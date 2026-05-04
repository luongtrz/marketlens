// ----------------------------------------------------------------
// Shared types — dùng chung cho tất cả services
// ----------------------------------------------------------------

export interface DailyRecord {
  date: string;        // "YYYY-MM-DD"
  factors: string[];   // raw strings từ LLM extraction
}

export interface LedgerEntry {
  group: string;
  activeTypes: string[];
  rawFactors: string[];
  hitCount: number;
  weight: number;      // normalized 0..1
  summary: string;
  updatedAt: string;
}

export type FactorLedger = Record<string, LedgerEntry>;

// HTTP response shapes
export interface ClassifyResponse {
  factor: string;
  group: string | null;
  source: "exact" | "cache" | "llm" | "unknown" | "llm-gemini" | "llm-gemini-unknown" | "llm-gemini-error" | "fuzzy-match" | "keyword-match";
}

export interface LedgerSnapshot {
  snapshotDate: string;
  windowDays: number;
  totalGroups: number;
  activeGroups: number;
  entries: FactorLedger;
}

export interface WeightVectorResponse {
  groups: string[];
  weights: number[];   // 13-dim, sum = 1.0
}