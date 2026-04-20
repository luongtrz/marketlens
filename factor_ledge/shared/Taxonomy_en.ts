 /**
 * Taxonomy theo paper StockMem (2512.02720, Appendix A)
 * Adapted cho crypto market tu bitcoin-stockmem-ts/Taxonomy_en.md
 *
 * 13 groups, 62 event types.
 * Moi factor string duoc map sang 1 event type.
 * Nhieu factors khac nhau co the cung map sang 1 type (consolidation).
 */

// ----------------------------------------------------------------
// Event Taxonomy: Group -> Types
// ----------------------------------------------------------------

export const EVENT_TAXONOMY: Record<string, string[]> = {
  "Regulation & Legal": [
    "Regulatory Announcement",
    "Enforcement Action",
    "Legislation Progress",
    "Government Stance",
    "International Sanctions or Bans",
  ],
  "Macroeconomic": [
    "Interest Rate Decision",
    "Inflation Data",
    "Dollar Index Movement",
    "Quantitative Easing or Tightening",
  ],
  "Industry Standards & Opinions": [
    "Protocol Proposal",
    "Industry Report",
    "Analyst or Influencer Opinion",
  ],
  "Protocol & Product": [
    "Protocol Upgrade",
    "New Feature Launch",
    "Testnet or Mainnet Launch",
    "Adoption Metric Change",
    "Fee or Gas Change",
    "Hash Rate Change",
    "Supply Dynamics",
  ],
  "Technology & Development": [
    "Technical Breakthrough",
    "Development Milestone",
    "Audit or Certification",
    "Node or Validator Update",
    "Ecosystem Integration",
    "Developer Tooling",
  ],
  "Exchange & Trading": [
    "Listing or Delisting",
    "Funding Round",
    "Revenue Report",
    "Acquisition",
    "Partnership Deal",
    "Custody Agreement",
    "Liquidation Event",
    "Reserve Proof",
  ],
  "DeFi & Ecosystem": [
    "Protocol Launch",
    "Protocol Migration",
    "Cross-chain Expansion",
  ],
  "Whale & On-chain": [
    "Whale Accumulation",
    "Whale Distribution",
    "On-chain Flow Anomaly",
    "Miner Selling",
  ],
  "Key Figures": [
    "Executive Appointment",
    "Founder Statement",
    "Legal Action Against Individual",
  ],
  "Market Performance": [
    "Market Cap Milestone",
    "Sector Rotation",
    "BTC Dominance Shift",
    "Volume Surge",
    "ETF Flow",
    "Institutional View",
  ],
  "TradFi Crossover": [
    "Stock Correlation",
    "Bond Signal",
    "Commodity Correlation",
    "Stablecoin Flow",
  ],
  "Partnership & Adoption": [
    "Strategic Partnership",
    "Payment Integration",
    "Institutional Adoption",
    "Alliance Formation",
  ],
  "Risk & Warning": [
    "Security Breach or Hack",
    "Rug Pull or Scam",
    "Regulatory Risk",
    "Systemic Risk",
    "Exchange Insolvency",
  ],
};

// ----------------------------------------------------------------
// Derived constants
// ----------------------------------------------------------------

export const GROUPS = Object.keys(EVENT_TAXONOMY);
export const NUM_GROUPS = GROUPS.length; // 13

export const ALL_TYPES: string[] = [];
for (const types of Object.values(EVENT_TAXONOMY)) {
  ALL_TYPES.push(...types);
}
export const NUM_TYPES = ALL_TYPES.length; // 62

export const GROUP_INDEX = new Map<string, number>(
  GROUPS.map((g, i) => [g, i])
);

export const TYPE_INDEX = new Map<string, number>(
  ALL_TYPES.map((t, i) => [t, i])
);

export const TYPE_TO_GROUP = new Map<string, string>();
for (const [group, types] of Object.entries(EVENT_TAXONOMY)) {
  for (const t of types) {
    TYPE_TO_GROUP.set(t, group);
  }
}

// ----------------------------------------------------------------
// Factor -> Event Type mapping (English)
// ----------------------------------------------------------------

const FACTOR_TYPE_MAP: Record<string, string> = {
  // === BULLISH ===
  "SEC reviewing new ETF approval": "Regulatory Announcement",
  "Strong whale accumulation": "Whale Accumulation",
  "CPI lower than expected": "Inflation Data",
  "Fed holds interest rate steady": "Interest Rate Decision",
  "BlackRock increases BTC holdings": "Institutional Adoption",
  "Major corporation accepts BTC payments": "Payment Integration",
  "Hash rate hits new all-time high": "Hash Rate Change",
  "Institutional adoption increasing": "Institutional Adoption",
  "Record ETF inflows": "ETF Flow",
  "Gold positively correlated with BTC": "Commodity Correlation",
  "Stablecoin inflows to exchanges rising": "Stablecoin Flow",
  "Partnership with major bank": "Strategic Partnership",
  "Successful protocol upgrade": "Protocol Upgrade",
  "Significant volume surge": "Volume Surge",
  "Developer activity surging": "Development Milestone",
  "Positive on-chain metrics": "On-chain Flow Anomaly",
  "Supply decreasing due to halving effect": "Supply Dynamics",
  "DXY dollar index declining": "Dollar Index Movement",
  "BTC dominance rising": "BTC Dominance Shift",
  "New payment integration": "Payment Integration",
  "Grayscale GBTC premium rising": "ETF Flow",
  "MicroStrategy buys more BTC": "Institutional Adoption",
  "El Salvador increases BTC reserves": "Government Stance",
  "Lightning Network adoption growing": "Adoption Metric Change",
  "DeFi TVL on Bitcoin rising": "Protocol Launch",
  "Fidelity opens BTC custody service": "Custody Agreement",
  "JP Morgan positive outlook on BTC": "Analyst or Influencer Opinion",
  "Hash rate recovering after sell-off": "Hash Rate Change",
  "Binance Proof of Reserve stable": "Reserve Proof",
  "Bitcoin spot volume hits record": "Volume Surge",
  "Mining difficulty adjustment decreasing": "Hash Rate Change",
  "Central bank record gold buying - bullish for BTC": "Commodity Correlation",
  "Nasdaq positively correlated with crypto": "Stock Correlation",
  "New Layer 2 scaling solution": "Technical Breakthrough",
  "Fed pivot signal - market expects rate cut": "Interest Rate Decision",
  "US Treasury yield falling - capital flows to risk assets": "Bond Signal",
  "Ordinals and BRC-20 adoption growing": "Adoption Metric Change",
  "Bitcoin ETF options approved": "Regulatory Announcement",
  "Tether minting USDT - liquidity flowing in": "Stablecoin Flow",
  "Coinbase revenue beats expectations": "Revenue Report",

  // === BEARISH ===
  "SEC rejects new ETF": "Enforcement Action",
  "Strong whale selling": "Whale Distribution",
  "CPI higher than expected": "Inflation Data",
  "Fed raises interest rate": "Interest Rate Decision",
  "Regulatory concerns from China": "Government Stance",
  "Major exchange hack": "Security Breach or Hack",
  "Miner selling pressure increasing": "Miner Selling",
  "Large market liquidations": "Liquidation Event",
  "Significant ETF outflows": "ETF Flow",
  "Stablecoin outflows from exchanges": "Stablecoin Flow",
  "Regulatory risk from EU": "Regulatory Risk",
  "Exchange insolvency concerns": "Exchange Insolvency",
  "Extreme greed index - correction risk": "Institutional View",
  "Dollar index surging": "Dollar Index Movement",
  "Bond yield rising - risk for crypto": "Bond Signal",
  "Volume declining - liquidity drying up": "Volume Surge",
  "BTC dominance declining": "BTC Dominance Shift",
  "Systemic risk concerns": "Systemic Risk",
  "Major project rug pull": "Rug Pull or Scam",
  "Legal action against founder": "Legal Action Against Individual",
  "FTX exchange collapse event": "Exchange Insolvency",
  "Terra Luna collapse impact": "Systemic Risk",
  "Celsius Network freezes withdrawals": "Exchange Insolvency",
  "Genesis Trading halts operations": "Exchange Insolvency",
  "Three Arrows Capital bankruptcy": "Systemic Risk",
  "USDC temporary depeg": "Systemic Risk",
  "Mt. Gox distributing BTC to creditors": "Whale Distribution",
  "SEC sues Binance and Coinbase": "Enforcement Action",
  "Silvergate Bank closes": "Exchange Insolvency",
  "Tether FUD - reserve concerns": "Regulatory Risk",
  "Mining ban in Kazakhstan": "International Sanctions or Bans",
  "China tightens crypto regulations again": "International Sanctions or Bans",
  "Iran temporary mining ban": "International Sanctions or Bans",
  "US debt ceiling concerns": "Quantitative Easing or Tightening",
  "Grayscale GBTC discount widening": "ETF Flow",
  "Leverage ratio too high - cascade liquidation risk": "Liquidation Event",
  "Dormant BTC wallet suddenly moves funds": "On-chain Flow Anomaly",
  "SEC investigates staking services": "Enforcement Action",
  "CBDC competing with crypto": "Government Stance",
  "Whale sends large BTC to exchange": "Whale Distribution",

  // === NEUTRAL ===
  "Market sideways waiting for signal": "Market Cap Milestone",
  "Analyst opinions divided": "Analyst or Influencer Opinion",
  "Neutral industry report": "Industry Report",
  "Protocol proposal under review": "Protocol Proposal",
  "Routine developer milestone": "Development Milestone",
  "Minor sector rotation": "Sector Rotation",
  "Market cap stable": "Market Cap Milestone",
  "Normal on-chain flow": "On-chain Flow Anomaly",
  "New testnet under testing": "Testnet or Mainnet Launch",
  "Industry report compilation": "Industry Report",
  "Consolidation phase - accumulation": "Market Cap Milestone",
  "Neutral funding rate": "Liquidation Event",
  "Open interest stable": "Liquidation Event",
  "Hashrate unchanged": "Hash Rate Change",
  "DXY sideways": "Dollar Index Movement",
  "No notable macro data": "Inflation Data",
  "Weekend options expiry": "Listing or Delisting",
  "Bitcoin Pizza Day - no price impact": "Market Cap Milestone",
  "Crypto conference in Europe": "Alliance Formation",
  "US Congress crypto hearing": "Legislation Progress",
};

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

export function getFactorType(factor: string): string | undefined {
  return FACTOR_TYPE_MAP[factor];
}

export function getFactorGroup(factor: string): string | undefined {
  const type = FACTOR_TYPE_MAP[factor];
  if (!type) return undefined;
  return TYPE_TO_GROUP.get(type);
}

/**
 * Cong thuc (3): V_t[m] = 1 neu co event type m.
 */
export function buildTypeVector(factors: string[]): number[] {
  const vec = new Array(NUM_TYPES).fill(0);
  for (const f of factors) {
    const type = FACTOR_TYPE_MAP[f];
    if (type) {
      const idx = TYPE_INDEX.get(type);
      if (idx !== undefined) vec[idx] = 1;
    }
  }
  return vec;
}

/**
 * Cong thuc (4): G_t[g] = 1 neu group g co event.
 */
export function buildGroupVector(factors: string[]): number[] {
  const vec = new Array(NUM_GROUPS).fill(0);
  for (const f of factors) {
    const group = getFactorGroup(f);
    if (group) {
      const idx = GROUP_INDEX.get(group);
      if (idx !== undefined) vec[idx] = 1;
    }
  }
  return vec;
}