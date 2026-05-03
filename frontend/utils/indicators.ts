
import { HistoryPoint } from '../types';

export const calculateSMA = (data: number[], period: number): number[] => {
    const sma = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            sma.push(NaN);
            continue;
        }
        const slice = data.slice(i - period + 1, i + 1);
        const sum = slice.reduce((a, b) => a + b, 0);
        sma.push(sum / period);
    }
    return sma;
};

export const calculateEMA = (data: number[], period: number): number[] => {
    const k = 2 / (period + 1);
    const ema = [];
    ema[0] = data[0];
    for (let i = 1; i < data.length; i++) {
        ema[i] = data[i] * k + ema[i - 1] * (1 - k);
    }
    return ema;
};

export const calculateRSI = (data: number[], period: number = 14): { time: number, value: number }[] => {
    // Need at least period + 1 data points
    if (data.length <= period) return [];

    let gains = 0;
    let losses = 0;

    // First RSI (Average Gain/Loss)
    for (let i = 1; i <= period; i++) {
        const change = data[i] - data[i - 1];
        if (change > 0) gains += change;
        else losses += Math.abs(change);
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;

    const rsiValues = [];

    // Push first valid RSI
    // To match timestamps, we need to handle the offset. 
    // This function assumes 'data' corresponds to indices. 
    // We will handle timestamps in the main loop or return aligned array.

    // We'll return simple array first matching the input index
    // Pre-fill NaN for first 'period' elements
    for (let i = 0; i < period; i++) rsiValues.push(NaN);

    let rs = avgGain / avgLoss;
    rsiValues.push(100 - (100 / (1 + rs)));

    for (let i = period + 1; i < data.length; i++) {
        const change = data[i] - data[i - 1];
        let gain = change > 0 ? change : 0;
        let loss = change < 0 ? Math.abs(change) : 0;

        avgGain = (avgGain * (period - 1) + gain) / period;
        avgLoss = (avgLoss * (period - 1) + loss) / period;

        if (avgLoss === 0) {
            rsiValues.push(100);
        } else {
            rs = avgGain / avgLoss;
            rsiValues.push(100 - (100 / (1 + rs)));
        }
    }

    return rsiValues.map((val, idx) => ({ time: idx, value: val }));
};

// Returns aligned objects with Time
export const calculateBollingerBands = (data: HistoryPoint[], period: number = 20, multiplier: number = 2) => {
    const closes = data.map(d => d.price);
    const sma = calculateSMA(closes, period);

    const bands = [];

    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            continue;
        }

        const slice = closes.slice(i - period + 1, i + 1);
        const mean = sma[i];

        // Standard Deviation
        const squaredDiffs = slice.map(val => Math.pow(val - mean, 2));
        const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
        const stdDev = Math.sqrt(variance);

        bands.push({
            time: Math.floor(data[i].ts / 1000),
            upper: mean + (multiplier * stdDev),
            lower: mean - (multiplier * stdDev),
            middle: mean
        });
    }
    return bands;
};

export const calculateMACD = (data: HistoryPoint[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9) => {
    const closes = data.map(d => d.price);

    // Calculate EMAs
    // Note: calculateEMA needs to handle the initial ramp up correctly or we just ignore first few
    // Ideally we use a library, but for this simplified version:

    // We need to compute Fast and Slow from the START
    const emaFast = calculateEMA(closes, fastPeriod);
    const emaSlow = calculateEMA(closes, slowPeriod);

    const macdLine = [];
    for (let i = 0; i < closes.length; i++) {
        macdLine.push(emaFast[i] - emaSlow[i]);
    }

    // Signal Line is EMA of MACD Line
    const signalLine = calculateEMA(macdLine, signalPeriod);

    const result = [];
    for (let i = 0; i < data.length; i++) {
        // Skip unreliable data at start
        if (i < slowPeriod + signalPeriod) continue;

        result.push({
            time: Math.floor(data[i].ts / 1000),
            macd: macdLine[i],
            signal: signalLine[i],
            histogram: macdLine[i] - signalLine[i]
        });
    }

    return result;
};

// Re-implementing RSI to be cleaner and accept HistoryPoint
export const getRSIValues = (data: HistoryPoint[], period: number = 14) => {
    const closes = data.map(d => d.price);
    const rsiRaw = calculateRSI(closes, period);

    const result = [];
    for (let i = 0; i < data.length; i++) {
        if (isNaN(rsiRaw[i]?.value)) continue;
        result.push({
            time: Math.floor(data[i].ts / 1000),
            value: rsiRaw[i].value
        });
    }
    return result;
};
