-- Migration: create per-model backtest tables
-- Run in Supabase SQL editor:
-- https://supabase.com/dashboard/project/esctepjpgpjgrcymnabx/sql

CREATE TABLE IF NOT EXISTS bt_qwen (
    id          BIGSERIAL PRIMARY KEY,
    backtest_date DATE NOT NULL UNIQUE,
    signal      TEXT NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    confidence  FLOAT NOT NULL,
    ret_1d      FLOAT,
    ret_7d      FLOAT,
    ret_30d     FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bt_kimi (
    id          BIGSERIAL PRIMARY KEY,
    backtest_date DATE NOT NULL UNIQUE,
    signal      TEXT NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    confidence  FLOAT NOT NULL,
    ret_1d      FLOAT,
    ret_7d      FLOAT,
    ret_30d     FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bt_deepseek (
    id          BIGSERIAL PRIMARY KEY,
    backtest_date DATE NOT NULL UNIQUE,
    signal      TEXT NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    confidence  FLOAT NOT NULL,
    ret_1d      FLOAT,
    ret_7d      FLOAT,
    ret_30d     FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Backfill bt_deepseek from existing backtest_results table
INSERT INTO bt_deepseek (backtest_date, signal, confidence, ret_1d, ret_7d, ret_30d, created_at)
SELECT
    backtest_date,
    signal,
    confidence,
    actual_return_1d,
    actual_return_7d,
    actual_return_30d,
    created_at
FROM backtest_results
WHERE symbol = 'BTC'
ON CONFLICT (backtest_date) DO NOTHING;
