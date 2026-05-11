-- ============================================================
-- Supabase: backtest_results table
-- Stores prediction + ground truth for each (symbol, date)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.backtest_results (
    id BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
    symbol TEXT NOT NULL,
    backtest_date DATE NOT NULL,
    run_id TEXT,
    signal TEXT CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    confidence DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    sentiment_score DOUBLE PRECISION,
    actual_return_1d DOUBLE PRECISION,
    actual_return_7d DOUBLE PRECISION,
    actual_return_30d DOUBLE PRECISION,
    correct_1d BOOLEAN,
    correct_7d BOOLEAN,
    correct_30d BOOLEAN,
    n_similar_cases INTEGER,
    top_similar_date DATE,
    top_similar_similarity DOUBLE PRECISION,
    top_similar_ret7d DOUBLE PRECISION,
    explanation TEXT,
    errors JSONB DEFAULT '[]'::jsonb,
    run_duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT backtest_results_pkey PRIMARY KEY (id),
    CONSTRAINT backtest_results_symbol_date_key UNIQUE (symbol, backtest_date)
) TABLESPACE pg_default;

-- Index for fast lookup by symbol + date
CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol_date
    ON public.backtest_results USING btree (symbol, backtest_date DESC)
    TABLESPACE pg_default;

-- Index for filtering rows that need re-testing (missing signal)
CREATE INDEX IF NOT EXISTS idx_backtest_results_pending
    ON public.backtest_results USING btree (symbol, backtest_date DESC)
    WHERE signal IS NULL
    TABLESPACE pg_default;
