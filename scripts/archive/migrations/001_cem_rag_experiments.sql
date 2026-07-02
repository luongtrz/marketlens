-- CEM-RAG experiment tracking table
-- Run in Supabase SQL editor: https://supabase.com/dashboard/project/esctepjpgpjgrcymnabx/editor

CREATE TABLE IF NOT EXISTS cem_rag_experiments (
    id                  TEXT        PRIMARY KEY,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    run_name            TEXT        NOT NULL,
    data_version        TEXT,
    retriever_version   TEXT,
    policy_version      TEXT,
    -- Data coverage
    total_rows          INT,
    event_vec_coverage  FLOAT,      -- fraction of rows with nonzero event_vec
    -- Retriever training
    val_hit_at_k        FLOAT,
    seed_std            FLOAT,
    trials              INT,
    epochs              INT,
    seeds               INT,
    -- Test set metrics (n=305, 2025-07 to 2026-05)
    test_n              INT,
    test_n_buy          INT,
    test_n_sell         INT,
    test_n_hold         INT,
    test_da             FLOAT,
    test_buy_da         FLOAT,
    test_sell_da        FLOAT,
    test_coverage       FLOAT,
    test_sharpe         FLOAT,
    test_sortino        FLOAT,
    test_max_drawdown   FLOAT,
    test_brier          FLOAT,
    test_ece            FLOAT,
    test_combined       FLOAT,      -- 0.6*DA + 0.4*Sharpe
    test_hit_at_5       FLOAT,
    -- Policy
    policy_tau          FLOAT,
    val_sharpe          FLOAT,
    -- Significance
    mcnemar_da_p        FLOAT,
    mcnemar_hit_p       FLOAT,
    -- Notes
    notes               TEXT
);

-- Enable RLS (read-only for anon, full access for service role)
ALTER TABLE cem_rag_experiments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_read" ON cem_rag_experiments
    FOR SELECT USING (true);

CREATE POLICY "service_all" ON cem_rag_experiments
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
