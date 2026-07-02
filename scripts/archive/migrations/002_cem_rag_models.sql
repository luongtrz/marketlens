-- CEM-RAG model artifact tables
-- Run in Supabase SQL editor: https://supabase.com/dashboard/project/esctepjpgpjgrcymnabx/editor

-- ── Learned retriever versions ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cem_rag_retrievers (
    id              TEXT        PRIMARY KEY,   -- e.g. "learned-diagonal-v4"
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    version         TEXT,                      -- e.g. "learned_cem_v2"
    type            TEXT,                      -- "learned_diagonal"
    dim             INT,                       -- total embedding dim (225)
    block_dims      JSONB,                     -- [85, 75, 5, 60]
    block_scales    JSONB,                     -- [w_event, w_factor, w_indicator, w_price]
    d               JSONB,                     -- 225-dim diagonal weights
    band            TEXT,                      -- "fixed" | "0.5sigma"
    splits          JSONB,                     -- {train, val, test, embargo_days}
    hyperparameters JSONB,                     -- temperature, ridge, hard_negs, lr, etc.
    mining_protocol JSONB,                     -- pair mining strategy details
    val_hit_at_k    FLOAT,
    val_combined    FLOAT,
    seed_std        FLOAT,
    seeds           JSONB,                     -- list of seed values used
    source_data     TEXT,                      -- data file used for training
    trained_at      TIMESTAMPTZ,
    notes           TEXT
);

ALTER TABLE cem_rag_retrievers ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read" ON cem_rag_retrievers FOR SELECT USING (true);
CREATE POLICY "service_all" ON cem_rag_retrievers
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- ── Policy (tau calibration) versions ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cem_rag_policies (
    id              TEXT        PRIMARY KEY,   -- e.g. "policy-v4"
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    retriever_id    TEXT        REFERENCES cem_rag_retrievers(id),
    method          TEXT,                      -- "knn_platt_v1"
    tau             FLOAT,
    k               INT,
    val_sharpe      FLOAT,
    -- Test set flat metrics (n=305)
    test_n          INT,
    test_n_buy      INT,
    test_n_sell     INT,
    test_n_hold     INT,
    test_da         FLOAT,
    test_buy_da     FLOAT,
    test_sell_da    FLOAT,
    test_coverage   FLOAT,
    test_sharpe     FLOAT,
    test_sortino    FLOAT,
    test_max_drawdown FLOAT,
    test_brier      FLOAT,
    test_ece        FLOAT,
    -- Full JSON blobs
    test_metrics    JSONB,
    val_best_metrics JSONB,
    notes           TEXT
);

ALTER TABLE cem_rag_policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_read" ON cem_rag_policies FOR SELECT USING (true);
CREATE POLICY "service_all" ON cem_rag_policies
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
