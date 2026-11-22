-- Decision log schema (SQLite) - created automatically by src/logging_store.py.
-- Follows the Governance Framework's minimum record (section 1): one row per decision per stage.
-- Serves FR-10, NFR-05, acceptance criterion A8. Customer text is never stored (input_summary = channel, length, hash).

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    input_file TEXT NOT NULL,
    tickets_in_file INTEGER NOT NULL,   -- for reconciliation
    config_json TEXT NOT NULL,          -- thresholds, model versions, kill switch state
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    timestamp TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    stage TEXT NOT NULL,                -- classification | routing | generation | validation
    input_summary TEXT,
    model_name TEXT,
    model_version TEXT,
    prediction_value TEXT,
    prediction_confidence REAL,
    alternatives TEXT,                  -- JSON [{value, confidence}]
    sources_used TEXT,                  -- JSON [{doc_id, score, chunk}]
    threshold_applied REAL,
    action_taken TEXT NOT NULL,         -- auto_respond | escalate | block | continue (passed to next stage)
    rule TEXT,                          -- which routing rule decided
    reason TEXT NOT NULL,               -- plain English, readable by a support manager
    guardrail_results TEXT,             -- JSON {check: result}
    prompt_version TEXT,                -- e.g. "PR-02 v1.0"
    requirement_ids TEXT,               -- JSON ["FR-04", ...]
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_run_ticket ON decisions(run_id, ticket_id);
CREATE INDEX IF NOT EXISTS idx_decisions_run_stage ON decisions(run_id, stage);

-- Reconciliation (A8): every ticket in the run has a routing decision
-- SELECT r.tickets_in_file, COUNT(DISTINCT d.ticket_id)
-- FROM runs r LEFT JOIN decisions d ON d.run_id = r.run_id AND d.stage = 'routing'
-- WHERE r.run_id = :run_id GROUP BY r.run_id;
