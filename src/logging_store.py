"""Decision log (B-10; FR-10, acceptance criterion A8, Governance Framework section 1).

One row per decision per stage (classification, routing, generation, validation), with the
Governance Framework's minimum record: who/what decided, with which model and prompt version,
what it predicted and how confident it was, which sources it used, which threshold applied,
what action followed and why, and which requirements it served.

    log = DecisionLog()                                   # sqlite file from DATABASE_URL
    run_id = log.start_run("data/validation_tickets.json", tickets_in_file=80, config={...})
    log.record(DecisionRecord(...))                       # one per stage
    log.finish_run(run_id)
    log.reconcile(run_id)  -> {"tickets_in_file": 80, "tickets_with_final_decision": 80, "missing": []}

Privacy: the ticket text is never stored; input_summary holds channel, length and a hash.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings

STAGES = ("classification", "routing", "generation", "validation")   # Governance Framework s.1
# "continue" = this stage passed the ticket on to the next stage (the final action comes later)
ACTIONS = ("auto_respond", "escalate", "block", "continue")
FINAL_STAGE = "routing"   # every ticket must have at least a routing decision (A8 reconciliation)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    input_file TEXT NOT NULL,
    tickets_in_file INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    git_commit TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    timestamp TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    input_summary TEXT,
    model_name TEXT,
    model_version TEXT,
    prediction_value TEXT,
    prediction_confidence REAL,
    alternatives TEXT,           -- JSON [{value, confidence}]
    sources_used TEXT,           -- JSON [{doc_id, score}]
    threshold_applied REAL,
    action_taken TEXT NOT NULL,  -- auto_respond | escalate | block | continue
    rule TEXT,
    reason TEXT NOT NULL,
    guardrail_results TEXT,      -- JSON {check: result}
    prompt_version TEXT,
    requirement_ids TEXT,        -- JSON ["FR-02", ...]
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_run_ticket ON decisions(run_id, ticket_id);
CREATE INDEX IF NOT EXISTS idx_decisions_run_stage ON decisions(run_id, stage);
"""


class DecisionRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    ticket_id: str
    stage: str
    action_taken: str
    reason: str
    input_summary: str = ""
    model_name: str = ""
    model_version: str = ""
    prediction_value: Optional[str] = None
    prediction_confidence: Optional[float] = None
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    sources_used: List[Dict[str, Any]] = Field(default_factory=list)
    threshold_applied: Optional[float] = None
    rule: Optional[str] = None
    guardrail_results: Dict[str, Any] = Field(default_factory=dict)
    prompt_version: Optional[str] = None
    requirement_ids: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    decision_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))


def input_summary(channel: str, text: str) -> str:
    """Enough to identify and reconstruct the input later, without storing customer text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"channel={channel}; chars={len(text)}; sha256={digest}"


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


class DecisionLog:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or get_settings().database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def start_run(self, input_file: str, tickets_in_file: int, config: Dict[str, Any]) -> str:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._conn.execute(
            "INSERT INTO runs (run_id, started_at, input_file, tickets_in_file, config_json, git_commit)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), str(input_file),
             tickets_in_file, json.dumps(config, default=str), _git_commit()))
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: str) -> None:
        self._conn.execute("UPDATE runs SET finished_at = ? WHERE run_id = ?",
                           (datetime.now(timezone.utc).isoformat(timespec="seconds"), run_id))
        self._conn.commit()

    def record(self, rec: DecisionRecord) -> None:
        if rec.stage not in STAGES:
            raise ValueError(f"Unknown stage {rec.stage!r}")
        if rec.action_taken not in ACTIONS:
            raise ValueError(f"Unknown action {rec.action_taken!r}")
        self._conn.execute(
            "INSERT INTO decisions (decision_id, run_id, timestamp, ticket_id, stage, input_summary,"
            " model_name, model_version, prediction_value, prediction_confidence, alternatives,"
            " sources_used, threshold_applied, action_taken, rule, reason, guardrail_results,"
            " prompt_version, requirement_ids, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec.decision_id, rec.run_id, rec.timestamp, rec.ticket_id, rec.stage, rec.input_summary,
             rec.model_name, rec.model_version, rec.prediction_value, rec.prediction_confidence,
             json.dumps(rec.alternatives), json.dumps(rec.sources_used), rec.threshold_applied,
             rec.action_taken, rec.rule, rec.reason, json.dumps(rec.guardrail_results),
             rec.prompt_version, json.dumps(rec.requirement_ids), rec.error))
        self._conn.commit()

    def decisions(self, run_id: str, ticket_id: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM decisions WHERE run_id = ?"
        args: List[Any] = [run_id]
        if ticket_id:
            q += " AND ticket_id = ?"
            args.append(ticket_id)
        cur = self._conn.execute(q + " ORDER BY timestamp", args)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            for k in ("alternatives", "sources_used", "guardrail_results", "requirement_ids"):
                r[k] = json.loads(r[k]) if r[k] else None
        return rows

    def reconcile(self, run_id: str, ticket_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """A8: every processed ticket must have a final (routing) decision logged."""
        expected = self._conn.execute("SELECT tickets_in_file FROM runs WHERE run_id = ?",
                                      (run_id,)).fetchone()
        logged = {r[0] for r in self._conn.execute(
            "SELECT DISTINCT ticket_id FROM decisions WHERE run_id = ? AND stage = ?",
            (run_id, FINAL_STAGE))}
        missing = sorted(set(ticket_ids) - logged) if ticket_ids is not None else []
        by_stage = dict(self._conn.execute(
            "SELECT stage, COUNT(*) FROM decisions WHERE run_id = ? GROUP BY stage", (run_id,)).fetchall())
        n = expected[0] if expected else None
        return {"tickets_in_file": n, "tickets_with_final_decision": len(logged),
                "reconciled": n == len(logged) and not missing, "missing": missing,
                "records_by_stage": by_stage}
