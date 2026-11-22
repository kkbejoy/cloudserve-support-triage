"""HTTP API: submit one ticket, get the system's decision back (B-16).

    uvicorn src.api:app                 # then open http://127.0.0.1:8000/docs to try it in the browser

    GET  /health                 is the system ready? index, classifier, model, kill switch
    POST /tickets                one ticket in (same fields as the ticket files); decision out:
                                 intent + confidence, sources, final action, the answer or the
                                 acknowledgement, the hand-off note, guardrail results
    GET  /decisions/{ticket_id}  that ticket's records in the decision log (this server's run)

Every ticket goes through the same pipeline and decision log as the batch harness. The server
opens one decision-log run when it starts and closes it when it stops. Same rules as the
harness: nothing is sent unless every guardrail passes; with no API key, or the provider down,
tickets are escalated rather than failing.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.ingest import normalise

logger = logging.getLogger("api")


class TicketIn(BaseModel):
    """One ticket, in the same shape as the records in the ticket files."""
    model_config = ConfigDict(extra="allow", json_schema_extra={"example": {
        "ticket_id": "DEMO-1", "channel": "email", "subject": "Locked out",
        "body": "I keep getting invalid credentials even though my password is right.",
        "customer_tier": "business", "customer_region": "europe", "language_fluency": "fluent"}})
    ticket_id: Optional[str] = Field(None, description="left empty: one is generated")
    channel: str = Field(..., description="email | chat | docs_comment | forum")
    subject: str = ""
    body: str
    customer_tier: Optional[str] = None
    customer_region: Optional[str] = None
    language_fluency: Optional[str] = None


def build_components() -> Dict[str, Any]:
    """The real system: loads the embedding model, index, classifier and model client."""
    from src.classify import Classifier
    from src.generate import Generator
    from src.guardrails import Guardrails
    from src.logging_store import DecisionLog
    from src.retrieve import Retriever
    from src.route import Router
    retriever = Retriever()
    classifier = Classifier(embedder=retriever.embedder)
    router = Router()
    return {"classifier": classifier, "retriever": retriever, "router": router, "log": DecisionLog(),
            "generator": Generator(),
            "guardrails": Guardrails(confidence_threshold=router.threshold, enabled=router.enabled)}


def create_app(build: Callable[[], Dict[str, Any]] = build_components) -> FastAPI:
    state: Dict[str, Any] = {"ready": False, "error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            from src.pipeline import Pipeline
            parts = build()
            state.update(parts)
            state["pipeline"] = Pipeline(parts["classifier"], parts["retriever"], parts["router"], parts["log"],
                                         parts.get("generator"), parts.get("guardrails"))
            state["run_id"] = parts["log"].start_run("api", 0, {"source": "api"})
            state["ready"] = True
        except Exception as exc:   # start anyway; /health and /tickets explain what is missing
            logger.exception("API start-up failed")
            state["error"] = f"{type(exc).__name__}: {exc}"
        yield
        if state.get("ready"):
            state["log"].finish_run(state["run_id"])

    app = FastAPI(title="CloudServe support triage", version="1.0", lifespan=lifespan,
                  description="Submit a support ticket; get the triage decision, answer or escalation.")

    def _require_ready():
        if not state["ready"]:
            raise HTTPException(status_code=503, detail=f"System not ready: {state['error']}. "
                                                        "Run: python -m scripts.setup")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        if not state["ready"]:
            return {"status": "not_ready", "error": state["error"], "fix": "python -m scripts.setup"}
        router, gen = state["router"], state.get("generator")
        llm = getattr(gen, "_llm", None)
        return {"status": "ok", "run_id": state["run_id"],
                "auto_respond_enabled": router.enabled, "confidence_threshold": router.threshold,
                "classifier": getattr(state["classifier"], "version", None),
                "index_strategy": state["retriever"].strategy,
                "model": getattr(llm, "name", "loaded on first ticket") if gen else "off"}

    @app.post("/tickets")
    async def submit(ticket: TicketIn) -> Dict[str, Any]:
        _require_ready()
        from evaluation.harness import _row
        record = ticket.model_dump(exclude_none=True)
        record["ticket_id"] = record.get("ticket_id") or f"API-{uuid.uuid4().hex[:8]}"
        t, problems, _ = normalise(record, 0, set())
        start = time.perf_counter()
        outcome = state["pipeline"].process(state["run_id"], [t])[0]
        row = _row(t, outcome, {}, {}, (time.perf_counter() - start) * 1000)
        row = {k: v for k, v in row.items() if not k.startswith(("true_", "hist_", "expected_"))
               and k not in ("answerable_from_docs", "must_not_auto_respond")}
        row["input_problems"] = [p.model_dump() for p in problems]
        row["decision_log"] = _log_rows(state["log"], state["run_id"], t.ticket_id)
        return row

    @app.get("/decisions/{ticket_id}")
    async def decisions(ticket_id: str) -> List[Dict[str, Any]]:
        _require_ready()
        rows = _log_rows(state["log"], state["run_id"], ticket_id)
        if not rows:
            raise HTTPException(status_code=404, detail=f"No decisions for {ticket_id} in this server's run")
        return rows

    return app


def _log_rows(log, run_id: str, ticket_id: str) -> List[Dict[str, Any]]:
    keep = ("timestamp", "stage", "action_taken", "rule", "reason", "prediction_value", "prediction_confidence",
            "model_name", "model_version", "guardrail_results", "prompt_version", "requirement_ids", "error")
    return [{k: r.get(k) for k in keep if k in r} for r in log.decisions(run_id, ticket_id)]


app = create_app()
