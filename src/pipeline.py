"""The processing chain for tickets, shared by the evaluation harness and the API.

    classify -> retrieve -> route -> (generate -> validate: B-08/B-09, not yet built)

Every step's decision is written to the decision log. A failure on one ticket never stops
the batch: the ticket is escalated with the error recorded (A9, A11), so logged decisions
always reconcile with tickets processed (A8).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.classify import Classification, Classifier
from src.logging_store import DecisionLog, DecisionRecord, input_summary
from src.models import Ticket
from src.retrieve import RetrievalResult, Retriever
from src.route import Router, RoutingDecision

logger = logging.getLogger(__name__)


class TicketOutcome(BaseModel):
    ticket_id: str
    classification: Optional[Classification] = None
    retrieval: Optional[RetrievalResult] = None
    routing: RoutingDecision


class Pipeline:
    def __init__(self, classifier: Classifier, retriever: Retriever, router: Router,
                 log: DecisionLog) -> None:
        self.classifier, self.retriever, self.router, self.log = classifier, retriever, router, log

    def _log_classification(self, run_id: str, t: Ticket, c: Classification) -> None:
        self.log.record(DecisionRecord(
            run_id=run_id, ticket_id=t.ticket_id, stage="classification",
            input_summary=input_summary(t.channel, t.text),
            model_name="intent/urgency classifier", model_version=c.model_version,
            prediction_value=c.intent, prediction_confidence=c.intent_confidence,
            alternatives=[{"value": a.label, "confidence": a.confidence} for a in c.intent_alternatives]
            + [{"value": f"urgency:{c.urgency}", "confidence": c.urgency_confidence}],
            threshold_applied=self.classifier.floor,
            action_taken="escalate" if c.fallback_reason else "continue",
            reason=c.fallback_reason or f"Classified as {c.intent} ({c.intent_confidence:.2f}); "
                                        f"urgency {c.urgency} ({c.urgency_confidence:.2f}).",
            requirement_ids=["FR-02"]))

    def _log_routing(self, run_id: str, t: Ticket, r: Optional[RetrievalResult], d: RoutingDecision,
                     error: Optional[str] = None) -> None:
        self.log.record(DecisionRecord(
            run_id=run_id, ticket_id=t.ticket_id, stage="routing",
            input_summary=input_summary(t.channel, t.text),
            model_name="rules + thresholds", model_version="route.py",
            prediction_value=d.action, rule=d.rule,
            sources_used=[{"doc_id": a.doc_id, "score": a.score, "chunk": a.matched_chunk_id}
                          for a in (r.articles if r else [])],
            threshold_applied=d.threshold_applied,
            action_taken="escalate" if d.action == "escalate" else "continue",
            reason=d.reason, requirement_ids=d.requirement_ids, error=error))

    def process(self, run_id: str, tickets: List[Ticket]) -> List[TicketOutcome]:
        """Process a batch. One classification call for all tickets, then retrieve and route each."""
        try:
            classes = self.classifier.classify_many(tickets)
        except Exception as exc:  # classifier itself never raises, but be safe
            logger.exception("classification failed")
            classes = [None] * len(tickets)
            err = f"classification error: {type(exc).__name__}"
        else:
            err = None
        outcomes: List[TicketOutcome] = []
        for t, c in zip(tickets, classes):
            retrieval, decision, error = None, None, err
            try:
                if c is not None:
                    self._log_classification(run_id, t, c)
                if c is not None and not c.fallback_reason:
                    retrieval = self.retriever.retrieve(t.text)
                decision = self.router.route(t, c, retrieval)
            except Exception as exc:
                logger.exception("ticket %s failed", t.ticket_id)
                error = f"{type(exc).__name__}: {exc}"[:300]
                decision = RoutingDecision(ticket_id=t.ticket_id, action="escalate", rule="processing_error",
                                           reason="An internal error occurred while processing this ticket, "
                                                  "so it was passed to a person.", requirement_ids=["FR-04"])
            self._log_routing(run_id, t, retrieval, decision, error)
            outcomes.append(TicketOutcome(ticket_id=t.ticket_id, classification=c,
                                          retrieval=retrieval, routing=decision))
        return outcomes


def summarise(outcomes: List[TicketOutcome]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(o.routing.rule for o in outcomes))
