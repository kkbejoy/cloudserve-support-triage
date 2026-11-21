"""The processing chain for tickets, shared by the evaluation harness and the API.

    classify -> retrieve -> route -> generate (B-08) -> validate (B-09 guardrails) -> send / block

Every step's decision is written to the decision log. A failure on one ticket never stops
the batch: the ticket is escalated with the error recorded (A9, A11), so logged decisions
always reconcile with tickets processed (A8).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.classify import Classification, Classifier
from src.generate import Draft, EscalationNote, Generator
from src.guardrails import Guardrails, Verdict
from src.logging_store import DecisionLog, DecisionRecord, input_summary
from src.models import Ticket
from src.retrieve import RetrievalResult, Retriever
from src.route import Router, RoutingDecision

logger = logging.getLogger(__name__)


# Final action per ticket. Nothing is sent until the guardrails pass it (B-09):
#   auto_respond    drafted AND every guardrail passed: the answer is sent
#   block           drafted but a guardrail failed: not sent, escalated to a person
#   escalate        routed or drafted to a person for another reason
#   pending_answer  router cleared it, no generator configured
#   drafted         an answer was drafted, no guardrails configured - NOT sent, NOT counted as resolved
FINAL_ACTIONS = ("auto_respond", "escalate", "block", "pending_answer", "drafted")


class TicketOutcome(BaseModel):
    ticket_id: str
    classification: Optional[Classification] = None
    retrieval: Optional[RetrievalResult] = None
    routing: RoutingDecision
    final_action: str = "escalate"
    final_rule: str = ""
    final_reason: str = ""
    draft: Optional[Draft] = None
    reply: Optional[str] = None                  # customer-facing answer (only if eventually sent)
    acknowledgement: Optional[str] = None        # immediate reply to an escalated customer (FR-12)
    escalation_note: Optional[EscalationNote] = None   # hand-off for the agent (FR-08)
    validation: Optional[Verdict] = None
    error: Optional[str] = None


class Pipeline:
    def __init__(self, classifier: Classifier, retriever: Retriever, router: Router,
                 log: DecisionLog, generator: Optional[Generator] = None,
                 guardrails: Optional[Guardrails] = None) -> None:
        self.classifier, self.retriever, self.router, self.log = classifier, retriever, router, log
        self.generator = generator
        self.guardrails = guardrails

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

    def _log_generation(self, run_id: str, t: Ticket, d: Draft, action: str, reason: str) -> None:
        self.log.record(DecisionRecord(
            run_id=run_id, ticket_id=t.ticket_id, stage="generation",
            input_summary=input_summary(t.channel, t.text),
            model_name="answer drafting (LLM)", model_version=d.model,
            prediction_value="error" if d.error else ("answerable" if d.answerable else "not_answerable"),
            sources_used=[{"doc_id": c} for c in d.cited_doc_ids],
            action_taken=action, rule="generation", reason=reason, prompt_version=d.prompt_version,
            requirement_ids=["FR-06", "FR-07"], error=d.error))

    def _answer_step(self, run_id: str, t: Ticket, o: "TicketOutcome") -> None:
        """Draft an answer for a ticket the router cleared; decide escalate or drafted."""
        arts = o.retrieval.articles if o.retrieval else []
        d = self.generator.draft_answer(t, arts)
        o.draft = d
        if d.error_kind == "provider_unavailable":
            o.final_action, o.final_rule = "escalate", "provider_unavailable"
            o.final_reason = ("The language model was unavailable, so no answer could be drafted; "
                              "passed to a person.")
        elif d.error:
            o.final_action, o.final_rule = "escalate", "generation_failed"
            o.final_reason = "The drafted answer could not be read reliably, so it was passed to a person."
        elif not d.answerable:
            o.final_action, o.final_rule = "escalate", "not_answerable"
            why = f" ({d.uncertainty})" if d.uncertainty else ""
            o.final_reason = (f"The help articles found ({', '.join(a.doc_id for a in arts)}) do not answer "
                              f"this question{why}; passed to a person.")
        else:
            o.final_action, o.final_rule = "drafted", "answer_drafted"
            o.final_reason = (f"Answer drafted from {', '.join(d.cited_doc_ids) or 'the retrieved articles'}; "
                              "awaiting guardrail checks before it can be sent.")
            o.reply = self.generator.compose_reply(d, arts)
        self._log_generation(run_id, t, d, "escalate" if o.final_action == "escalate" else "continue",
                             o.final_reason)

    def _validate_step(self, run_id: str, t: Ticket, c: Optional[Classification], o: "TicketOutcome") -> None:
        """Run the guardrails on a drafted answer; send it or block it (FR-09)."""
        arts = o.retrieval.articles if o.retrieval else []
        v = self.guardrails.check(t, o.draft, arts, c)
        o.validation = v
        if v.passed:
            o.final_action, o.final_rule = "auto_respond", "guardrails_passed"
            o.final_reason = (f"Answered automatically from {', '.join(o.draft.cited_doc_ids) or 'the cited article'}: "
                              "all guardrail checks passed.")
        else:
            o.final_action, o.final_rule = "block", f"blocked_{v.failed_check}"
            o.final_reason = f"Drafted answer blocked and passed to a person. {v.reason}"
            o.reply = None
        self.log.record(DecisionRecord(
            run_id=run_id, ticket_id=t.ticket_id, stage="validation",
            input_summary=input_summary(t.channel, t.text),
            model_name="guardrails" + (" + fact-check (LLM)" if v.prompt_version else ""),
            model_version=v.model, prediction_value="passed" if v.passed else v.failed_check,
            sources_used=[{"doc_id": d} for d in o.draft.cited_doc_ids],
            threshold_applied=self.guardrails.threshold, action_taken=o.final_action, rule=o.final_rule,
            reason=o.final_reason, prompt_version=v.prompt_version,
            guardrail_results={r.name: {"passed": r.passed, "reason": r.reason, **r.details} for r in v.checks},
            requirement_ids=["FR-09"] if v.passed else v.requirement_ids))

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
            o = TicketOutcome(ticket_id=t.ticket_id, classification=c, retrieval=retrieval, routing=decision,
                              final_action="escalate" if decision.action == "escalate" else "pending_answer",
                              final_rule=decision.rule, final_reason=decision.reason, error=error)
            if self.generator is not None:
                try:
                    if o.final_action == "pending_answer":
                        self._answer_step(run_id, t, o)
                    if o.final_action == "drafted" and self.guardrails is not None:
                        self._validate_step(run_id, t, c, o)
                    if o.final_action in ("escalate", "block"):
                        o.acknowledgement = self.generator.acknowledgement(t, retrieval)
                        o.escalation_note = self.generator.escalation_note(t, c, o.final_reason, retrieval, o.draft)
                except Exception as exc:   # never lose a ticket (A9, A11)
                    logger.exception("answer step failed for %s", t.ticket_id)
                    o.final_action, o.final_rule = "escalate", "processing_error"
                    o.final_reason = "An internal error occurred while drafting or checking; passed to a person."
                    o.reply = None
                    o.error = f"{type(exc).__name__}: {exc}"[:300]
            outcomes.append(o)
        return outcomes


def summarise(outcomes: List[TicketOutcome]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(o.final_rule or o.routing.rule for o in outcomes))
