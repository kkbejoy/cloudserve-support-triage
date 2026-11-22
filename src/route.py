"""Routing: escalate, or pass the ticket on to answer drafting (B-07; FR-04, FR-05).

    decision = Router().route(ticket, classification, retrieval)
    decision.action  -> "escalate" or "answer"   ("answer" = go on to generation + guardrails;
                                                    the final action may still become escalate/block)
    decision.rule    -> which rule decided, e.g. "must_escalate_intent"
    decision.reason  -> plain-English explanation a support manager can read

Rules run in a fixed order and the first that applies decides. Nothing here is random,
so the same inputs always give the same decision (A5, NFR-08).
Urgency never affects routing: it is not predictable from the text (docs/decisions/classifier.md).
"""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from src.classify import Classification
from src.config import Settings, get_settings
from src.models import Ticket
from src.retrieve import RetrievalResult

BLOCKING_FLAGS = {"empty_body", "malformed", "unknown_channel", "missing_ticket_id"}
BILLING_INTENTS = {"billing_query", "quota_or_overage"}
# money commitments are not the system's to make (FR-05, Governance: tone and scope)
BILLING_DISPUTE = re.compile(
    r"\b(refund\w*|credit(s|ed)?|dispute\w*|chargeback\w*|overcharg\w*|compensat\w*|"
    r"reimburs\w*|money back|charged twice|double charged|cancel\w* (my |our )?(plan|subscription|contract))\b",
    re.IGNORECASE)


def _a(intent: str) -> str:
    """'a billing query' / 'an api key issue'."""
    words = intent.replace("_", " ")
    return ("an " if words[:1] in "aeiou" else "a ") + words


class RoutingDecision(BaseModel):
    ticket_id: str
    action: str                       # "answer" | "escalate"
    rule: str                         # which rule decided
    reason: str                       # plain English, for the log and the escalation note
    threshold_applied: Optional[float] = None
    requirement_ids: List[str] = Field(default_factory=list)


class Router:
    def __init__(self, settings: Optional[Settings] = None,
                 confidence_threshold: Optional[float] = None,
                 auto_respond_enabled: Optional[bool] = None) -> None:
        s = settings or get_settings()
        self.threshold = s.confidence_threshold if confidence_threshold is None else confidence_threshold
        self.enabled = s.auto_respond_enabled if auto_respond_enabled is None else auto_respond_enabled
        self.must_escalate = s.must_escalate

    def _decide(self, t: Ticket, action: str, rule: str, reason: str, reqs: List[str],
                threshold: Optional[float] = None) -> RoutingDecision:
        return RoutingDecision(ticket_id=t.ticket_id, action=action, rule=rule, reason=reason,
                               threshold_applied=threshold, requirement_ids=reqs)

    def route(self, ticket: Ticket, cls: Optional[Classification],
              retrieval: Optional[RetrievalResult]) -> RoutingDecision:
        # 0. kill switch
        if not self.enabled:
            return self._decide(ticket, "escalate", "kill_switch",
                                "Automatic answers are switched off (AUTO_RESPOND_ENABLED=false), "
                                "so every ticket goes to a person.", ["FR-04"])
        # 1. unreadable ticket
        blocking = sorted(BLOCKING_FLAGS.intersection(ticket.flags))
        if blocking:
            return self._decide(ticket, "escalate", "unreadable_ticket",
                                f"The ticket could not be read reliably ({', '.join(blocking)}), "
                                "so no answer was attempted.", ["FR-01", "FR-04"])
        # 2. no usable classification (missing confidence is not a high one - Governance s.4)
        if cls is None or cls.fallback_reason:
            why = cls.fallback_reason if cls else "no classification was produced"
            return self._decide(ticket, "escalate", "no_reliable_classification",
                                f"The system could not tell what the customer needs ({why}).",
                                ["FR-02", "FR-04"])
        # 3. intents that always go to a person
        if cls.intent in self.must_escalate:
            return self._decide(ticket, "escalate", "must_escalate_intent",
                                f"Recognised as {_a(cls.intent)} (confidence {cls.intent_confidence:.2f}); "
                                f"these always go to a person.", ["FR-05"])
        # 4. billing disputes and money commitments
        if cls.intent in BILLING_INTENTS:
            m = BILLING_DISPUTE.search(ticket.text)
            if m:
                return self._decide(ticket, "escalate", "billing_dispute",
                                    f"Billing ticket mentions '{m.group(0)}'; refunds, credits and "
                                    "disputes are decided by a person.", ["FR-05"])
        # 5. nothing relevant in the documentation
        if retrieval is None or not retrieval.articles:
            thr = retrieval.threshold if retrieval else None
            return self._decide(ticket, "escalate", "no_relevant_documentation",
                                "No help article was relevant enough to answer from "
                                f"(best match {retrieval.top_score:.2f}, needed {thr:.2f})."
                                if retrieval else "Documentation search did not run.",
                                ["FR-03", "FR-04"], thr)
        # 6. classifier not confident enough
        if cls.intent_confidence < self.threshold:
            return self._decide(ticket, "escalate", "below_confidence_threshold",
                                f"Probably {_a(cls.intent)}, but confidence "
                                f"{cls.intent_confidence:.2f} is below the {self.threshold:.2f} needed "
                                "to answer automatically.", ["FR-02", "FR-04"], self.threshold)
        # 7. go on to drafting an answer
        docs = ", ".join(retrieval.doc_ids)
        return self._decide(ticket, "answer", "eligible_for_answer",
                            f"Recognised as {_a(cls.intent)} (confidence "
                            f"{cls.intent_confidence:.2f} >= {self.threshold:.2f}) with relevant "
                            f"documentation ({docs}); passed to answer drafting.",
                            ["FR-04", "FR-06"], self.threshold)
