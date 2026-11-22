"""Shared data shapes for the CloudServe support triage pipeline.

Serves FR-01 (one normalised internal representation for all four channels).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from pydantic import BaseModel, Field

CHANNELS = ("email", "chat", "docs_comment", "forum")
TIERS = ("enterprise", "business", "standard")
REGIONS = ("north_america", "europe", "asia_pacific", "latin_america")
FLUENCY = ("fluent", "non_fluent")


class Ticket(BaseModel):
    """One support ticket in the common format every later step reads.

    `text` is what classification and retrieval use. `raw` keeps the original
    record (without labels or history) so nothing the customer sent is lost.
    """

    ticket_id: str
    channel: str
    subject: str = ""
    body: str = ""
    text: str = ""
    received_at: str = ""
    customer_id: str = ""
    customer_name: str = ""
    customer_tier: str = "unknown"
    customer_region: str = "unknown"
    language_fluency: str = "unknown"
    flags: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:  # short, print-friendly view
        preview = self.text.replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        flags = f" flags={self.flags}" if self.flags else ""
        return f"[{self.ticket_id} | {self.channel} | {self.customer_tier}]{flags} {preview}"


class Problem(BaseModel):
    """Something wrong (or cleaned) in one input record.

    action_taken is "escalate" for a flag (the router will not auto-answer the
    ticket) and "none" for information-only entries such as removed characters.
    """

    ticket_id: str
    problem: str
    detail: str = ""
    action_taken: str = "escalate"


@dataclass
class IngestResult:
    """What load_tickets returns.

    tickets:    one Ticket per input record, in input order; none dropped.
    answer_key: {ticket_id: {"labels": ..., "history": ...}} for scoring only;
                the pipeline must never read it.
    problems:   every issue found, for the decision log and metrics report.
    """

    tickets: List[Ticket] = field(default_factory=list)
    answer_key: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    problems: List[Problem] = field(default_factory=list)
