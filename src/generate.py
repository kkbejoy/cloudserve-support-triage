"""Answer drafting and escalation hand-offs (B-08; FR-06, FR-07, FR-08, FR-12).

    gen = Generator()
    draft = gen.draft_answer(ticket, retrieval.articles)   # PR-02: answer with citations, or not answerable
    reply = gen.compose_reply(draft, retrieval.articles)   # adds the automatic-answer footer (code, not model)
    note  = gen.escalation_note(ticket, cls, reason, retrieval, draft)   # hand-off for the agent
    ack   = gen.acknowledgement(ticket, retrieval)          # immediate reply to an escalated customer

The model only drafts. Whether a draft is sent is decided later by the guardrails (B-09).
Any model failure returns a draft with an error, never an exception, so the ticket escalates (A11).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.classify import Classification
from src.config import get_settings
from src.llm import LLM, LLMUnavailable, get_llm
from src.models import Ticket
from src.prompts import fill, load_prompt, neutralise_tags
from src.retrieve import RetrievalResult, RetrievedArticle

logger = logging.getLogger(__name__)
_JSON = re.compile(r"\{.*\}", re.DOTALL)


class Draft(BaseModel):
    answerable: bool = False
    passage_doc_ids: List[str] = Field(default_factory=list)   # articles actually sent to the model
    tokens: int = 0
    answer: str = ""
    cited_doc_ids: List[str] = Field(default_factory=list)
    uncertainty: str = ""
    prompt_version: str = ""
    model: str = ""
    cached: bool = False
    error: Optional[str] = None        # set when no usable draft was produced
    error_kind: Optional[str] = None   # provider_unavailable | unparseable_output


class EscalationNote(BaseModel):
    summary: str
    intent: str
    urgency: str
    confidence: float
    why_escalated: str
    already_known: List[str] = Field(default_factory=list)
    relevant_doc_ids: List[str] = Field(default_factory=list)
    also_see: List[str] = Field(default_factory=list)
    uncertain_about: str = ""
    draft_answer: str = ""
    source: str = "template"           # template | PR-03


def _parse(text: str) -> Dict[str, Any]:
    m = _JSON.search(text or "")
    if not m:
        raise ValueError("no JSON object in model output")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


_DROP = re.compile(r"^\s*(#\s.*|\*\*Applies to:\*\*.*)$")


def _compact(content: str) -> str:
    """Article text without its title line, the 'Applies to' line and blank lines (fewer tokens)."""
    return "\n".join(ln for ln in content.splitlines() if ln.strip() and not _DROP.match(ln))


def _passages(articles: List[RetrievedArticle]) -> str:
    return "\n\n".join(f"[{a.doc_id}] {a.title}\n{_compact(a.content)}" for a in articles)


def select_articles(articles: List[RetrievedArticle], max_articles: int, margin: float) -> List[RetrievedArticle]:
    """The articles sent to the model: the best one, plus the next ones only if they score
    within `margin` of it (a much weaker match adds tokens but rarely the answer)."""
    if not articles:
        return []
    best = articles[0].score
    return [a for a in articles[:max_articles] if a.score >= best - margin]


class Generator:
    def __init__(self, llm: Optional[LLM] = None) -> None:
        self._llm = llm
        s = get_settings()
        self.note_with_llm = s.escalation_note_llm
        self.prompt_version = s.prompt_pr02_version or None
        self.max_articles, self.margin = s.generation_max_articles, s.generation_score_margin
        self.max_tokens = s.generation_max_tokens

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def draft_answer(self, ticket: Ticket, articles: List[RetrievedArticle]) -> Draft:
        template, version = load_prompt("PR-02", self.prompt_version)
        sent = select_articles(articles, self.max_articles, self.margin)
        ids = [a.doc_id for a in sent]
        prompt = fill(template, passages=_passages(sent), channel=ticket.channel,
                      ticket_text=neutralise_tags(ticket.text, "ticket", "passages"))
        last_err, tokens = None, 0
        for attempt in range(2):                          # one retry on unparseable output
            try:
                reply = self.llm.complete(prompt if attempt == 0 else
                                          prompt + "\n\nReturn ONLY the JSON object, nothing else.",
                                          max_tokens=self.max_tokens)
            except LLMUnavailable as exc:
                return Draft(prompt_version=version, model=getattr(self.llm, "name", ""), passage_doc_ids=ids,
                             tokens=tokens, error=str(exc)[:200], error_kind="provider_unavailable")
            tokens += reply.tokens_in + reply.tokens_out
            try:
                data = _parse(reply.text)
                cited = data.get("cited_doc_ids") or []
                cited = [str(c) for c in cited if str(c) in ids] if isinstance(cited, list) else []
                return Draft(answerable=bool(data.get("answerable")) and bool(str(data.get("answer", "")).strip()),
                             answer=str(data.get("answer", "")).strip(), cited_doc_ids=cited,
                             uncertainty=str(data.get("uncertainty", "") or ""), passage_doc_ids=ids,
                             prompt_version=version, model=reply.model, cached=reply.cached, tokens=tokens)
            except (ValueError, json.JSONDecodeError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
        return Draft(prompt_version=version, model=getattr(self.llm, "name", ""), passage_doc_ids=ids, tokens=tokens,
                     error=f"unparseable model output ({last_err})", error_kind="unparseable_output")

    @staticmethod
    def compose_reply(draft: Draft, articles: List[RetrievedArticle]) -> str:
        """Customer-facing text: the draft plus a disclosure footer added by code (FR-07)."""
        cited = [a for a in articles if a.doc_id in draft.cited_doc_ids] or articles[:1]
        refs = "; ".join(f"{a.title} ({a.doc_id})" for a in cited)
        return (f"{draft.answer}\n\n---\nThis reply was prepared automatically from CloudServe's help "
                f"articles: {refs}. If it doesn't solve the problem, just reply and a member of our "
                f"support team will pick it up.")

    @staticmethod
    def acknowledgement(ticket: Ticket, retrieval: Optional[RetrievalResult]) -> str:
        """Immediate first reply for an escalated ticket (FR-12): no answer, no promises."""
        text = (f"Thanks for contacting CloudServe support. We've received your request (ref {ticket.ticket_id}) "
                "and a member of our support team will follow up.")
        arts = retrieval.articles[:3] if retrieval else []
        if arts:
            text += " In the meantime, these articles may be useful: " + "; ".join(
                f"{a.title} ({a.doc_id})" for a in arts) + "."
        return text

    def escalation_note(self, ticket: Ticket, cls: Optional[Classification], why: str,
                        retrieval: Optional[RetrievalResult], draft: Optional[Draft] = None) -> EscalationNote:
        """Hand-off for the agent (FR-08). Template by default; PR-03 when ESCALATION_NOTE_LLM=true."""
        arts = retrieval.articles if retrieval else []
        also = sorted({d for a in arts for d in a.related_docs} - {a.doc_id for a in arts})
        first = re.split(r"(?<=[.!?])\s+", ticket.body.strip() or ticket.text.strip())[0][:240]
        note = EscalationNote(
            summary=f"{ticket.channel} ticket: \"{first}\"" if first else f"{ticket.channel} ticket (no readable text)",
            intent=cls.intent if cls else "unknown", urgency=cls.urgency if cls else "unknown",
            confidence=cls.intent_confidence if cls else 0.0, why_escalated=why,
            relevant_doc_ids=[a.doc_id for a in arts], also_see=also,
            uncertain_about=(draft.uncertainty if draft and draft.uncertainty else
                             (cls.fallback_reason or "") if cls else "classification unavailable"),
            draft_answer=draft.answer if draft and draft.answer else "")
        if not self.note_with_llm or cls is None:
            return note
        try:
            template, version = load_prompt("PR-03")
            prompt = fill(template, intent=cls.intent, intent_confidence=f"{cls.intent_confidence:.2f}",
                          urgency=cls.urgency, urgency_confidence=f"{cls.urgency_confidence:.2f}",
                          routing_reason=why, passages_or_none=", ".join(note.relevant_doc_ids) or "none",
                          draft_or_none=note.draft_answer or "none", channel=ticket.channel,
                          ticket_text=neutralise_tags(ticket.text, "ticket"))
            data = _parse(self.llm.complete(prompt, max_tokens=400).text)
            note.summary = str(data.get("summary") or note.summary)
            note.already_known = [str(x) for x in (data.get("already_known") or [])][:10]
            note.uncertain_about = str(data.get("uncertain_about") or note.uncertain_about)
            note.source = version
        except Exception as exc:  # template note stands
            logger.info("PR-03 note failed (%s); using template", type(exc).__name__)
        return note
