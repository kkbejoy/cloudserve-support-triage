"""Guardrails: the checks every drafted answer must pass before it is sent (B-09; FR-09,
NFR-03, NFR-04, NFR-10; Governance Framework section 4).

    guard = Guardrails()
    verdict = guard.check(ticket, draft, articles, classification)
    verdict.passed        -> True: send (auto_respond); False: block and escalate
    verdict.failed_check  -> which check stopped it, verdict.reason -> why, for the support manager

Checks run cheapest first and stop at the first failure, so the paid model call (grounding)
only runs for drafts that already passed every free check:

    1 policy        kill switch off, or intent confidence below the threshold
    2 private_data  email address, API key/token, password, card number, phone number
    3 scope_tone    promises (refund, credit, "this will fix it", dates), unprofessional words,
                    talk about the system itself
    4 citations     no citation of an article the model was not given
    5 grounding     every sentence supported by the articles sent (PR-04, one model call); the
                    fact-check also names the supporting article for each sentence
    6 citation_coverage  code adds that article as [DOC-...] to any sentence the model left
                    uncited; a claim that still has no citation blocks the draft (FR-06)

If a check cannot run (model down, unreadable reply) the draft is blocked: when in doubt a
person answers. Nothing here raises; the pipeline always gets a verdict.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.classify import Classification
from src.config import Settings, get_settings
from src.generate import Draft, _compact, _parse
from src.llm import LLM, LLMUnavailable, OpenAICompatibleLLM, get_llm
from src.models import Ticket
from src.prompts import fill, load_prompt, neutralise_tags
from src.retrieve import RetrievedArticle

logger = logging.getLogger(__name__)

CHECKS = ("policy", "private_data", "scope_tone", "citations", "grounding", "citation_coverage")
REQUIREMENTS = {"policy": ["FR-04", "FR-09"], "private_data": ["FR-09", "NFR-04"],
                "scope_tone": ["FR-05", "FR-09"], "citations": ["FR-06", "NFR-03"],
                "grounding": ["FR-06", "FR-09", "NFR-03"], "citation_coverage": ["FR-06", "NFR-03"]}

# --- 2. private data ---------------------------------------------------------------------------
_PRIVATE = [
    ("email address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("API key or token", re.compile(r"\b(?:sk|pk|rk|key|tok|api|ghp|xox[abp])[-_](?=[A-Za-z0-9_\-]*\d)"
                                    r"[A-Za-z0-9_\-]{12,}\b", re.I)),
    ("API key or token", re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{32,}\b")),
    ("password", re.compile(r"\b(?:password|passwd|pwd)\s*(?:is|:|=)\s*\S+", re.I)),
]
_PHONE = re.compile(r"(?<![\w-])\+?\d[\d \-()]{8,}\d(?![\w-])")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def find_private_data(text: str) -> List[str]:
    """Kinds of private data found in text (empty list = none)."""
    found = [kind for kind, rx in _PRIVATE if rx.search(text)]
    for m in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= 9 and not _DATE.match(m.group().strip()):
            found.append("phone number")
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.append("card number")
    return sorted(set(found))


# --- 3. scope and tone -------------------------------------------------------------------------
_DAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|end of (?:the )?(?:day|week))"
_SCOPE = [
    ("refund or credit", re.compile(r"\b(?:refund\w*|reimburs\w*|compensat\w*|credit(?:s|ed)? (?:to|on|for) "
                                    r"(?:your|the) (?:account|invoice|bill))\b", re.I)),
    ("guarantee or promise", re.compile(r"\b(?:guarantee\w*|promise\w*|rest assured|we will (?:fix|resolve|refund|"
                                        r"credit|waive|make sure)|i will (?:fix|resolve|make sure))\b", re.I)),
    ("promised outcome", re.compile(r"\b(?:this|that|it|these steps?) (?:will|should|is going to) "
                                    r"(?:fix|resolve|solve|sort)\b|\bwill be (?:fixed|resolved|solved)\b", re.I)),
    ("promised date", re.compile(rf"\b(?:by|before|until) {_DAYS}\b|\bwithin \d+ (?:business |working )?"
                                 r"(?:hours?|days?)\b.{0,40}\b(?:we|our team)\b", re.I)),
    ("waived charge", re.compile(r"\b(?:waive\w*|free of charge|no charge)\b", re.I)),
    ("unprofessional wording", re.compile(r"\b(?:stupid|dumb|obviously|calm down|your fault|idiot\w*)\b", re.I)),
    ("talks about the system", re.compile(r"\b(?:as an ai|language model|the passages?|my instructions|"
                                          r"system prompt)\b", re.I)),
]


def find_scope_problems(text: str) -> List[str]:
    return [kind for kind, rx in _SCOPE if rx.search(text)]


# --- 4/5. citations and grounding helpers ------------------------------------------------------
_CITE = re.compile(r"\[(DOC-[A-Z0-9-]+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SENT.split(text) if s.strip()] if text else []


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s+([.!?])", r"\1", _CITE.sub("", sentence)).strip()


def add_citation(sentence: str, doc_id: str) -> str:
    """'Do X.' -> 'Do X [DOC-A].' (citation before the final punctuation, as PR-02 writes them)."""
    m = re.match(r"^(.*?)([.!?]+[\"')\]]*)?$", sentence.rstrip(), re.S)
    body, end = m.group(1).rstrip(), m.group(2) or ""
    return f"{body} [{doc_id}]{end}"


class CheckResult(BaseModel):
    name: str
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    passed: bool
    failed_check: Optional[str] = None
    reason: str = ""
    checks: List[CheckResult] = Field(default_factory=list)
    model: str = ""
    prompt_version: Optional[str] = None
    tokens: int = 0
    cached: Optional[bool] = None
    # the answer as it will be sent: every claim carries a citation (some added by code)
    cited_answer: Optional[str] = None
    cited_doc_ids: List[str] = Field(default_factory=list)
    citations_added: int = 0

    @property
    def requirement_ids(self) -> List[str]:
        return REQUIREMENTS.get(self.failed_check or "", ["FR-09"])


class Guardrails:
    def __init__(self, llm: Optional[LLM] = None, settings: Optional[Settings] = None,
                 confidence_threshold: Optional[float] = None, enabled: Optional[bool] = None) -> None:
        self.s = settings or get_settings()
        self._llm = llm
        self.threshold = self.s.confidence_threshold if confidence_threshold is None else confidence_threshold
        self.enabled = self.s.auto_respond_enabled if enabled is None else enabled
        self.allow_partial = self.s.grounding_allow_partial

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            if self.s.grounding_provider or self.s.grounding_model:
                self._llm = OpenAICompatibleLLM(provider=self.s.grounding_provider or None,
                                                model=self.s.grounding_model or None)
            else:
                self._llm = get_llm()
        return self._llm

    # -- the five checks ------------------------------------------------------------------------
    def _policy(self, cls: Optional[Classification]) -> CheckResult:
        if not self.enabled:
            return CheckResult(name="policy", passed=False,
                               reason="Automatic answers are switched off (kill switch).")
        if cls is None or cls.intent_confidence < self.threshold:
            conf = f"{cls.intent_confidence:.2f}" if cls else "none"
            return CheckResult(name="policy", passed=False,
                               reason=f"Intent confidence {conf} is below the {self.threshold:.2f} threshold.")
        return CheckResult(name="policy", passed=True)

    @staticmethod
    def _private(answer: str) -> CheckResult:
        kinds = find_private_data(answer)
        if kinds:
            return CheckResult(name="private_data", passed=False, details={"found": kinds},
                               reason=f"The draft contains private data ({', '.join(kinds)}).")
        return CheckResult(name="private_data", passed=True)

    @staticmethod
    def _scope(answer: str) -> CheckResult:
        kinds = find_scope_problems(answer)
        if kinds:
            return CheckResult(name="scope_tone", passed=False, details={"found": kinds},
                               reason=f"The draft makes a commitment or uses wording we do not allow "
                                      f"({', '.join(kinds)}).")
        return CheckResult(name="scope_tone", passed=True)

    @staticmethod
    def _citations(draft: Draft) -> CheckResult:
        in_text = set(_CITE.findall(draft.answer))
        sent = set(draft.passage_doc_ids)
        foreign = sorted(in_text - sent)
        if foreign:
            return CheckResult(name="citations", passed=False, details={"not_given": foreign},
                               reason=f"The draft cites articles it was not given ({', '.join(foreign)}).")
        if not sent:
            return CheckResult(name="citations", passed=False, reason="No help article was given to cite.")
        return CheckResult(name="citations", passed=True,
                           details={"cited": sorted(in_text | set(draft.cited_doc_ids))})

    def _grounding(self, draft: Draft, articles: List[RetrievedArticle]) -> Tuple[CheckResult, Dict[str, Any]]:
        meta: Dict[str, Any] = {"model": "", "prompt_version": None, "tokens": 0, "cached": None,
                                "originals": [], "sources": {}}
        passages = [a for a in articles if a.doc_id in draft.passage_doc_ids] or articles[:1]
        originals = split_sentences(draft.answer)
        sentences = [strip_citations(x) for x in originals]
        meta["originals"] = originals
        if not sentences:
            return CheckResult(name="grounding", passed=False, reason="The draft is empty."), meta
        template, version = load_prompt("PR-04")
        meta["prompt_version"] = version
        prompt = fill(template,
                      passages="\n\n".join(f"[{a.doc_id}] {a.title}\n{_compact(a.content)}" for a in passages),
                      sentences="\n".join(f"{i}. {neutralise_tags(s, 'sentences', 'passages')}"
                                          for i, s in enumerate(sentences, 1)))
        try:
            llm = self.llm
            meta["model"] = getattr(llm, "name", "")
            last = None
            for attempt in range(2):
                reply = llm.complete(prompt if attempt == 0 else prompt + "\n\nReturn ONLY the JSON object.",
                                     max_tokens=300)
                meta["tokens"] += reply.tokens_in + reply.tokens_out
                meta["cached"] = reply.cached
                try:
                    rows = _parse(reply.text).get("sentences")
                    if not isinstance(rows, list):
                        raise ValueError("no 'sentences' list")
                    break
                except ValueError as exc:
                    last, rows = exc, None
            if rows is None:
                return CheckResult(name="grounding", passed=False,
                                   reason=f"The fact-check reply could not be read ({last}); blocked to be safe."), meta
        except LLMUnavailable as exc:
            return CheckResult(name="grounding", passed=False, details={"error": str(exc)[:200]},
                               reason="The fact-check could not run (model unavailable); blocked to be safe."), meta
        ok = {"yes", "no_claim"} | ({"partial"} if self.allow_partial else set())
        verdicts = {int(r.get("n", 0)): r for r in rows if isinstance(r, dict) and str(r.get("n", "")).isdigit()}
        bad = []
        for i, s in enumerate(sentences, 1):
            r = verdicts.get(i)
            label = str(r.get("supported", "")).lower() if r else "missing"
            meta["sources"][i] = (label, str((r or {}).get("doc_id", "") or "").strip())
            if label not in ok:
                bad.append({"n": i, "sentence": s[:200], "supported": label,
                            "unsupported_part": str((r or {}).get("unsupported_part", ""))[:200]})
        if bad:
            first = bad[0]
            part = f": \"{first['unsupported_part']}\"" if first["unsupported_part"] else ""
            return CheckResult(name="grounding", passed=False, details={"unsupported": bad},
                               reason=f"{len(bad)} of {len(sentences)} sentences are not supported by the cited "
                                      f"articles (sentence {first['n']} {first['supported']}{part})."), meta
        return CheckResult(name="grounding", passed=True, details={"sentences": len(sentences)}), meta

    @staticmethod
    def _coverage(draft: Draft, meta: Dict[str, Any]) -> Tuple[CheckResult, Dict[str, Any]]:
        """Give every claim a citation (FR-06). Keep the model's own; otherwise use the article the
        fact-check named, or the only article sent. A claim left without one blocks the draft."""
        sent = list(draft.passage_doc_ids)
        out, added, missing, used = [], 0, [], []
        for i, original in enumerate(meta.get("originals", []), 1):
            label, doc = meta.get("sources", {}).get(i, ("missing", ""))
            own = _CITE.findall(original)
            if own:
                out.append(original)
                used += own
                continue
            if label == "no_claim":
                out.append(original)
                continue
            doc = doc if doc in sent else (sent[0] if len(sent) == 1 else "")
            if not doc:
                missing.append(i)
                out.append(original)
                continue
            out.append(add_citation(original, doc))
            used.append(doc)
            added += 1
        cited_ids = list(dict.fromkeys(used))
        extra = {"cited_answer": " ".join(out), "cited_doc_ids": cited_ids, "citations_added": added}
        if missing:
            return CheckResult(name="citation_coverage", passed=False, details={"uncited_sentences": missing},
                               reason=f"Sentence {missing[0]} has no citation and the supporting article could "
                                      "not be identified."), extra
        if not cited_ids:
            return CheckResult(name="citation_coverage", passed=False,
                               reason="The draft makes no claim that can be cited."), extra
        return CheckResult(name="citation_coverage", passed=True,
                           details={"citations_added": added, "cited": cited_ids}), extra

    # -- run all ----------------------------------------------------------------------------------
    def check(self, ticket: Ticket, draft: Draft, articles: List[RetrievedArticle],
              cls: Optional[Classification]) -> Verdict:
        results: List[CheckResult] = []
        meta: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        try:
            for name in CHECKS:
                if name == "policy":
                    r = self._policy(cls)
                elif name == "private_data":
                    r = self._private(draft.answer)
                elif name == "scope_tone":
                    r = self._scope(draft.answer)
                elif name == "citations":
                    r = self._citations(draft)
                elif name == "grounding":
                    r, meta = self._grounding(draft, articles)
                else:
                    r, extra = self._coverage(draft, meta)
                results.append(r)
                if not r.passed:
                    break
        except Exception as exc:  # never let a guardrail bug send an answer
            logger.exception("guardrail error for %s", ticket.ticket_id)
            results.append(CheckResult(name="error", passed=False,
                                       reason=f"A guardrail failed to run ({type(exc).__name__}); blocked to be safe."))
        failed = next((r for r in results if not r.passed), None)
        return Verdict(passed=failed is None, failed_check=failed.name if failed else None,
                       reason=failed.reason if failed else "All checks passed: no private data, no commitments, "
                       "every sentence supported by and cited to a help article.",
                       checks=results, model=meta.get("model", ""), prompt_version=meta.get("prompt_version"),
                       tokens=meta.get("tokens", 0), cached=meta.get("cached"),
                       cited_answer=extra.get("cited_answer") if failed is None else None,
                       cited_doc_ids=extra.get("cited_doc_ids", []),
                       citations_added=extra.get("citations_added", 0))
