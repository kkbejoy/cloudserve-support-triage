"""Text checks shared by the guardrails and the prompts (FR-09, NFR-04, NFR-10).

    find_private_data(text) -> kinds of private data found ("email address", "API key or token", ...)
    redact(text)            -> text with private data replaced by [redacted ...]; used on ticket
                               text BEFORE it is sent to the model (NFR-04: only the data needed).
                               Answers are never redacted: a reply containing private data is
                               blocked (Governance Framework: "never redact and send").
    find_injection(text)    -> phrases in a ticket that try to change the system's instructions
                               (Governance Framework, instruction integrity)
"""
from __future__ import annotations

import re
from typing import List, Tuple

_PRIVATE = [
    ("email address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("API key or token", re.compile(r"\b(?:sk|pk|rk|key|tok|api|ghp|xox[abp])[-_](?=[A-Za-z0-9_\-]*\d)"
                                    r"[A-Za-z0-9_\-]{12,}\b", re.I)),
    ("API key or token", re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{32,}\b")),
    # a value after "password is/:/=" that contains a digit ("password is correct" is not a password)
    ("password", re.compile(r"\b(?:password|passwd|pwd)\s*(?:is|:|=)\s*(?=\S*\d)\S+", re.I)),
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


def _spans(text: str) -> List[Tuple[int, int, str]]:
    spans = [(m.start(), m.end(), kind) for kind, rx in _PRIVATE for m in rx.finditer(text)]
    for m in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= 9 and not _DATE.match(m.group().strip()):
            spans.append((m.start(), m.end(), "phone number"))
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            spans.append((m.start(), m.end(), "card number"))
    return spans


def find_private_data(text: str) -> List[str]:
    """Kinds of private data found in text (empty list = none)."""
    return sorted({kind for _, _, kind in _spans(text)})


def redact(text: str) -> str:
    """Replace private data with [redacted <kind>]; overlapping matches merge into one."""
    spans = sorted(_spans(text))
    if not spans:
        return text
    out, pos = [], 0
    merged: List[List] = []
    for s, e, kind in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, kind])
    for s, e, kind in merged:
        out.append(text[pos:s])
        out.append(f"[redacted {kind}]")
        pos = e
    out.append(text[pos:])
    return "".join(out)


# --- instruction integrity ------------------------------------------------------------------------
_INJECTION = [
    ("override instructions", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,40}?\b(?:previous|prior|above|earlier|all|any|"
        r"your|the|these|those|system)\b.{0,20}?\b(?:instructions?|rules|prompts?|guidelines|guardrails)\b",
        re.I | re.S)),
    ("reveal instructions", re.compile(
        r"\b(?:reveal|show|print|repeat|output|tell me|give me|what (?:is|are))\b.{0,30}?\b(?:your|the)\s+"
        r"(?:system\s+prompt|hidden\s+prompt|instructions|initial\s+prompt|rules)\b", re.I | re.S)),
    ("role change", re.compile(
        r"\b(?:you are now|from now on,? you|act as (?:an? )?(?:unrestricted|different|new|evil) |"
        r"pretend (?:to be|you are)|roleplay as|developer mode|jailbreak|do anything now)\b", re.I)),
    ("system-prompt wording", re.compile(r"\bsystem\s+prompt\b|\bnew instructions\s*:|\bBEGIN (?:SYSTEM|INSTRUCTIONS)\b"
                                         r"|^\s*#{2,}\s*(?:system|instructions)\b", re.I | re.M)),
    ("fake prompt tags", re.compile(r"</?\s*(?:ticket|passages|sentences|system|assistant|instructions)\b[^>]*>",
                                    re.I)),
]


def find_injection(text: str) -> List[Tuple[str, str]]:
    """(kind, matched words) for each attempt to change the system's instructions."""
    found = []
    for kind, rx in _INJECTION:
        m = rx.search(text or "")
        if m:
            found.append((kind, re.sub(r"\s+", " ", m.group())[:120]))
    return found
