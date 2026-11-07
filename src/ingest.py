"""Ticket ingestion: load a ticket file and normalise every record.

Serves FR-01 / backlog B-02 (acceptance criterion A2, and A11 for malformed input).

    result = load_tickets("data/validation_tickets.json")
    result.tickets      -> list[Ticket], one per record, none dropped
    result.answer_key   -> labels + history, for the evaluation harness only
    result.problems     -> what was wrong with any record

Only an unreadable file stops ingestion. A bad record is kept, flagged and later
escalated by the router, so every ticket still produces a logged decision.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.models import CHANNELS, FLUENCY, REGIONS, TIERS, IngestResult, Problem, Ticket

logger = logging.getLogger(__name__)

ANSWER_KEY_FIELDS = ("labels", "history")
_KEEP_CONTROL = {"\n", "\t"}
_SPACES = re.compile(r"[ \t]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(value: str) -> Tuple[str, int]:
    """Normalise Unicode (NFC), remove invisible control/format characters except
    newline and tab, and collapse extra whitespace. Wording is never changed.

    Returns the cleaned text and the number of characters removed.
    """
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    kept: List[str] = []
    removed = 0
    for ch in text:
        if ch not in _KEEP_CONTROL and unicodedata.category(ch) in ("Cc", "Cf"):
            removed += 1
            continue
        kept.append(ch)
    text = "".join(kept)
    text = _SPACES.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip(), removed


def _as_text(record: Dict[str, Any], name: str, ticket_id: str,
             flags: List[str], problems: List[Problem]) -> str:
    """Return a field as a string. Missing or None -> "". Other types are
    converted where sensible and flagged as wrong_type_<field>."""
    value = record.get(name)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        flag = f"wrong_type_{name}"
        flags.append(flag)
        problems.append(Problem(ticket_id=ticket_id, problem=flag,
                                detail=f"{name} was {type(value).__name__}; converted to text"))
        return str(value)
    flag = f"wrong_type_{name}"
    flags.append(flag)
    problems.append(Problem(ticket_id=ticket_id, problem=flag,
                            detail=f"{name} was {type(value).__name__}; ignored"))
    return ""


def _check_allowed(value: str, allowed: Tuple[str, ...], name: str, ticket_id: str,
                   flags: List[str], problems: List[Problem]) -> str:
    """Return value if allowed, else 'unknown' and flag invalid_<name>."""
    cleaned = value.strip().lower()
    if cleaned in allowed:
        return cleaned
    flag = f"invalid_{name}"
    flags.append(flag)
    problems.append(Problem(ticket_id=ticket_id, problem=flag,
                            detail=f"{name}={value!r} is not one of {list(allowed)}"))
    return "unknown"


def normalise(record: Any, index: int, seen_ids: Optional[Set[str]] = None
              ) -> Tuple[Ticket, List[Problem], Optional[Dict[str, Any]]]:
    """Convert one raw record into a Ticket.

    Returns (ticket, problems, answer_key_entry). answer_key_entry holds labels and
    history if present, otherwise None. `seen_ids` is used to detect duplicates.
    """
    seen_ids = seen_ids if seen_ids is not None else set()
    flags: List[str] = []
    problems: List[Problem] = []

    if not isinstance(record, dict):
        ticket_id = f"UNKNOWN-{index}"
        problems.append(Problem(ticket_id=ticket_id, problem="malformed",
                                detail=f"record is {type(record).__name__}, not an object"))
        return Ticket(ticket_id=ticket_id, channel="unknown", flags=["malformed"]), problems, None

    # 1. Split off the answer key so the pipeline can never see it.
    key_entry = {f: record[f] for f in ANSWER_KEY_FIELDS if f in record}
    raw = {k: v for k, v in record.items() if k not in ANSWER_KEY_FIELDS}

    # 2. Ticket ID: present, text, unique.
    ticket_id = record.get("ticket_id")
    if not isinstance(ticket_id, str) or not ticket_id.strip():
        original = ticket_id
        ticket_id = f"UNKNOWN-{index}"
        flags.append("missing_ticket_id")
        problems.append(Problem(ticket_id=ticket_id, problem="missing_ticket_id",
                                detail=f"ticket_id was {original!r}; generated {ticket_id}"))
    ticket_id = ticket_id.strip()
    if ticket_id in seen_ids:
        n = 2
        while f"{ticket_id}-dup{n}" in seen_ids:
            n += 1
        new_id = f"{ticket_id}-dup{n}"
        flags.append("duplicate_id")
        problems.append(Problem(ticket_id=new_id, problem="duplicate_id",
                                detail=f"{ticket_id} already seen; renamed {new_id}"))
        ticket_id = new_id
    seen_ids.add(ticket_id)

    # 3. Channel (kept as given if unknown, so the log shows what arrived).
    channel_raw = _as_text(record, "channel", ticket_id, flags, problems).strip().lower()
    channel = channel_raw if channel_raw in CHANNELS else (channel_raw or "unknown")
    if channel not in CHANNELS:
        flags.append("unknown_channel")
        problems.append(Problem(ticket_id=ticket_id, problem="unknown_channel",
                                detail=f"channel={record.get('channel')!r}"))

    # 4. Customer fields with allowed values.
    tier = _check_allowed(_as_text(record, "customer_tier", ticket_id, flags, problems),
                          TIERS, "customer_tier", ticket_id, flags, problems)
    region = _check_allowed(_as_text(record, "customer_region", ticket_id, flags, problems),
                            REGIONS, "customer_region", ticket_id, flags, problems)
    fluency = _check_allowed(_as_text(record, "language_fluency", ticket_id, flags, problems),
                             FLUENCY, "language_fluency", ticket_id, flags, problems)

    # 5. Text fields. Missing subject is normal (chat), so it is not flagged.
    subject, removed_s = clean_text(_as_text(record, "subject", ticket_id, flags, problems))
    body, removed_b = clean_text(_as_text(record, "body", ticket_id, flags, problems))
    if not body:
        flags.append("empty_body")
        problems.append(Problem(ticket_id=ticket_id, problem="empty_body",
                                detail="body missing or empty after cleaning"))
    if removed_s + removed_b:
        problems.append(Problem(ticket_id=ticket_id, problem="control_chars_removed",
                                detail=f"{removed_s + removed_b} invisible character(s) removed",
                                action_taken="none"))

    text = f"{subject}\n\n{body}" if subject and body else (subject or body)

    ticket = Ticket(
        ticket_id=ticket_id,
        channel=channel,
        subject=subject,
        body=body,
        text=text,
        received_at=_as_text(record, "received_at", ticket_id, flags, problems).strip(),
        customer_id=_as_text(record, "customer_id", ticket_id, flags, problems).strip(),
        customer_name=_as_text(record, "customer_name", ticket_id, flags, problems).strip(),
        customer_tier=tier,
        customer_region=region,
        language_fluency=fluency,
        flags=flags,
        raw=raw,
    )
    return ticket, problems, (key_entry or None)


def load_tickets(path: Union[str, Path]) -> IngestResult:
    """Load a ticket file (development, validation or the hidden set) and normalise it.

    Raises ValueError only if the file itself cannot be read as a JSON list.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Ticket file not found: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Ticket file could not be read as JSON: {path} ({exc})") from exc
    if not isinstance(data, list):
        raise ValueError(f"Ticket file must contain a JSON list of tickets: {path}")

    result = IngestResult()
    seen: Set[str] = set()
    for index, record in enumerate(data):
        ticket, problems, key_entry = normalise(record, index, seen)
        result.tickets.append(ticket)
        result.problems.extend(problems)
        if key_entry:
            result.answer_key[ticket.ticket_id] = key_entry

    flagged = sum(1 for t in result.tickets if t.flags)
    logger.info("Loaded %d tickets from %s (%d flagged, %d answer-key entries)",
                len(result.tickets), path, flagged, len(result.answer_key))
    return result


def _main(argv: List[str]) -> int:
    """Day-1 checkpoint: python -m src.ingest <ticket file>"""
    if len(argv) != 1:
        print("usage: python -m src.ingest <path to ticket json>")
        return 2
    result = load_tickets(argv[0])
    flagged = [t for t in result.tickets if t.flags]
    print(f"Tickets: {len(result.tickets)} | flagged: {len(flagged)} | "
          f"problems: {len(result.problems)} | answer-key entries: {len(result.answer_key)}")
    print("\nOne normalised ticket per channel:")
    shown = set()
    for t in result.tickets:
        if t.channel not in shown:
            shown.add(t.channel)
            print(" ", t)
    if result.problems:
        print("\nProblems:")
        for p in result.problems[:20]:
            print(f"  {p.ticket_id}: {p.problem} - {p.detail} (action: {p.action_taken})")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(_main(sys.argv[1:]))
