"""Tests for ticket ingestion (FR-01). Test IDs match the Stage 3 traceability table."""
from pathlib import Path

import pytest

from src.ingest import clean_text, load_tickets

FIX = Path(__file__).parent / "fixtures"


def test_T01_one_ticket_per_channel_normalises_cleanly():
    result = load_tickets(FIX / "four_channels.json")
    assert [t.channel for t in result.tickets] == ["email", "chat", "docs_comment", "forum"]
    assert all(t.flags == [] for t in result.tickets)
    chat = result.tickets[1]
    assert chat.subject == "" and chat.text == chat.body  # empty subject is normal for chat
    email = result.tickets[0]
    assert email.text.startswith("Cannot log in") and "sign in" in email.text
    for t in result.tickets:
        print(t)  # day-1 checkpoint: one normalised ticket per channel


def test_T01b_malformed_records_are_flagged_not_dropped():
    result = load_tickets(FIX / "malformed.json")
    assert len(result.tickets) == 9  # 9 records in, 9 tickets out
    by_id = {t.ticket_id: t for t in result.tickets}

    assert "empty_body" in by_id["T-010"].flags
    missing = [t for t in result.tickets if "missing_ticket_id" in t.flags]
    assert len(missing) == 1 and missing[0].ticket_id.startswith("UNKNOWN-")
    assert by_id["T-012"].channel == "sms" and "unknown_channel" in by_id["T-012"].flags
    assert by_id["T-013"].customer_tier == "unknown" and "invalid_customer_tier" in by_id["T-013"].flags
    assert by_id["T-014"].body == "12345" and "wrong_type_body" in by_id["T-014"].flags
    assert "T-015" in by_id and "duplicate_id" in by_id["T-015-dup2"].flags
    assert by_id["T-016"].body == "hidden chars here" and by_id["T-016"].flags == []
    assert "malformed" in by_id["UNKNOWN-8"].flags

    problems = {(p.ticket_id, p.problem) for p in result.problems}
    assert ("T-010", "empty_body") in problems
    assert ("T-016", "control_chars_removed") in problems
    info = [p for p in result.problems if p.problem == "control_chars_removed"]
    assert info[0].action_taken == "none"


def test_T01c_labels_and_history_go_to_answer_key_only():
    result = load_tickets(FIX / "four_channels.json")
    assert set(result.answer_key) == {"T-001", "T-002", "T-003", "T-004"}
    assert result.answer_key["T-001"]["labels"]["intent"] == "rate_limit"
    for t in result.tickets:
        dumped = t.model_dump()
        assert "labels" not in dumped and "history" not in dumped
        assert "labels" not in t.raw and "history" not in t.raw


def test_T01d_file_without_labels_loads():
    result = load_tickets(FIX / "no_labels.json")
    assert len(result.tickets) == 4 and result.answer_key == {}


def test_T01e_unreadable_file_raises_clear_error():
    with pytest.raises(ValueError, match="not found"):
        load_tickets(FIX / "does_not_exist.json")
    with pytest.raises(ValueError, match="JSON list"):
        load_tickets(FIX / "not_a_list.json")


def test_T01f_nfc_and_whitespace_without_changing_words():
    decomposed = "café   is   down\n\n\n\nplease  help"
    cleaned, removed = clean_text(decomposed)
    assert cleaned == "café is down\n\nplease help"
    assert removed == 0
    assert clean_text("we are having not change our code")[0] == "we are having not change our code"
