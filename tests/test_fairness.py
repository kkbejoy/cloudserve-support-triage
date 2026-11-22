"""Fairness audit (B-14, NFR-06)."""
from scripts.fairness_audit import audit, to_markdown


def test_T14a_fcr_and_answerable_answered_per_group():
    rows = [{"ticket_id": f"T{i}", "customer_tier": "standard" if i < 4 else "business", "customer_region": "europe",
             "language_fluency": "fluent", "channel": "email",
             "final_action": "auto_respond" if i in (0, 4, 5) else "escalate"} for i in range(6)]
    tickets = {f"T{i}": {"labels": {"expected_route": "auto_respond" if i in (0, 1, 4, 5) else "escalate"},
                         "history": {"first_contact_resolution": i == 4}} for i in range(6)}
    res = audit(rows, tickets)
    tier = {t["group"]: t for t in res["customer_tier"]}
    assert tier["standard"]["fcr"] == 25.0 and tier["standard"]["answerable_answered"] == 50.0
    assert tier["business"]["fcr"] == 100.0 and tier["business"]["historical_fcr"] == 50.0
    assert tier["business"]["small"] is True
    md = to_markdown(res, "run_x")
    assert "## customer_tier" in md and "standard*" in md
