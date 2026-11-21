"""Tests for routing (B-07; FR-04, FR-05; tests T-04, T-05)."""
import pytest

from src.classify import Alternative, Classification
from src.config import Settings
from src.models import Ticket
from src.retrieve import RetrievalResult, RetrievedArticle
from src.route import Router


def T(text="please help", flags=None, tid="T-1"):
    return Ticket(ticket_id=tid, channel="email", body=text, text=text, flags=flags or [])


def C(intent="rate_limit", conf=0.95, fallback=None):
    return Classification(ticket_id="T-1", intent=intent, intent_confidence=conf,
                          intent_alternatives=[Alternative(label=intent, confidence=conf)],
                          urgency="medium", urgency_confidence=0.5, fallback_reason=fallback)


def R(found=True):
    arts = [RetrievedArticle(doc_id="DOC-API-001", title="Rate limits", score=0.7,
                             matched_chunk_id="DOC-API-001#symptoms", matched_section="Symptoms",
                             content="...")] if found else []
    return RetrievalResult(articles=arts, hits=[{"score": 0.7 if found else 0.2}],
                           threshold=0.35, strategy="section")


@pytest.fixture
def router():
    return Router(settings=Settings(), confidence_threshold=0.8, auto_respond_enabled=True)


def test_T04a_confident_with_docs_goes_to_answer(router):
    d = router.route(T(), C(), R())
    assert d.action == "answer" and d.rule == "eligible_for_answer"
    assert "DOC-API-001" in d.reason and d.threshold_applied == 0.8


@pytest.mark.parametrize("intent", ["security_incident", "compliance_request", "feature_request"])
def test_T05_must_escalate_intents_always_escalate(router, intent):
    d = router.route(T(), C(intent=intent, conf=0.999), R())
    assert d.action == "escalate" and d.rule == "must_escalate_intent" and "FR-05" in d.requirement_ids


def test_T04b_rule_order_and_reasons(router):
    assert router.route(T(flags=["empty_body"]), C(), R()).rule == "unreadable_ticket"
    assert router.route(T(), C(intent="unclear_request", conf=0.0, fallback="below floor"), R()).rule \
        == "no_reliable_classification"
    assert router.route(T(), None, R()).rule == "no_reliable_classification"
    assert router.route(T(), C(), R(found=False)).rule == "no_relevant_documentation"
    low = router.route(T(), C(conf=0.55), R())
    assert low.rule == "below_confidence_threshold" and "0.55" in low.reason and "0.80" in low.reason


def test_T05b_billing_dispute_keywords(router):
    d = router.route(T("We were charged twice, please refund the overage"), C(intent="billing_query"), R())
    assert d.action == "escalate" and d.rule == "billing_dispute"
    ok = router.route(T("How is proration calculated on my invoice?"), C(intent="billing_query"), R())
    assert ok.action == "answer"


def test_T04c_kill_switch_escalates_everything():
    off = Router(settings=Settings(), confidence_threshold=0.8, auto_respond_enabled=False)
    d = off.route(T(), C(), R())
    assert d.action == "escalate" and d.rule == "kill_switch"


def test_T04d_same_input_same_decision(router):
    assert router.route(T(), C(), R()) == router.route(T(), C(), R())
