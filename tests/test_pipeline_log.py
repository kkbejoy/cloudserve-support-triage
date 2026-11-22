"""Decision log and the classify -> retrieve -> route chain (B-10, B-07; FR-10, A8; test T-10).
Offline: fake embedder, small fixtures, temporary files."""
import json
from pathlib import Path

import pytest

from scripts.train_classifier import train
from src.chunking import chunk_documents, load_docs
from src.classify import Classifier
from src.config import Settings
from src.ingest import load_tickets
from src.logging_store import DecisionLog, DecisionRecord, input_summary
from src.pipeline import Pipeline
from src.retrieve import Retriever
from src.route import Router
from src.vector_store import build_collection, get_client
from tests.fake_embedder import FakeEmbedder
from tests.test_classify import _tickets

DOCS = Path(__file__).parent / "fixtures" / "docs_small.json"


def test_T10a_log_round_trip_and_privacy(tmp_path):
    log = DecisionLog(str(tmp_path / "d.db"))
    run = log.start_run("x.json", 1, {"t": 0.8})
    text = "my secret password is hunter2"
    log.record(DecisionRecord(run_id=run, ticket_id="A", stage="routing", action_taken="escalate",
                              reason="because", input_summary=input_summary("email", text),
                              sources_used=[{"doc_id": "D", "score": 0.5}], requirement_ids=["FR-04"]))
    rows = log.decisions(run)
    assert rows[0]["sources_used"] == [{"doc_id": "D", "score": 0.5}] and rows[0]["requirement_ids"] == ["FR-04"]
    assert "hunter2" not in json.dumps(rows)          # customer text never stored
    with pytest.raises(ValueError):
        log.record(DecisionRecord(run_id=run, ticket_id="A", stage="nonsense", action_taken="escalate", reason="x"))
    rec = log.reconcile(run, ["A", "B"])
    assert rec["missing"] == ["B"] and not rec["reconciled"]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    emb = FakeEmbedder()
    tickets = _tickets(n_per_class=12)
    tickets.append({"ticket_id": "T-EMPTY", "channel": "chat", "subject": "", "body": "   ",
                    "labels": {"intent": "billing_query", "urgency": "low"}})
    (d / "t.json").write_text(json.dumps(tickets), encoding="utf-8")
    train(str(d / "t.json"), str(d / "clf.joblib"), str(d / "res"), embedder=emb, n_splits=3)
    build_collection(get_client(str(d / "chroma")), "section", chunk_documents(load_docs(DOCS), "section"), emb)
    clf = Classifier(model_path=str(d / "clf.joblib"), embedder=emb)
    ret = Retriever(embedder=emb, chroma_path=str(d / "chroma"), docs_path=str(DOCS), strategy="section",
                    threshold=0.1)
    log = DecisionLog(str(d / "decisions.db"))
    return Pipeline(clf, ret, Router(settings=Settings(), confidence_threshold=0.8), log), d


def test_T10b_every_ticket_gets_a_logged_routing_decision(pipeline):
    pipe, d = pipeline
    ingest = load_tickets(d / "t.json")
    run = pipe.log.start_run(str(d / "t.json"), len(ingest.tickets), {})
    outcomes = pipe.process(run, ingest.tickets)
    assert len(outcomes) == len(ingest.tickets)
    rec = pipe.log.reconcile(run, [t.ticket_id for t in ingest.tickets])
    assert rec["reconciled"], rec
    empty = [o for o in outcomes if o.ticket_id == "T-EMPTY"][0]
    assert empty.routing.action == "escalate"
    assert empty.routing.rule in ("unreadable_ticket", "no_reliable_classification")
    rows = pipe.log.decisions(run, "T-001")
    assert {r["stage"] for r in rows} == {"classification", "routing"}
    assert all(r["reason"] for r in rows)


def test_T10c_processing_error_is_escalated_and_logged(pipeline):
    pipe, d = pipeline

    class Boom:
        threshold, doc_ids = 0.1, []

        def retrieve(self, text):
            raise RuntimeError("vector store unavailable")

    broken = Pipeline(pipe.classifier, Boom(), pipe.router, pipe.log)
    ingest = load_tickets(d / "t.json")
    run = pipe.log.start_run("t.json", 3, {})
    outcomes = broken.process(run, ingest.tickets[:3])
    assert all(o.routing.rule == "processing_error" for o in outcomes)
    rows = [r for r in pipe.log.decisions(run) if r["stage"] == "routing"]
    assert len(rows) == 3 and all("vector store unavailable" in r["error"] for r in rows)


def test_T10d_generation_stage_logged_and_outcomes(pipeline):
    """B-08: cleared tickets go to drafting; refusals and provider errors escalate with a note."""
    from src.generate import Generator
    from src.llm import LLMUnavailable
    from tests.test_generate import ANSWER, REFUSE, FakeLLM

    pipe, d = pipeline
    ingest = load_tickets(d / "t.json")
    for llm, final, rule in ((FakeLLM(ANSWER), "drafted", "answer_drafted"),
                             (FakeLLM(REFUSE), "escalate", "not_answerable"),
                             (FakeLLM(LLMUnavailable("down")), "escalate", "provider_unavailable")):
        p = Pipeline(pipe.classifier, pipe.retriever, pipe.router, pipe.log, Generator(llm=llm))
        run = pipe.log.start_run("t.json", len(ingest.tickets), {})
        outcomes = p.process(run, ingest.tickets)
        assert pipe.log.reconcile(run, [t.ticket_id for t in ingest.tickets])["reconciled"]
        sent = [o for o in outcomes if o.draft is not None]
        assert sent, "no ticket was cleared by routing - fixture too strict"
        assert all(o.final_action == final and o.final_rule == rule for o in sent)
        gen_rows = [r for r in pipe.log.decisions(run) if r["stage"] == "generation"]
        assert len(gen_rows) == len(sent) and all(r["prompt_version"].startswith("PR-02 v") for r in gen_rows)
        for o in outcomes:
            assert o.final_action in ("drafted", "escalate")      # never auto_respond before B-09
            if o.final_action == "escalate":
                assert o.acknowledgement and o.escalation_note and o.reply is None
            else:
                assert o.reply and "prepared automatically" in o.reply


def test_T05a_same_ticket_twice_gets_the_same_routing_decision(pipeline):
    """A5: the threshold is applied deterministically - same input, same decision."""
    pipe, d = pipeline
    ingest = load_tickets(d / "t.json")
    first = pipe.process(pipe.log.start_run("a", len(ingest.tickets), {}), ingest.tickets)
    second = pipe.process(pipe.log.start_run("b", len(ingest.tickets), {}), ingest.tickets)
    for a, b in zip(first, second):
        assert (a.routing.action, a.routing.rule, a.final_action) == (b.routing.action, b.routing.rule, b.final_action)
        assert a.classification.intent == b.classification.intent
        assert a.classification.intent_confidence == b.classification.intent_confidence
