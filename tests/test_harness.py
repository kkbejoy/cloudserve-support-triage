"""Evaluation harness (B-05; FR-11, A9, A10; test T-11). Offline with the fake embedder."""
import json

import pytest

from evaluation.harness import main, run
from scripts.train_classifier import train
from src.chunking import chunk_documents, load_docs
from src.classify import Classifier
from src.config import Settings
from src.logging_store import DecisionLog
from src.retrieve import Retriever
from src.route import Router
from src.vector_store import build_collection, get_client
from tests.fake_embedder import FakeEmbedder
from tests.test_classify import _tickets
from tests.test_pipeline_log import DOCS


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    d = tmp_path_factory.mktemp("harness")
    emb = FakeEmbedder()
    tickets = _tickets(n_per_class=12)
    for t in tickets[:6]:
        t["labels"].update({"expected_doc_ids": ["DOC-T-LOGIN"]})
    tickets.append({"ticket_id": "BAD", "channel": "sms", "body": ""})      # malformed record
    tickets.append("not even an object")
    (d / "in.json").write_text(json.dumps(tickets), encoding="utf-8")
    train(str(d / "in.json"), str(d / "clf.joblib"), str(d / "res"), embedder=emb, n_splits=3)
    build_collection(get_client(str(d / "chroma")), "section", chunk_documents(load_docs(DOCS), "section"), emb)

    def make():
        return dict(classifier=Classifier(model_path=str(d / "clf.joblib"), embedder=emb),
                    retriever=Retriever(embedder=emb, chroma_path=str(d / "chroma"), docs_path=str(DOCS),
                                        strategy="section", threshold=0.1),
                    router=Router(settings=Settings(), confidence_threshold=0.8),
                    log=DecisionLog(str(d / "decisions.db")))
    return d, make


def test_T11a_full_run_processes_every_ticket_and_reconciles(parts):
    d, make = parts
    m = run(str(d / "in.json"), str(d / "out"), **make())
    n = len(json.loads((d / "in.json").read_text()))
    v, g = m["volume"], m["governance"]
    assert v["tickets_processed"] == n
    assert v["answered_automatically"] + v["pending_answer_step"] + v["escalated"] + v["blocked_by_guardrails"] == n
    assert g["log_reconciled"] and g["tickets_with_logged_decision"] == n
    assert v["answered_automatically"] == 0          # nothing is sent before the answer step exists
    assert m["business"]["first_contact_resolution"] == 0.0
    out = next((d / "out").iterdir())
    lines = (out / "responses.jsonl").read_text().splitlines()
    assert len(lines) == n and all("true_intent" not in ln for ln in lines)   # labels not leaked to output
    assert (out / "metrics.md").exists() and "## Governance" in (out / "metrics.md").read_text()
    assert set(m["by_group"]) == {"channel", "customer_tier", "customer_region", "language_fluency"}
    assert m["technical"]["intent_classification"]["labelled_tickets"] > 0


def test_T11b_unlabelled_file_still_runs(parts, tmp_path):
    d, make = parts
    raw = [{k: v for k, v in t.items() if k != "labels"} for t in _tickets(n_per_class=2)]
    (tmp_path / "nolabels.json").write_text(json.dumps(raw), encoding="utf-8")
    m = run(str(tmp_path / "nolabels.json"), str(tmp_path / "out"), **make())
    assert m["volume"]["tickets_processed"] == len(raw)
    assert m["technical"]["intent_classification"] == {"labelled_tickets": 0}
    assert m["governance"]["log_reconciled"]


def test_T11c_unreadable_input_exits_with_code_2(tmp_path, capsys):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert main(["--input", str(tmp_path / "bad.json"), "--output", str(tmp_path / "o")]) == 2
    assert "ERROR" in capsys.readouterr().err
