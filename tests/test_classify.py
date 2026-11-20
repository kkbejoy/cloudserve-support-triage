"""Tests for the classifier (B-06, FR-02 / test T-02).
A small synthetic ticket set and the fake embedder keep these offline and fast."""
import json
import random

import pytest

from scripts.train_classifier import text_groups, train
from src.classify import UNCLEAR, Classifier, apply_temperature
from src.models import Ticket
from tests.fake_embedder import FakeEmbedder

VOCAB = {
    "authentication_failure": ["login", "password", "credentials", "locked", "signin", "mfa"],
    "billing_query": ["invoice", "charge", "billing", "payment", "refund", "overage"],
    "webhook_issue": ["webhook", "delivery", "retry", "endpoint", "signature", "events"],
}
URGENCY = ["low", "medium", "high"]


def _tickets(n_per_class=14, seed=7):
    rnd = random.Random(seed)
    out, i = [], 0
    for intent, words in VOCAB.items():
        for _ in range(n_per_class):
            body = " ".join(rnd.sample(words, 4)) + f" ticket number {i} please help"
            out.append({"ticket_id": f"T-{i:03d}", "channel": "email", "subject": "", "body": body,
                        "customer_tier": "business", "customer_region": "europe",
                        "language_fluency": "fluent",
                        "labels": {"intent": intent, "urgency": URGENCY[i % 3],
                                   "expected_route": "auto_respond", "must_not_auto_respond": False,
                                   "answerable_from_docs": True}})
            i += 1
    return out


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("clf")
    path = d / "tickets.json"
    path.write_text(json.dumps(_tickets()), encoding="utf-8")
    model = d / "classifier.joblib"
    summary = train(str(path), str(model), str(d / "results"), embedder=FakeEmbedder(), n_splits=3)
    return summary, model, d


def test_T02a_training_compares_models_and_saves_outputs(trained):
    summary, model, d = trained
    assert model.exists()
    assert set(summary["results"]) == {"logreg_emb", "knn_emb", "tfidf_lr"}
    assert summary["meta"]["chosen"] in summary["results"]
    files = {p.name.split("_2")[0] for p in (d / "results").iterdir()}
    assert {"classifier_comparison", "classifier_oof"} <= files
    intent = summary["results"][summary["meta"]["chosen"]]["intent"]
    assert intent["macro_precision"] > 0.8   # easy synthetic classes


def test_T02b_output_shape_and_confidence_range(trained):
    _, model, _ = trained
    clf = Classifier(model_path=str(model), embedder=FakeEmbedder(), floor=0.0)
    r = clf.classify(Ticket(ticket_id="X1", channel="chat", text="my login password is locked", body="x"))
    assert r.intent == "authentication_failure"
    assert 0.0 <= r.intent_confidence <= 1.0 and 0.0 <= r.urgency_confidence <= 1.0
    assert len(r.intent_alternatives) == 3 and r.intent_alternatives[0].label == r.intent
    assert r.urgency in URGENCY and r.fallback_reason is None


def test_T02c_unreadable_or_low_confidence_falls_back_to_unclear(trained):
    _, model, _ = trained
    clf = Classifier(model_path=str(model), embedder=FakeEmbedder(), floor=0.99)
    empty = Ticket(ticket_id="E1", channel="email", text="", flags=["empty_body"])
    r = clf.classify(empty)
    assert r.intent == UNCLEAR and r.intent_confidence == 0.0 and "unreadable" in r.fallback_reason
    low = clf.classify(Ticket(ticket_id="L1", channel="email", text="something completely different", body="x"))
    assert low.intent == UNCLEAR and "below floor" in low.fallback_reason


def test_T02d_same_input_same_output(trained):
    _, model, _ = trained
    clf = Classifier(model_path=str(model), embedder=FakeEmbedder())
    t = Ticket(ticket_id="D1", channel="forum", text="webhook delivery retry failing", body="x")
    assert clf.classify(t) == clf.classify(t)


def test_T02e_internal_error_never_raises(trained):
    _, model, _ = trained

    class Broken(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("model unavailable")

    clf = Classifier(model_path=str(model), embedder=Broken())
    r = clf.classify(Ticket(ticket_id="B1", channel="email", text="invoice charge", body="x"))
    assert r.intent == UNCLEAR and "classifier error" in r.fallback_reason


def test_T02f_identical_texts_share_a_group_and_temperature_keeps_ranking():
    g = text_groups(["Hello, World!", "hello world", "something else"])
    assert g[0] == g[1] != g[2]
    import numpy as np
    p = np.array([[0.7, 0.2, 0.1]])
    soft = apply_temperature(p, 2.0)
    assert soft.argmax() == 0 and soft.max() < 0.7 and abs(soft.sum() - 1) < 1e-9
