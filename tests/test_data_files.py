"""The data shipped in data/ loads cleanly (A1, A2: a clean checkout has what the system needs)."""
import json
from collections import Counter
from pathlib import Path

import pytest

from src.chunking import chunk_documents, load_docs
from src.ingest import load_tickets
from src.models import CHANNELS

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.parametrize("name,count", [("development_tickets.json", 500), ("validation_tickets.json", 80)])
def test_T02d_ticket_files_load_with_all_four_channels(name, count):
    res = load_tickets(DATA / name)
    assert len(res.tickets) == count and len(res.answer_key) == count
    assert set(Counter(t.channel for t in res.tickets)) == set(CHANNELS)
    assert res.problems == []                     # the shipped files are clean


def test_T03d_documentation_has_29_articles_and_section_chunks():
    docs = load_docs(str(DATA / "documentation.json"))
    assert len(docs) == 29 and len({d["doc_id"] for d in docs}) == 29
    chunks = chunk_documents(docs, "section")
    assert len(chunks) > len(docs) and all(c.metadata["doc_id"] for c in chunks)


def test_T15a_reference_responses_point_at_real_articles():
    ids = {d["doc_id"] for d in load_docs(str(DATA / "documentation.json"))}
    refs = json.loads((DATA / "ground_truth_responses.json").read_text(encoding="utf-8"))
    assert len(refs) == 200 and all(set(r["expected_doc_ids"]) <= ids for r in refs)
