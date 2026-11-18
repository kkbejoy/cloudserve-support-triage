"""Tests for chunking (B-03) and retrieval (B-04, FR-03 / test T-03).
Uses a fake embedder and a temporary Chroma folder, so no model download is needed."""
from pathlib import Path

import pytest

from src.chunking import chunk_by_section, chunk_documents, chunk_fixed, chunk_whole, load_docs
from src.retrieve import Retriever
from src.vector_store import build_collection, get_client
from tests.fake_embedder import FakeEmbedder

FIX = Path(__file__).parent / "fixtures"
DOCS = FIX / "docs_small.json"


@pytest.fixture(scope="module")
def docs():
    return load_docs(DOCS)


def test_T03a_section_chunks_have_four_sections_with_links(docs):
    chunks = chunk_by_section(docs[0])
    assert [c.metadata["section"] for c in chunks] == ["Symptoms", "Common causes", "Resolution", "Notes"]
    assert [c.metadata["section_order"] for c in chunks] == [0, 1, 2, 3]
    assert {c.metadata["doc_id"] for c in chunks} == {"DOC-T-LOGIN"}   # the link back to the article
    assert chunks[0].id == "DOC-T-LOGIN#symptoms"
    assert chunks[0].text.startswith("Fixing login password errors\nApplies to: Console\nSection: Symptoms")
    assert chunks[0].metadata["related_docs"] == "DOC-T-INVOICE"       # stored as a string for Chroma
    resolution = chunks[2].text
    assert "1. wait thirty minutes" in resolution and "2. clear browser cookies" in resolution


def test_T03b_whole_and_fixed_strategies(docs):
    assert len(chunk_whole(docs[0])) == 1
    pieces = chunk_fixed(docs[0], size=120, overlap=20)
    assert len(pieces) > 1 and all(len(p.text) <= 140 for p in pieces)
    all_ids = [c.id for c in chunk_documents(docs, "section")]
    assert len(all_ids) == len(set(all_ids)) == 12


@pytest.fixture(scope="module")
def retriever(tmp_path_factory, docs):
    path = str(tmp_path_factory.mktemp("chroma"))
    emb = FakeEmbedder()
    build_collection(get_client(path), "section", chunk_documents(docs, "section"), emb)
    return Retriever(embedder=emb, chroma_path=path, docs_path=str(DOCS), strategy="section",
                     threshold=0.2, top_k=12, max_docs=3)


def test_T03c_question_finds_right_article_with_full_text(retriever):
    result = retriever.retrieve("My login keeps failing with invalid password and a locked account")
    assert result.articles and result.articles[0].doc_id == "DOC-T-LOGIN"
    top = result.articles[0]
    assert "## Resolution" in top.content and "## Notes" in top.content   # full article returned
    assert top.matched_chunk_id.startswith("DOC-T-LOGIN#")
    assert top.related_docs == ["DOC-T-INVOICE"]


def test_T03d_every_returned_doc_id_exists_in_corpus(retriever, docs):
    known = {d["doc_id"] for d in docs}
    result = retriever.retrieve("webhook events delivered twice")
    assert result.doc_ids and set(result.doc_ids) <= known
    assert result.articles[0].doc_id == "DOC-T-WEBHOOK"


def test_T03e_irrelevant_or_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("When will you add a dark mode to the mobile app?").articles == []
    empty = retriever.retrieve("   ")
    assert empty.articles == [] and empty.hits == []


def test_T03f_same_query_same_result(retriever):
    q = "invoice total is higher than expected this month"
    assert retriever.retrieve(q).doc_ids == retriever.retrieve(q).doc_ids
