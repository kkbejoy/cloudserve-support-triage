"""Retrieval over the documentation index (B-04, FR-03).

    retriever = Retriever()                    # opens the index built by scripts/build_index.py
    result = retriever.retrieve(ticket.text)
    result.articles  -> up to RETRIEVAL_MAX_DOCS full articles above the relevance threshold
    result.hits      -> the raw chunk matches (for the decision log)

Steps: embed the ticket -> nearest chunks -> group by doc_id (best score per article)
-> keep articles above the threshold -> attach each article's FULL text by doc_id.
If nothing clears the threshold, `articles` is empty: returning nothing is a valid
answer, and the router escalates the ticket.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.chunking import load_docs
from src.config import get_settings
from src.embeddings import Embedder, get_embedder
from src.vector_store import get_client, open_collection, query


class RetrievedArticle(BaseModel):
    doc_id: str
    title: str
    score: float                      # best cosine similarity among this article's chunks
    matched_chunk_id: str             # which chunk matched best, e.g. DOC-AUTH-001#symptoms
    matched_section: str
    content: str                      # full article text, for the answer step
    related_docs: List[str] = Field(default_factory=list)  # for the escalation package only


class RetrievalResult(BaseModel):
    articles: List[RetrievedArticle] = Field(default_factory=list)
    hits: List[Dict] = Field(default_factory=list)          # raw chunk hits (no text), for logging
    threshold: float
    strategy: str

    @property
    def doc_ids(self) -> List[str]:
        return [a.doc_id for a in self.articles]

    @property
    def top_score(self) -> float:
        return max((h["score"] for h in self.hits), default=0.0)


class Retriever:
    def __init__(self, embedder: Optional[Embedder] = None, chroma_path: Optional[str] = None,
                 docs_path: Optional[str] = None, strategy: Optional[str] = None,
                 threshold: Optional[float] = None, top_k: Optional[int] = None,
                 max_docs: Optional[int] = None) -> None:
        s = get_settings()
        self.embedder = embedder or get_embedder(s.embedding_model)
        self.strategy = strategy or s.chunk_strategy
        self.threshold = s.relevance_threshold if threshold is None else threshold
        self.top_k = top_k or s.retrieval_top_k
        self.max_docs = max_docs or s.retrieval_max_docs
        self._collection = open_collection(get_client(chroma_path or s.chroma_path),
                                           self.strategy, self.embedder)
        # full articles by doc_id: how a matched section is linked back to its article
        self.docs_by_id = {d["doc_id"]: d for d in load_docs(docs_path or s.docs_path)}

    def ranked_articles(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """(best hit per article ranked by score, all raw chunk hits). No threshold applied;
        used by retrieve() and by the chunking comparison."""
        if not text or not text.strip():
            return [], []
        hits = query(self._collection, self.embedder.embed([text])[0], self.top_k)
        best: Dict[str, Dict] = {}
        for h in hits:  # hits arrive sorted by score, so the first per doc is its best
            if h["doc_id"] not in best:
                best[h["doc_id"]] = h
        return sorted(best.values(), key=lambda h: h["score"], reverse=True), hits

    def retrieve(self, text: str) -> RetrievalResult:
        best_per_doc, hits = self.ranked_articles(text)
        articles = []
        for h in best_per_doc:
            if h["score"] < self.threshold or len(articles) >= self.max_docs:
                continue
            doc = self.docs_by_id.get(h["doc_id"])
            if doc is None:  # index and documentation file out of step; never invent a source
                continue
            articles.append(RetrievedArticle(
                doc_id=h["doc_id"], title=doc.get("title", ""), score=h["score"],
                matched_chunk_id=h["chunk_id"], matched_section=h["section"] or "",
                content=doc.get("content", ""), related_docs=list(doc.get("related_docs") or []),
            ))
        log_hits = [{k: h[k] for k in ("chunk_id", "doc_id", "section", "score")} for h in hits]
        return RetrievalResult(articles=articles, hits=log_hits,
                               threshold=self.threshold, strategy=self.strategy)
