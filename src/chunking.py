"""Split the knowledge-base articles into chunks for the vector index (B-03, FR-03).

Three strategies, compared in scripts/compare_chunking.py:
  section - one chunk per '## ' section (Symptoms, Common causes, Resolution, Notes),
            each prefixed with the article title and 'applies to'. Recommended.
  whole   - one chunk per article.
  fixed   - 800-character chunks with 120 overlap (the Setup Guide's starting point).

Every chunk carries metadata linking it back to its article (doc_id), so retrieval
can group hits by article and fetch the full text for answering.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

STRATEGIES = ("section", "whole", "fixed")
SECTION_ORDER = {"symptoms": 0, "common causes": 1, "resolution": 2, "notes": 3}
_SECTION_SPLIT = re.compile(r"^## ", flags=re.MULTILINE)


@dataclass
class Chunk:
    id: str                      # unique, e.g. DOC-AUTH-001#symptoms
    text: str                    # what gets embedded
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_docs(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read documentation.json (a list of articles)."""
    with Path(path).open(encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list):
        raise ValueError(f"Documentation file must contain a JSON list: {path}")
    return docs


def _base_metadata(doc: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    """Article-level fields copied onto every chunk. Chroma metadata must be
    str / int / float / bool, so the related_docs list becomes a comma string."""
    return {
        "doc_id": doc["doc_id"],
        "title": doc.get("title", ""),
        "category": doc.get("category", ""),
        "applies_to": doc.get("applies_to", ""),
        "related_docs": ",".join(doc.get("related_docs") or []),
        "last_reviewed_days_ago": int(doc.get("last_reviewed_days_ago") or 0),
        "strategy": strategy,
    }


def chunk_by_section(doc: Dict[str, Any]) -> List[Chunk]:
    """One chunk per '## ' heading, prefixed with title and applies-to.
    Falls back to the whole article if it has no '## ' headings."""
    content = doc.get("content", "")
    parts = _SECTION_SPLIT.split(content)[1:]
    if not parts:
        return chunk_whole(doc, strategy="section")
    chunks = []
    for position, part in enumerate(parts):
        heading, _, body = part.partition("\n")
        heading, body = heading.strip(), body.strip()
        slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
        text = (f"{doc.get('title', '')}\nApplies to: {doc.get('applies_to', '')}\n"
                f"Section: {heading}\n{body}")
        meta = _base_metadata(doc, "section")
        meta.update({
            "chunk_id": f"{doc['doc_id']}#{slug}",
            "section": heading,
            "section_order": SECTION_ORDER.get(heading.lower(), position),
        })
        chunks.append(Chunk(id=meta["chunk_id"], text=text, metadata=meta))
    return chunks


def chunk_whole(doc: Dict[str, Any], strategy: str = "whole") -> List[Chunk]:
    """The whole article as one chunk. (MiniLM reads ~256 tokens, so the end of
    the longest articles is not seen by the embedding.)"""
    meta = _base_metadata(doc, strategy)
    meta.update({"chunk_id": f"{doc['doc_id']}#all", "section": "all", "section_order": 0})
    return [Chunk(id=meta["chunk_id"], text=doc.get("content", ""), metadata=meta)]


def _split_fixed(text: str, size: int, overlap: int) -> List[str]:
    """Character windows of `size` with `overlap`, breaking at whitespace where possible."""
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > start else end
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_fixed(doc: Dict[str, Any], size: int = 800, overlap: int = 120) -> List[Chunk]:
    """Fixed-size baseline from the Setup Guide (chunk_size=800, chunk_overlap=120)."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        texts = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap
                                               ).split_text(doc.get("content", ""))
    except ImportError:
        texts = _split_fixed(doc.get("content", ""), size, overlap)
    chunks = []
    for i, text in enumerate(texts):
        meta = _base_metadata(doc, "fixed")
        meta.update({"chunk_id": f"{doc['doc_id']}#part{i}", "section": f"part {i}",
                     "section_order": i})
        chunks.append(Chunk(id=meta["chunk_id"], text=text, metadata=meta))
    return chunks


def chunk_documents(docs: List[Dict[str, Any]], strategy: str = "section") -> List[Chunk]:
    """Chunk every article with the chosen strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose one of {STRATEGIES}")
    splitter = {"section": chunk_by_section, "whole": chunk_whole, "fixed": chunk_fixed}[strategy]
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(splitter(doc))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk ids; check doc_id uniqueness in documentation.json")
    return chunks
