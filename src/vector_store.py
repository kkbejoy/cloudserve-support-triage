"""Thin wrapper around Chroma so the rest of the code never imports chromadb directly.

One collection per chunking strategy (docs_section, docs_whole, docs_fixed), stored on
disk under CHROMA_PATH and compared with cosine similarity. Swapping the vector store
later means changing only this file.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

# No usage data leaves the machine; also silences noisy telemetry errors from
# some posthog versions bundled with chromadb 0.4.x.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

from src.chunking import Chunk  # noqa: E402
from src.embeddings import ChromaEmbeddingFunction, Embedder  # noqa: E402


def collection_name(strategy: str) -> str:
    return f"docs_{strategy}"


def get_client(path: str):
    """Persistent local Chroma client; anonymous telemetry switched off."""
    return chromadb.PersistentClient(path=path,
                                     settings=ChromaSettings(anonymized_telemetry=False))


def build_collection(client, strategy: str, chunks: List[Chunk], embedder: Embedder,
                     extra_metadata: Optional[Dict[str, Any]] = None):
    """Delete and rebuild the collection for `strategy`, so it never mixes old and new chunks."""
    name = collection_name(strategy)
    try:
        client.delete_collection(name)
    except Exception:  # collection did not exist yet
        pass
    meta = {"hnsw:space": "cosine", "embedding_model": getattr(embedder, "name", "unknown")}
    meta.update(extra_metadata or {})
    collection = client.create_collection(name=name, metadata=meta,
                                          embedding_function=ChromaEmbeddingFunction(embedder))
    if chunks:
        collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embedder.embed([c.text for c in chunks]),
        )
    return collection


def open_collection(client, strategy: str, embedder: Embedder):
    """Open an existing collection; raises a clear error if the index was never built."""
    try:
        return client.get_collection(collection_name(strategy),
                                     embedding_function=ChromaEmbeddingFunction(embedder))
    except Exception as exc:
        raise RuntimeError(
            f"Vector index '{collection_name(strategy)}' not found. "
            "Run: python -m scripts.setup   (builds it; or python -m scripts.build_index "
            f"--strategy {strategy})") from exc


def query(collection, query_embedding: List[float], n_results: int) -> List[Dict[str, Any]]:
    """Nearest chunks as dicts: chunk_id, doc_id, section, score (cosine similarity), text, metadata."""
    count = collection.count()
    if count == 0:
        return []
    res = collection.query(query_embeddings=[query_embedding], n_results=min(n_results, count),
                           include=["documents", "metadatas", "distances"])
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        hits.append({
            "chunk_id": cid,
            "doc_id": meta.get("doc_id"),
            "section": meta.get("section"),
            "score": round(1.0 - float(dist), 4),  # cosine distance -> similarity
            "text": doc,
            "metadata": meta,
        })
    return hits
