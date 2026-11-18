"""Build the documentation vector index (B-03). Run once after cloning, and again if
documentation.json changes:

    python -m scripts.build_index                      # strategy from .env (default: section)
    python -m scripts.build_index --strategy all       # build all three, for the comparison

Writes the Chroma index under CHROMA_PATH and a manifest (storage/index_manifest.json)
recording which documentation file, model and strategy built it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.chunking import STRATEGIES, chunk_documents, load_docs
from src.config import get_settings
from src.embeddings import get_embedder
from src.vector_store import build_collection, get_client

logger = logging.getLogger("build_index")


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(docs_path: str, chroma_path: str, strategies, embedder) -> dict:
    """Build one collection per strategy; return the manifest written to disk."""
    docs = load_docs(docs_path)
    client = get_client(chroma_path)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "docs_path": str(docs_path),
        "docs_sha256": file_sha256(docs_path),
        "articles": len(docs),
        "embedding_model": getattr(embedder, "name", "unknown"),
        "collections": {},
    }
    for strategy in strategies:
        chunks = chunk_documents(docs, strategy)
        build_collection(client, strategy, chunks, embedder,
                         extra_metadata={"docs_sha256": manifest["docs_sha256"]})
        manifest["collections"][strategy] = {"chunks": len(chunks)}
        logger.info("Built '%s': %d articles -> %d chunks", strategy, len(docs), len(chunks))
    out = Path(chroma_path).parent / "index_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", default=s.docs_path, help="path to documentation.json")
    p.add_argument("--chroma-path", default=s.chroma_path)
    p.add_argument("--strategy", default=s.chunk_strategy, choices=list(STRATEGIES) + ["all"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    manifest = build(args.docs, args.chroma_path, strategies, get_embedder(s.embedding_model))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
