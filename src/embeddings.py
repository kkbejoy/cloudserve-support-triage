"""Text embeddings with all-MiniLM-L6-v2 (sentence-transformers).

Used by the document index (B-03), retrieval (B-04) and later the classifier (B-06),
so tickets and articles are embedded by exactly the same model.
Vectors are L2-normalised, so cosine similarity = dot product.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Protocol, Sequence


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. The model downloads once (~90 MB) on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts), normalize_embeddings=True,
                                     show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=2)
def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformerEmbedder:
    """One shared model per process (loading it takes a few seconds)."""
    return SentenceTransformerEmbedder(model_name)


class ChromaEmbeddingFunction:
    """Adapter so Chroma uses our embedder (and never downloads its own default model)."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def __call__(self, input):  # noqa: A002 - Chroma requires the parameter name 'input'
        return self._embedder.embed(list(input))
