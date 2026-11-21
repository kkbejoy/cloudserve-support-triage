"""Settings read from the environment (.env). One place, so modules don't each read .env.

Every value has a safe default so tests and a clean checkout work without a .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # python-dotenv is in requirements.txt; tolerate its absence in minimal test envs
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_path: str = os.getenv("CHROMA_PATH", "./storage/chroma")
    docs_path: str = os.getenv("DOCS_PATH", "./data/documentation.json")
    # section | whole | fixed  (decision recorded in docs/decisions/chunking.md)
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "section")
    # chunks fetched before grouping by article
    retrieval_top_k: int = _int("RETRIEVAL_TOP_K", 12)
    # articles handed to generation at most
    retrieval_max_docs: int = _int("RETRIEVAL_MAX_DOCS", 3)
    # cosine similarity cut-off; below it retrieval returns nothing.
    # Placeholder until set from scripts/compare_chunking.py (B-04).
    relevance_threshold: float = _float("RELEVANCE_THRESHOLD", 0.35)
    # trained classifier (scripts/train_classifier.py) and its low-confidence floor:
    # below it the intent becomes 'unclear_request' (escalated). Starting value; see
    # docs/decisions/classifier.md.
    classifier_path: str = os.getenv("CLASSIFIER_PATH", "./storage/classifier.joblib")
    classifier_floor: float = _float("CLASSIFIER_FLOOR", 0.30)


def get_settings() -> Settings:
    return Settings()
