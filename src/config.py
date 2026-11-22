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
    # routing (B-07): below this intent confidence the ticket escalates. Chosen from
    # out-of-fold predictions - see docs/decisions/routing.md.
    confidence_threshold: float = _float("CONFIDENCE_THRESHOLD", 0.80)
    # kill switch: false = every ticket escalates, no automatic answers (Governance section 5)
    auto_respond_enabled: bool = os.getenv("AUTO_RESPOND_ENABLED", "true").strip().lower() not in (
        "false", "0", "no", "off")
    # intents that always go to a person (FR-05), whatever the confidence
    must_escalate_intents: str = os.getenv(
        "MUST_ESCALATE_INTENTS",
        "security_incident,compliance_request,feature_request,unclear_request")
    # decision log (B-10); sqlite:///<path>
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./storage/decisions.db")
    # language model for answer drafting (B-08). Provider: groq | openrouter | auto
    # (auto = Groq if GROQ_API_KEY is set, otherwise OpenRouter).
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    model_name: str = os.getenv("MODEL_NAME", "")
    llm_timeout_s: float = _float("LLM_TIMEOUT_S", 30.0)
    llm_max_retries: int = _int("LLM_MAX_RETRIES", 3)
    # minimum seconds between calls (free tiers limit requests per minute); empty = provider default
    llm_min_interval_s: str = os.getenv("LLM_MIN_INTERVAL_S", "")
    # after this many consecutive failed calls the provider is treated as down for the run (A11)
    llm_circuit_breaker: int = _int("LLM_CIRCUIT_BREAKER", 3)
    llm_cache_path: str = os.getenv("LLM_CACHE_PATH", "./storage/llm_cache.sqlite")
    # answer drafting (B-08): prompt version ("" = newest file in prompts/), and how much text
    # is sent: at most this many articles, and a 2nd/3rd article only if its score is within
    # the margin of the best one (fewer tokens; the best article answers most tickets)
    prompt_pr02_version: str = os.getenv("PROMPT_PR02_VERSION", "").strip()
    generation_max_articles: int = _int("GENERATION_MAX_ARTICLES", 2)
    generation_score_margin: float = _float("GENERATION_SCORE_MARGIN", 0.10)
    generation_max_tokens: int = _int("GENERATION_MAX_TOKENS", 400)
    # guardrails (B-09): the grounding check (PR-04) can use its own provider/model, which on
    # free tiers gives it its own daily allowance. Empty = same as answer drafting.
    grounding_provider: str = os.getenv("GROUNDING_PROVIDER", "").strip()
    grounding_model: str = os.getenv("GROUNDING_MODEL", "").strip()
    # true = a sentence the checker calls "partial" still passes (default: strict, it blocks)
    grounding_allow_partial: bool = os.getenv("GROUNDING_ALLOW_PARTIAL", "false").strip().lower() in (
        "true", "1", "yes", "on")
    # tickets blocked by the instruction-integrity check are kept here for review (git-ignored)
    review_queue_path: str = os.getenv("REVIEW_QUEUE_PATH", "./storage/review_queue.jsonl")
    # use the LLM (PR-03) for escalation notes; false = deterministic template (no API calls)
    escalation_note_llm: bool = os.getenv("ESCALATION_NOTE_LLM", "false").strip().lower() in (
        "true", "1", "yes", "on")

    @property
    def must_escalate(self) -> frozenset:
        return frozenset(i.strip() for i in self.must_escalate_intents.split(",") if i.strip())

    @property
    def database_path(self) -> str:
        return self.database_url.replace("sqlite:///", "", 1)


def get_settings() -> Settings:
    return Settings()
