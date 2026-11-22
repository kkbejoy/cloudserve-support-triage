"""One-time setup after cloning: everything the system needs that is not stored in git.

    python -m scripts.setup             # build what is missing, then a quick self-test
    python -m scripts.setup --force     # rebuild the index and retrain the classifier
    python -m scripts.setup --check-llm # also make one tiny call to the model provider

What it does, in order:
  1. checks the Python version and that the pinned packages are installed
  2. checks .env (creates it from .env.example if missing) and that an API key is filled in
  3. checks the data files are present
  4. builds the documentation search index (storage/chroma) - skipped if it exists and the
     documentation has not changed
  5. trains the intent/urgency classifier (storage/classifier.joblib) - skipped if it exists
  6. self-test: one retrieval and one classification, no model-provider call

The first run downloads the embedding model all-MiniLM-L6-v2 (about 90 MB) from Hugging Face
and takes a few minutes; later runs take seconds. storage/ is not in git: it is rebuilt from
the data in data/, and the builds are deterministic (same data and settings, same result).

Exit code 0 = ready (a missing API key is a warning: routing-only runs still work),
1 = something must be fixed first (the message says what).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable, List, Optional

# Versions the system was built and tested with (requirements.txt pins them all).
KEY_PACKAGES = ["chromadb", "sentence-transformers", "torch", "transformers", "scikit-learn",
                "pydantic", "requests", "python-dotenv"]
CLASSIFIER_MODEL = "knn_emb"   # chosen in docs/decisions/classifier.md; fixed here so a rebuild matches
SELF_TEST_TEXT = "I keep getting invalid credentials when I log in, even though my password is right."
REPORTS_DIR = "storage/setup_reports"


class SetupError(Exception):
    """A problem the user must fix before the system can run."""


def say(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


# --- 1. environment ------------------------------------------------------------------------------
def check_python(version=None) -> None:
    v = tuple(version or sys.version_info)[:3]
    if v < (3, 10):
        raise SetupError(f"Python {v[0]}.{v[1]} found; 3.10 or newer is needed (built and tested on 3.11).")
    say("python", ".".join(str(x) for x in v))


def check_packages(packages: List[str] = KEY_PACKAGES) -> None:
    missing, found = [], []
    for name in packages:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise SetupError(f"Packages not installed: {', '.join(missing)}. Run: pip install -r requirements.txt")
    say("packages", ", ".join(found))


# --- 2. configuration ----------------------------------------------------------------------------
def check_env(env_path: Path = Path(".env"), example: Path = Path(".env.example")) -> bool:
    """True if a provider key is set. Creates .env from .env.example when it is missing."""
    if not env_path.exists():
        if not example.exists():
            raise SetupError(".env and .env.example are both missing - is this the project folder?")
        shutil.copyfile(example, env_path)
        say("config", f"created {env_path} from {example}. Put your GROQ_API_KEY in it before running "
                      "answer drafting (the index and classifier do not need it).")
        return False
    from src.llm import _key_from_env   # reads os.environ, which config.py filled from .env
    import src.config  # noqa: F401  (loads .env)
    keys = [k for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY") if _key_from_env(k)]
    if keys:
        say("config", f".env found; key set for: {', '.join(keys)} (value not shown)")
        return True
    say("config", "WARNING: no API key in .env (GROQ_API_KEY or OPENROUTER_API_KEY). Answer drafting "
                  "and guardrails need one; routing-only runs (--no-generate) work without it.")
    return False


def check_data(paths: List[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SetupError(f"Data files missing: {', '.join(missing)}. They are part of the repository - "
                         "check the clone completed.")
    say("data", ", ".join(paths))


# --- 4. search index -----------------------------------------------------------------------------
def index_is_current(chroma_path: str, docs_path: str, strategy: str, embedder) -> bool:
    from scripts.build_index import file_sha256
    from src.vector_store import get_client, open_collection
    manifest = Path(chroma_path).parent / "index_manifest.json"
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        open_collection(get_client(chroma_path), strategy, embedder)
    except Exception:
        return False
    return info.get("docs_sha256") == file_sha256(docs_path) and strategy in info.get("collections", {})


def ensure_index(chroma_path: str, docs_path: str, strategy: str, embedder, force: bool = False) -> str:
    from scripts.build_index import build
    if not force and index_is_current(chroma_path, docs_path, strategy, embedder):
        say("index", f"up to date ({chroma_path}, strategy '{strategy}') - skipped")
        return "skipped"
    say("index", f"building from {docs_path} (strategy '{strategy}') ...")
    m = build(docs_path, chroma_path, [strategy], embedder)
    say("index", f"built: {m['articles']} articles -> {m['collections'][strategy]['chunks']} chunks")
    return "built"


# --- 5. classifier -------------------------------------------------------------------------------
def classifier_is_usable(model_path: str) -> bool:
    try:
        import joblib
        bundle = joblib.load(model_path)
        return {"intent", "urgency", "version"} <= set(bundle)
    except Exception:
        return False


def ensure_classifier(model_path: str, tickets_path: str, embedder, force: bool = False,
                      folds: int = 5) -> str:
    from scripts.train_classifier import train
    if not force and classifier_is_usable(model_path):
        say("classifier", f"found ({model_path}) - skipped")
        return "skipped"
    say("classifier", f"training on {tickets_path} (compares 3 models with {folds}-fold cross-validation, "
                      f"keeps '{CLASSIFIER_MODEL}') ...")
    s = train(tickets_path, model_path, REPORTS_DIR, embedder=embedder, n_splits=folds,
              force_model=CLASSIFIER_MODEL)
    r = s["results"][CLASSIFIER_MODEL]["intent"]
    say("classifier", f"saved {model_path} ({s['meta']['chosen']}; out-of-fold intent precision "
                      f"{r['macro_precision']:.3f}); report in {REPORTS_DIR}/")
    return "built"


# --- 6. self-test --------------------------------------------------------------------------------
def self_test(embedder, chroma_path: Optional[str] = None, docs_path: Optional[str] = None,
              model_path: Optional[str] = None, strategy: Optional[str] = None,
              threshold: Optional[float] = None, text: str = SELF_TEST_TEXT) -> None:
    from src.classify import Classifier
    from src.models import Ticket
    from src.retrieve import Retriever
    r = Retriever(embedder=embedder, chroma_path=chroma_path, docs_path=docs_path, strategy=strategy,
                  threshold=threshold).retrieve(text)
    if not r.articles:
        raise SetupError("Self-test: retrieval returned no article for a login question. "
                         "Rebuild with: python -m scripts.setup --force")
    c = Classifier(model_path=model_path, embedder=embedder).classify_many(
        [Ticket(ticket_id="SETUP-1", channel="email", subject="", body=text, text=text)])[0]
    say("self-test", f"retrieval -> {r.articles[0].doc_id} (score {r.articles[0].score:.2f}); "
                     f"classifier -> {c.intent} ({c.intent_confidence:.2f})")


def check_llm() -> None:
    from src.llm import LLMUnavailable, get_llm
    llm = get_llm()
    try:
        llm.check()
    except LLMUnavailable as exc:
        raise SetupError(f"Model provider check failed for {llm.name}: {str(exc)[:200]}")
    say("llm", f"{llm.name} answered")


# --- main ----------------------------------------------------------------------------------------
def run(force: bool = False, llm: bool = False, embedder=None,
        get_embedder: Optional[Callable] = None) -> int:
    from src.config import get_settings
    s = get_settings()
    check_python()
    check_packages()
    has_key = check_env()
    check_data([s.docs_path, "data/development_tickets.json"])
    if embedder is None:
        say("embeddings", f"loading {s.embedding_model} (first run downloads about 90 MB) ...")
        from src.embeddings import get_embedder as _ge
        embedder = (get_embedder or _ge)(s.embedding_model)
    ensure_index(s.chroma_path, s.docs_path, s.chunk_strategy, embedder, force)
    ensure_classifier(s.classifier_path, "data/development_tickets.json", embedder, force)
    self_test(embedder)
    if llm:
        if not has_key:
            raise SetupError("--check-llm needs an API key in .env")
        check_llm()
    print("\nReady.")
    print("  Full run:   python -m evaluation.harness --input data/validation_tickets.json "
          "--output evaluation/results/")
    print("  Tests:      python -m pytest")
    if not has_key:
        print("  (Add an API key to .env for answer drafting; until then use --no-generate.)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="rebuild the index and retrain the classifier")
    p.add_argument("--check-llm", action="store_true", help="also make one tiny call to the model provider")
    args = p.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return run(force=args.force, llm=args.check_llm)
    except SetupError as exc:
        print(f"\nSETUP STOPPED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
