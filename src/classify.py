"""Intent and urgency classification (B-06, FR-02; acceptance criterion A3).

    clf = Classifier()                 # loads storage/classifier.joblib (scripts/train_classifier.py)
    result = clf.classify(ticket)
    result.intent, result.intent_confidence, result.intent_alternatives
    result.urgency, result.urgency_confidence

Reads the ticket TEXT only. Customer tier, region and language fluency are never used,
so the classifier cannot treat customer groups differently (NFR-06).

Never raises: an unreadable ticket, a confidence below the floor, or any internal error
returns intent 'unclear_request', which the router always escalates (FR-05).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings
from src.embeddings import Embedder, get_embedder
from src.models import Ticket

logger = logging.getLogger(__name__)

UNCLEAR = "unclear_request"
# ingestion flags that mean the text cannot be trusted for classification
BLOCKING_FLAGS = {"empty_body", "malformed"}


class Alternative(BaseModel):
    label: str
    confidence: float


class Classification(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticket_id: str
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    intent_alternatives: List[Alternative] = Field(default_factory=list)   # top 3, first = chosen
    urgency: str
    urgency_confidence: float = Field(ge=0.0, le=1.0)
    urgency_alternatives: List[Alternative] = Field(default_factory=list)
    fallback_reason: Optional[str] = None    # why 'unclear_request' was forced, if it was
    model_version: str = ""


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Soften (T > 1) or sharpen (T < 1) probabilities so stated confidence matches
    observed accuracy. T is fitted on out-of-fold predictions during training."""
    if not temperature or abs(temperature - 1.0) < 1e-9:
        return proba
    logits = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _top(proba_row: np.ndarray, classes: Sequence[str], k: int = 3) -> List[Alternative]:
    order = np.argsort(-proba_row)[:k]
    return [Alternative(label=str(classes[i]), confidence=round(float(proba_row[i]), 4)) for i in order]


class Classifier:
    """Loads the trained bundle once and classifies tickets in batches."""

    def __init__(self, model_path: Optional[str] = None, embedder: Optional[Embedder] = None,
                 floor: Optional[float] = None) -> None:
        s = get_settings()
        path = Path(model_path or s.classifier_path)
        if not path.exists():
            raise RuntimeError(f"Classifier model not found at {path}. "
                               f"Run: python -m scripts.train_classifier")
        self.bundle: Dict[str, Any] = joblib.load(path)
        self.feature = self.bundle["feature"]                  # "embedding" or "text"
        self.floor = s.classifier_floor if floor is None else floor
        self.version = self.bundle.get("version", "")
        self._embedder = embedder
        if self.feature == "embedding" and self._embedder is None:
            self._embedder = get_embedder(self.bundle.get("embedding_model", s.embedding_model))

    def _features(self, texts: List[str]):
        if self.feature == "embedding":
            return np.asarray(self._embedder.embed(texts), dtype=float)
        return texts

    def _proba(self, target: str, features) -> tuple:
        part = self.bundle[target]
        proba = part["model"].predict_proba(features)
        return apply_temperature(proba, part.get("temperature", 1.0)), list(part["model"].classes_)

    def _fallback(self, ticket: Ticket, reason: str) -> Classification:
        return Classification(ticket_id=ticket.ticket_id, intent=UNCLEAR, intent_confidence=0.0,
                              urgency="medium", urgency_confidence=0.0, fallback_reason=reason,
                              model_version=self.version)

    def classify_many(self, tickets: List[Ticket]) -> List[Classification]:
        """Classify a batch (one embedding call for all readable tickets)."""
        results: Dict[int, Classification] = {}
        readable = []
        for i, t in enumerate(tickets):
            blocking = BLOCKING_FLAGS.intersection(t.flags)
            if blocking or not t.text.strip():
                results[i] = self._fallback(t, f"unreadable ticket: {', '.join(sorted(blocking)) or 'no text'}")
            else:
                readable.append(i)
        if readable:
            try:
                feats = self._features([tickets[i].text for i in readable])
                p_int, c_int = self._proba("intent", feats)
                p_urg, c_urg = self._proba("urgency", feats)
                for row, i in enumerate(readable):
                    t = tickets[i]
                    alts = _top(p_int[row], c_int)
                    urg = _top(p_urg[row], c_urg)
                    intent, conf, reason = alts[0].label, alts[0].confidence, None
                    if conf < self.floor and intent != UNCLEAR:
                        reason = f"top intent confidence {conf:.2f} below floor {self.floor:.2f}"
                        intent = UNCLEAR
                    if not (0.0 <= conf <= 1.0) or math.isnan(conf):
                        raise ValueError(f"invalid confidence {conf}")
                    results[i] = Classification(
                        ticket_id=t.ticket_id, intent=intent, intent_confidence=conf,
                        intent_alternatives=alts, urgency=urg[0].label,
                        urgency_confidence=urg[0].confidence, urgency_alternatives=urg,
                        fallback_reason=reason, model_version=self.version)
            except Exception as exc:  # never crash the pipeline (A11): escalate instead
                logger.exception("Classifier failed; falling back to unclear_request")
                for i in readable:
                    results[i] = self._fallback(tickets[i], f"classifier error: {type(exc).__name__}")
        return [results[i] for i in range(len(tickets))]

    def classify(self, ticket: Ticket) -> Classification:
        return self.classify_many([ticket])[0]
