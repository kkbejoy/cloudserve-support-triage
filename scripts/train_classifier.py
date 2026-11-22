"""Train and compare intent/urgency classifiers on the DEVELOPMENT tickets (B-06, FR-02).

    python -m scripts.train_classifier                       # compare, choose, save
    python -m scripts.train_classifier --model logreg_emb    # force a model

What it does
 1. Loads tickets with src.ingest (text only; labels from the answer key).
 2. Embeds every ticket once with MiniLM.
 3. Grouped, stratified 5-fold cross-validation: tickets with identical wording stay in the
    same fold, otherwise repeated texts would inflate the scores.
 4. Compares three models for intent and urgency:
      logreg_emb  logistic regression on MiniLM embeddings   (main candidate)
      knn_emb     10 nearest neighbours on MiniLM embeddings
      tfidf_lr    logistic regression on word/bigram counts   (baseline)
 5. Measures precision/recall per class, top-3 accuracy, most-confused pairs and
    calibration (expected calibration error, ECE). Fits a temperature on the out-of-fold
    predictions so stated confidence matches observed accuracy.
 6. Chooses the model with the best intent macro precision (FR-02 target 0.85), trains it
    on all tickets and saves storage/classifier.joblib.
 7. Writes evaluation/results/classifier_comparison_<stamp>.md/.json and
    classifier_oof_<stamp>.csv (out-of-fold predictions, used to set the routing
    confidence threshold in B-07).
The validation set is not used here.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from src.classify import apply_temperature
from src.config import get_settings
from src.embeddings import get_embedder
from src.ingest import load_tickets

logger = logging.getLogger("train_classifier")
SEED = 42
MODELS = ("logreg_emb", "knn_emb", "tfidf_lr")


def make_model(name: str):
    if name == "logreg_emb":
        return LogisticRegression(C=10.0, max_iter=3000, random_state=SEED)
    if name == "knn_emb":
        return KNeighborsClassifier(n_neighbors=10, weights="distance", metric="cosine")
    if name == "tfidf_lr":
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                             LogisticRegression(C=10.0, max_iter=3000, random_state=SEED))
    raise ValueError(name)


def uses_embeddings(name: str) -> bool:
    return name.endswith("_emb")


def text_groups(texts: List[str]) -> np.ndarray:
    """Same group for tickets whose wording is identical after normalisation."""
    norm = [re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", t.lower())).strip() for t in texts]
    ids: Dict[str, int] = {}
    return np.array([ids.setdefault(n, len(ids)) for n in norm])


def oof_predict(name: str, X, y: np.ndarray, groups: np.ndarray, classes: List[str],
                n_splits: int) -> np.ndarray:
    """Out-of-fold probabilities, columns aligned to `classes`."""
    proba = np.zeros((len(y), len(classes)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    col = {c: i for i, c in enumerate(classes)}
    for train_idx, test_idx in cv.split(np.zeros(len(y)), y, groups):
        model = make_model(name)
        Xtr = X[train_idx] if not isinstance(X, list) else [X[i] for i in train_idx]
        Xte = X[test_idx] if not isinstance(X, list) else [X[i] for i in test_idx]
        model.fit(Xtr, y[train_idx])
        p = model.predict_proba(Xte)
        for j, c in enumerate(model.classes_):
            proba[test_idx, col[c]] = p[:, j]
    return proba


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10):
    """Expected calibration error (0-1) and a reliability table."""
    edges = np.linspace(0, 1, bins + 1)
    table, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc, avg = correct[mask].mean(), conf[mask].mean()
        total += mask.mean() * abs(acc - avg)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                      "mean_confidence": round(float(avg), 3), "accuracy": round(float(acc), 3)})
    return float(total), table


def fit_temperature(proba: np.ndarray, y_idx: np.ndarray) -> float:
    """Temperature minimising negative log-likelihood on out-of-fold predictions."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.round(np.arange(0.1, 6.01, 0.05), 2):
        p = apply_temperature(proba, t)
        nll = -np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, 1)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def evaluate(proba: np.ndarray, y: np.ndarray, classes: List[str]) -> Dict:
    y_idx = np.array([classes.index(v) for v in y])
    pred_idx = proba.argmax(axis=1)
    pred = np.array([classes[i] for i in pred_idx])
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    top3 = np.mean([y_idx[i] in np.argsort(-proba[i])[:3] for i in range(len(y))])
    conf = proba.max(axis=1)
    raw_ece, _ = ece(conf, pred == y)
    temp = fit_temperature(proba, y_idx)
    cal = apply_temperature(proba, temp)
    cal_ece, table = ece(cal.max(axis=1), pred == y)
    confusions = Counter((t, q) for t, q in zip(y, pred) if t != q).most_common(5)
    return {
        "accuracy": round(float((pred == y).mean()), 3),
        "macro_precision": round(float(p.mean()), 3),
        "macro_recall": round(float(r.mean()), 3),
        "macro_f1": round(float(f.mean()), 3),
        "top3_accuracy": round(float(top3), 3),
        "ece_raw": round(raw_ece, 3), "temperature": temp, "ece_calibrated": round(cal_ece, 3),
        "reliability": table,
        "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                          "support": int(s[i])} for i, c in enumerate(classes)},
        "top_confusions": [{"true": a, "predicted": b, "count": n} for (a, b), n in confusions],
        "_proba_calibrated": cal, "_pred": pred,
    }


def train(tickets_path: str, model_path: str, out_dir: str, embedder=None,
          n_splits: int = 5, force_model: Optional[str] = None) -> Dict:
    s = get_settings()
    ingest = load_tickets(tickets_path)
    rows = [(t, ingest.answer_key.get(t.ticket_id, {}).get("labels") or {}) for t in ingest.tickets]
    rows = [(t, lab) for t, lab in rows if lab.get("intent") and lab.get("urgency") and t.text.strip()]
    if not rows:
        raise ValueError(f"No labelled tickets in {tickets_path}")
    texts = [t.text for t, _ in rows]
    y = {"intent": np.array([lab["intent"] for _, lab in rows]),
         "urgency": np.array([lab["urgency"] for _, lab in rows])}
    classes = {k: sorted(set(v)) for k, v in y.items()}
    groups = text_groups(texts)
    embedder = embedder or get_embedder(s.embedding_model)
    logger.info("Embedding %d tickets with %s", len(texts), getattr(embedder, "name", "?"))
    X_emb = np.asarray(embedder.embed(texts), dtype=float)

    results: Dict[str, Dict] = {}
    for name in MODELS:
        X = X_emb if uses_embeddings(name) else texts
        results[name] = {tgt: evaluate(oof_predict(name, X, y[tgt], groups, classes[tgt], n_splits),
                                       y[tgt], classes[tgt]) for tgt in ("intent", "urgency")}
        r = results[name]["intent"]
        logger.info("%-10s intent precision %.3f recall %.3f | urgency acc %.3f",
                    name, r["macro_precision"], r["macro_recall"], results[name]["urgency"]["accuracy"])

    def rank(m):
        return results[m]["intent"]["macro_precision"], results[m]["intent"]["macro_f1"]

    chosen = force_model or max(MODELS, key=rank)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    version = f"{chosen}-{stamp}"

    # final model on all tickets
    X = X_emb if uses_embeddings(chosen) else texts
    bundle = {"version": version, "feature": "embedding" if uses_embeddings(chosen) else "text",
              "embedding_model": getattr(embedder, "name", s.embedding_model),
              "trained_on": tickets_path, "n_tickets": len(texts), "trained_at": stamp}
    for tgt in ("intent", "urgency"):
        model = make_model(chosen).fit(X, y[tgt])
        bundle[tgt] = {"model": model, "temperature": results[chosen][tgt]["temperature"]}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    # out-of-fold predictions of the chosen model, for the routing threshold (B-07)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ri, ru = results[chosen]["intent"], results[chosen]["urgency"]
    with (out / f"classifier_oof_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticket_id", "true_intent", "pred_intent", "intent_confidence", "intent_correct",
                    "true_urgency", "pred_urgency", "urgency_confidence",
                    "expected_route", "must_not_auto_respond", "answerable_from_docs"])
        for i, (t, lab) in enumerate(rows):
            w.writerow([t.ticket_id, y["intent"][i], ri["_pred"][i],
                        round(float(ri["_proba_calibrated"][i].max()), 4), int(ri["_pred"][i] == y["intent"][i]),
                        y["urgency"][i], ru["_pred"][i], round(float(ru["_proba_calibrated"][i].max()), 4),
                        lab.get("expected_route"), lab.get("must_not_auto_respond"),
                        lab.get("answerable_from_docs")])

    clean = {m: {t: {k: v for k, v in r.items() if not k.startswith("_")} for t, r in res.items()}
             for m, res in results.items()}
    meta = {"run_at": stamp, "tickets": tickets_path, "n_tickets": len(texts),
            "unique_texts": int(len(set(groups))), "folds": n_splits, "chosen": chosen,
            "embedding_model": bundle["embedding_model"], "model_path": model_path}
    (out / f"classifier_comparison_{stamp}.json").write_text(
        json.dumps({"meta": meta, "results": clean}, indent=2), encoding="utf-8")
    (out / f"classifier_comparison_{stamp}.md").write_text(to_markdown(meta, clean), encoding="utf-8")
    return {"meta": meta, "results": clean, "report": str(out / f"classifier_comparison_{stamp}.md")}


def to_markdown(meta: Dict, res: Dict) -> str:
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    L = [f"# Classifier comparison ({meta['run_at']})", "",
         f"Development tickets: {meta['n_tickets']} ({meta['unique_texts']} unique texts), "
         f"{meta['folds']}-fold grouped cross-validation. Embeddings: {meta['embedding_model']}. "
         f"Chosen: **{meta['chosen']}**.", ""]
    for tgt in ("intent", "urgency"):
        L += [f"## {tgt.capitalize()}", "",
              "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Top-3 | ECE raw | "
              "Temperature | ECE calibrated |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, r in res.items():
            x = r[tgt]
            L.append(f"| {m} | {pct(x['accuracy'])} | {pct(x['macro_precision'])} | {pct(x['macro_recall'])} | "
                     f"{pct(x['macro_f1'])} | {pct(x['top3_accuracy'])} | {pct(x['ece_raw'])} | "
                     f"{x['temperature']} | {pct(x['ece_calibrated'])} |")
        L.append("")
    c = res[meta["chosen"]]["intent"]
    L += [f"## Chosen model ({meta['chosen']}): intent per class", "",
          "| Intent | Precision | Recall | Support |", "|---|---|---|---|"]
    L += [f"| {k} | {pct(v['precision'])} | {pct(v['recall'])} | {v['support']} |"
          for k, v in sorted(c["per_class"].items(), key=lambda kv: kv[1]["precision"])]
    L += ["", "Most confused (true -> predicted):", ""]
    L += [f"- {x['true']} -> {x['predicted']}: {x['count']}" for x in c["top_confusions"]]
    L += ["", f"## Calibration ({meta['chosen']}, intent, after temperature)", "",
          "Temperature is fitted on the same out-of-fold predictions it is measured on, so the "
          "calibrated ECE is slightly optimistic.", "",
          "| Confidence bin | Tickets | Mean confidence | Accuracy |", "|---|---|---|---|"]
    L += [f"| {b['bin']} | {b['n']} | {b['mean_confidence']} | {b['accuracy']} |" for b in c["reliability"]]
    return "\n".join(L) + "\n"


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickets", default="data/development_tickets.json")
    p.add_argument("--model-path", default=s.classifier_path)
    p.add_argument("--output", default="evaluation/results")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--model", choices=MODELS, help="force this model instead of the best one")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    summary = train(args.tickets, args.model_path, args.output, n_splits=args.folds,
                    force_model=args.model)
    res = summary["results"]
    print(f"\n{'INTENT':10} precision  recall   F1     top-3   ECE raw -> calibrated")
    for m, r in res.items():
        x = r["intent"]
        print(f"{m:10} {x['macro_precision']:.3f}     {x['macro_recall']:.3f}    {x['macro_f1']:.3f}  "
              f"{x['top3_accuracy']:.3f}   {x['ece_raw']:.3f} -> {x['ece_calibrated']:.3f}")
    print(f"\n{'URGENCY':10} accuracy   macro P  macro R")
    for m, r in res.items():
        x = r["urgency"]
        print(f"{m:10} {x['accuracy']:.3f}      {x['macro_precision']:.3f}    {x['macro_recall']:.3f}")
    print(f"\nChosen: {summary['meta']['chosen']}  ->  saved {summary['meta']['model_path']}")
    print(f"Report: {summary['report']}")


if __name__ == "__main__":
    main()
