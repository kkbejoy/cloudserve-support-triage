"""Compare chunking strategies on the DEVELOPMENT tickets (B-03 decision, B-04 threshold).

    python -m scripts.build_index --strategy all
    python -m scripts.compare_chunking --tickets data/development_tickets.json

For each strategy it reports, using the labels in the ticket file:
  - Hit@1 / Hit@3 / Hit@5: share of answerable tickets whose expected article is ranked
    in the top 1 / 3 / 5 articles (labels.expected_doc_ids)
  - MRR: mean reciprocal rank of the first correct article
  - best-score distribution for answerable vs unanswerable tickets (the gap the
    relevance threshold relies on)
  - a threshold table: for each cut-off, the share of answerable tickets that keep their
    correct article and the share of unanswerable tickets that return nothing.
Results are written to evaluation/results/ as Markdown and JSON.
Do not tune on the validation set; this script is meant for development data.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.chunking import STRATEGIES
from src.config import get_settings
from src.embeddings import get_embedder
from src.ingest import load_tickets
from src.retrieve import Retriever

THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(11)]  # 0.20 .. 0.70


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    v = sorted(values)
    return {"min": round(v[0], 3), "p25": round(v[len(v) // 4], 3),
            "median": round(statistics.median(v), 3), "p75": round(v[(3 * len(v)) // 4], 3),
            "max": round(v[-1], 3)}


def evaluate(retriever: Retriever, tickets, answer_key) -> Dict:
    answerable, unanswerable = [], []
    for t in tickets:
        labels = (answer_key.get(t.ticket_id) or {}).get("labels") or {}
        expected = set(labels.get("expected_doc_ids") or [])
        ranked, _ = retriever.ranked_articles(t.text)
        ids = [h["doc_id"] for h in ranked]
        best = ranked[0]["score"] if ranked else 0.0
        if labels.get("answerable_from_docs") and expected:
            rank = next((i + 1 for i, d in enumerate(ids) if d in expected), None)
            correct_score = next((h["score"] for h in ranked if h["doc_id"] in expected), 0.0)
            answerable.append({"rank": rank, "best": best, "correct_score": correct_score})
        elif labels.get("answerable_from_docs") is False:
            unanswerable.append({"best": best})

    n = len(answerable) or 1
    hit = lambda k: sum(1 for a in answerable if a["rank"] and a["rank"] <= k) / n  # noqa: E731
    mrr = sum(1 / a["rank"] for a in answerable if a["rank"]) / n
    table = []
    for th in THRESHOLDS:
        keep = sum(1 for a in answerable if a["correct_score"] >= th) / n
        reject = (sum(1 for u in unanswerable if u["best"] < th) / len(unanswerable)
                  if unanswerable else 0.0)
        table.append({"threshold": th, "answerable_kept": round(keep, 3),
                      "unanswerable_rejected": round(reject, 3)})
    return {
        "answerable_tickets": len(answerable), "unanswerable_tickets": len(unanswerable),
        "hit@1": round(hit(1), 3), "hit@3": round(hit(3), 3), "hit@5": round(hit(5), 3),
        "mrr": round(mrr, 3),
        "best_score_answerable": _summary([a["best"] for a in answerable]),
        "best_score_unanswerable": _summary([u["best"] for u in unanswerable]),
        "threshold_table": table,
    }


def to_markdown(results: Dict, meta: Dict) -> str:
    lines = [f"# Chunking comparison ({meta['run_at']})", "",
             f"Tickets: `{meta['tickets']}` - development data. Model: {meta['model']}.", "",
             "| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR | Median best score (answerable) | "
             "Median best score (unanswerable) |", "|---|---|---|---|---|---|---|"]
    for s, r in results.items():
        lines.append(f"| {s} | {_pct(r['hit@1'])} | {_pct(r['hit@3'])} | {_pct(r['hit@5'])} | "
                     f"{r['mrr']:.3f} | {r['best_score_answerable'].get('median', '-')} | "
                     f"{r['best_score_unanswerable'].get('median', '-')} |")
    for s, r in results.items():
        lines += ["", f"## Threshold table - {s}", "",
                  "| Threshold | Answerable keep correct article | Unanswerable return nothing |",
                  "|---|---|---|"]
        lines += [f"| {row['threshold']:.2f} | {_pct(row['answerable_kept'])} | "
                  f"{_pct(row['unanswerable_rejected'])} |" for row in r["threshold_table"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickets", default="data/development_tickets.json")
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    p.add_argument("--output", default="evaluation/results")
    args = p.parse_args()

    embedder = get_embedder(s.embedding_model)
    ingest = load_tickets(args.tickets)
    results = {}
    for strategy in args.strategies:
        retriever = Retriever(embedder=embedder, strategy=strategy, threshold=0.0, top_k=20)
        results[strategy] = evaluate(retriever, ingest.tickets, ingest.answer_key)
        r = results[strategy]
        print(f"{strategy:8} Hit@1 {_pct(r['hit@1']):>6}  Hit@3 {_pct(r['hit@3']):>6}  "
              f"Hit@5 {_pct(r['hit@5']):>6}  MRR {r['mrr']:.3f}  "
              f"median best: answerable {r['best_score_answerable'].get('median')} / "
              f"unanswerable {r['best_score_unanswerable'].get('median')}")

    meta = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "tickets": args.tickets,
            "model": getattr(embedder, "name", "unknown")}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (out / f"chunking_comparison_{stamp}.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    (out / f"chunking_comparison_{stamp}.md").write_text(to_markdown(results, meta), encoding="utf-8")
    print(f"\nFull results: {out / f'chunking_comparison_{stamp}.md'}")


if __name__ == "__main__":
    main()
