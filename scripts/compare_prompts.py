"""Compare PR-02 versions on how well they separate answerable from not-answerable tickets.

    python -m scripts.compare_prompts --input data/development_tickets.json --versions 1.1
    python -m scripts.compare_prompts --input data/development_tickets.json --versions 1.0,1.1 --per-group 30

Tune on the development set only; the validation set is for the final check.

Only tickets that routing clears go to drafting, so only those are sampled. Identical texts are
drafted once. The sample is: up to --per-group tickets labelled NOT answerable (the gap) and the
same number labelled answerable, chosen with a fixed seed so every version sees the same tickets.
Replies are cached, so a re-run costs no tokens. The table is printed and saved to
evaluation/results/prompt_comparison_<time>.md, with every draft in the matching .jsonl.

Two numbers matter:
  caught           not-answerable tickets the prompt refused   (higher is better)
  wrongly refused  answerable tickets the prompt refused       (lower is better; each one is a
                   customer who could have had an instant answer)
Note: on this data many ticket texts carry contradictory labels (the same text is labelled
answerable once and not answerable once), so neither number can reach its ideal.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.classify import Classifier
from src.generate import Generator
from src.ingest import load_tickets
from src.retrieve import Retriever
from src.route import Router


def cleared_sample(input_path: str, per_group: int, seed: int = 7) -> List[dict]:
    ingest = load_tickets(input_path)
    classifier, retriever, router = Classifier(), Retriever(), Router()
    cls = {c.ticket_id: c for c in classifier.classify_many(ingest.tickets)}
    seen, groups = set(), {True: [], False: []}
    for t in ingest.tickets:
        labels = (ingest.answer_key.get(t.ticket_id) or {}).get("labels") or {}
        ans = labels.get("answerable_from_docs")
        c = cls[t.ticket_id]
        if ans is None or c.fallback_reason or t.text in seen:
            continue
        r = retriever.retrieve(t.text)
        if router.route(t, c, r).action != "answer":
            continue
        seen.add(t.text)
        groups[bool(ans)].append({"ticket": t, "retrieval": r, "answerable": bool(ans)})
    rnd = random.Random(seed)
    out = []
    for g in (False, True):
        rnd.shuffle(groups[g])
        out += groups[g][:per_group]
    return out


def evaluate(sample: List[dict], version: str) -> Dict[str, object]:
    gen = Generator()
    gen.prompt_version = version
    res = {"version": version, "not_answerable": 0, "caught": 0, "answerable": 0, "wrongly_refused": 0,
           "errors": 0, "tokens": 0, "refusal_reasons": {}}
    examples, per_ticket = [], []
    for item in sample:
        d = gen.draft_answer(item["ticket"], item["retrieval"].articles)
        per_ticket.append({"version": version, "ticket_id": item["ticket"].ticket_id,
                           "labelled_answerable": item["answerable"], "model_answerable": d.answerable,
                           "uncertainty": d.uncertainty, "error": d.error, "passages": d.passage_doc_ids,
                           "answer": d.answer, "ticket_text": item["ticket"].text})
        res["tokens"] += d.tokens
        if d.error:
            res["errors"] += 1
            continue
        refused = not d.answerable
        if item["answerable"]:
            res["answerable"] += 1
            res["wrongly_refused"] += refused
            if refused and len(examples) < 5:
                examples.append((item["ticket"].ticket_id, d.uncertainty, item["ticket"].text[:140]))
        else:
            res["not_answerable"] += 1
            res["caught"] += refused
        if refused:
            key = (d.uncertainty or "?").strip()[:1].lower()
            res["refusal_reasons"][key] = res["refusal_reasons"].get(key, 0) + 1
    res["wrongly_refused_examples"] = examples
    res["per_ticket"] = per_ticket
    llm = gen._llm
    res["llm"] = {k: getattr(llm, k, None) for k in ("name", "calls", "cache_hits", "failures")} if llm else {}
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/development_tickets.json")
    p.add_argument("--versions", default="1.1", help="comma-separated PR-02 versions, e.g. 1.0,1.1")
    p.add_argument("--per-group", type=int, default=30, help="tickets per label group (default 30)")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    sample = cleared_sample(args.input, args.per_group)
    n_not = sum(not s["answerable"] for s in sample)
    print(f"Sample: {len(sample)} unique cleared tickets ({n_not} not answerable, {len(sample) - n_not} answerable)")
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"  # noqa: E731
    per_ticket = []
    lines = [f"# PR-02 comparison - {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Input `{args.input}`, {len(sample)} unique tickets cleared by routing "
             f"({n_not} labelled not answerable, {len(sample) - n_not} answerable).", "",
             "| Version | Caught (not answerable refused) | Wrongly refused (answerable) | Errors | Tokens | "
             "Refusals by test |", "|---|---|---|---|---|---|"]
    for v in [x.strip() for x in args.versions.split(",") if x.strip()]:
        r = evaluate(sample, v)
        per_ticket += r["per_ticket"]
        lines.append(f"| {v} | {r['caught']}/{r['not_answerable']} ({pct(r['caught'], r['not_answerable'])}) | "
                     f"{r['wrongly_refused']}/{r['answerable']} ({pct(r['wrongly_refused'], r['answerable'])}) | "
                     f"{r['errors']} | {r['tokens']} | {r['refusal_reasons']} |")
        print(lines[-1])
        for tid, why, text in r["wrongly_refused_examples"]:
            print(f"    wrongly refused {tid}: {why!r} <- {text!r}")
    out = Path("evaluation/results") / f"prompt_comparison_{datetime.now():%Y%m%d_%H%M}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    detail = out.with_suffix(".jsonl")
    detail.write_text("".join(json.dumps(x) + "\n" for x in per_ticket), encoding="utf-8")
    print(f"Saved {out} and {detail} (one line per ticket: the draft and the model's reason)")


if __name__ == "__main__":
    main()
