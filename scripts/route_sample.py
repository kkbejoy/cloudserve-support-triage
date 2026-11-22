"""Build Specification day-3 checkpoint: run N tickets through classify -> retrieve -> route,
write every decision to the log, then print the log so you can read each reason.

    python -m scripts.route_sample --tickets data/validation_tickets.json --limit 20
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter

from src.classify import Classifier
from src.config import get_settings
from src.ingest import load_tickets
from src.logging_store import DecisionLog
from src.pipeline import Pipeline
from src.retrieve import Retriever
from src.route import Router


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickets", default="data/validation_tickets.json")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    s = get_settings()

    ingest = load_tickets(args.tickets)
    tickets = ingest.tickets[: args.limit]
    classifier, retriever, router = Classifier(), Retriever(), Router()
    log = DecisionLog()
    run_id = log.start_run(args.tickets, len(tickets), {
        "confidence_threshold": router.threshold, "relevance_threshold": retriever.threshold,
        "classifier_floor": classifier.floor, "classifier": classifier.version,
        "chunk_strategy": retriever.strategy, "auto_respond_enabled": router.enabled,
        "limit": args.limit})
    outcomes = Pipeline(classifier, retriever, router, log).process(run_id, tickets)
    log.finish_run(run_id)

    for o in outcomes:
        c = o.classification
        intent = f"{c.intent} {c.intent_confidence:.2f}" if c else "-"
        print(f"{o.ticket_id:10} {o.routing.action:8} {o.routing.rule:28} {intent:32} {o.routing.reason}")
    print("\nDecisions by rule:", dict(Counter(o.routing.rule for o in outcomes)))
    rec = log.reconcile(run_id, [t.ticket_id for t in tickets])
    print(f"Log reconciliation: {rec['tickets_with_final_decision']}/{rec['tickets_in_file']} tickets "
          f"have a routing decision -> {'OK' if rec['reconciled'] else 'MISMATCH ' + str(rec['missing'])}; "
          f"records by stage {rec['records_by_stage']}")
    print(f"Run id: {run_id}   log: {s.database_path}")
    log.close()


if __name__ == "__main__":
    main()
