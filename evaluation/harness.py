"""Evaluation harness: one command, any ticket file, unattended (B-05; FR-11, A9, A10).

    python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/

For every ticket in --input: ingest -> classify -> retrieve -> route (-> generate -> validate once
built), every decision written to the decision log. No ticket is skipped: a ticket that fails is
escalated and logged. At the end it writes, in a new folder <output>/run_<run_id>/:
  responses.jsonl  one line per ticket (action, rule, reason, intent, sources, latency)
  metrics.json     volume, business, technical and governance figures, by channel and group
  metrics.md       the same as a readable report
Works on files with or without labels; the input path is never hardcoded (A9, hidden set).
Exit code 0 when the run completes (even if some tickets were escalated because of errors);
2 when it cannot start (unreadable input, index or classifier not built).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.metrics import compute, to_markdown
from src.classify import Classifier
from src.ingest import load_tickets
from src.logging_store import DecisionLog
from src.pipeline import Pipeline, TicketOutcome
from src.retrieve import Retriever
from src.route import Router

logger = logging.getLogger("harness")
STAGES_BUILT = ["ingest", "classify", "retrieve", "route", "log"]


def _row(t, outcome: TicketOutcome, labels: Dict[str, Any], history: Dict[str, Any],
         latency_ms: float) -> Dict[str, Any]:
    c, r = outcome.classification, outcome.retrieval
    expected = labels.get("expected_doc_ids") or []
    returned = r.doc_ids if r else []
    return {
        "ticket_id": t.ticket_id, "channel": t.channel, "customer_tier": t.customer_tier,
        "customer_region": t.customer_region, "language_fluency": t.language_fluency,
        "received_at": t.received_at, "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_action": outcome.final_action, "rule": outcome.routing.rule, "reason": outcome.routing.reason,
        "intent": c.intent if c else None, "intent_confidence": c.intent_confidence if c else None,
        "urgency": c.urgency if c else None, "urgency_confidence": c.urgency_confidence if c else None,
        "retrieval_ran": r is not None,
        "sources": [{"doc_id": a.doc_id, "score": a.score, "section": a.matched_section}
                    for a in (r.articles if r else [])],
        "retrieval_hit": bool(set(expected) & set(returned)) if expected else None,
        "answer": None, "guardrails_fired": [], "latency_ms": round(latency_ms, 1), "error": outcome.error,
        # labels and history are used only for scoring, never by the pipeline
        "true_intent": labels.get("intent"), "true_urgency": labels.get("urgency"),
        "expected_route": labels.get("expected_route"), "expected_doc_ids": expected,
        "answerable_from_docs": labels.get("answerable_from_docs"),
        "must_not_auto_respond": labels.get("must_not_auto_respond"),
        "hist_first_contact_resolution": history.get("first_contact_resolution"),
        "hist_escalated": history.get("escalated"),
    }


def run(input_path: str, output_dir: str, classifier: Optional[Classifier] = None,
        retriever: Optional[Retriever] = None, router: Optional[Router] = None,
        log: Optional[DecisionLog] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    ingest = load_tickets(input_path)
    tickets = ingest.tickets[:limit] if limit else ingest.tickets
    classifier = classifier or Classifier()
    retriever = retriever or Retriever()
    router = router or Router()
    log = log or DecisionLog()
    config = {"confidence_threshold": router.threshold, "relevance_threshold": retriever.threshold,
              "classifier_floor": classifier.floor, "classifier_version": classifier.version,
              "chunk_strategy": retriever.strategy, "auto_respond_enabled": router.enabled,
              "must_escalate_intents": sorted(router.must_escalate)}
    run_id = log.start_run(str(input_path), len(tickets), config)
    pipeline = Pipeline(classifier, retriever, router, log)

    out = Path(output_dir) / f"run_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out / "responses.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(tickets, 1):
            t0 = time.perf_counter()
            outcome = pipeline.process(run_id, [t])[0]
            latency = (time.perf_counter() - t0) * 1000
            key = ingest.answer_key.get(t.ticket_id, {})
            row = _row(t, outcome, key.get("labels") or {}, key.get("history") or {}, latency)
            rows.append(row)
            f.write(json.dumps({k: v for k, v in row.items() if not k.startswith(("true_", "hist_"))}) + "\n")
            if i % 20 == 0 or i == len(tickets):
                logger.info("processed %d/%d", i, len(tickets))
    log.finish_run(run_id)
    reconciliation = log.reconcile(run_id, [t.ticket_id for t in tickets])
    run_info = {"run_id": run_id, "input_file": str(input_path), "output_dir": str(out),
                "duration_s": round(time.perf_counter() - start, 1), "stages_built": STAGES_BUILT,
                "problems_in_input": [p.model_dump() for p in ingest.problems], "config": config}
    metrics = compute(rows, reconciliation, run_info)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out / "metrics.md").write_text(to_markdown(metrics), encoding="utf-8")
    return metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="ticket JSON file (development, validation or hidden set)")
    p.add_argument("--output", required=True, help="folder for results; a run_<id> subfolder is created")
    p.add_argument("--limit", type=int, help="process only the first N tickets (for a quick check)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("src.ingest").setLevel(logging.WARNING)
    try:
        m = run(args.input, args.output, limit=args.limit)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:   # setup problems: say what to do
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    v, b, t, g = m["volume"], m["business"], m["technical"], m["governance"]
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    print(f"\nRun {m['run']['run_id']}: {v['tickets_processed']} tickets in {m['run']['duration_s']} s")
    print(f"  answered {v['answered_automatically']} | pending answer step {v['pending_answer_step']} | "
          f"escalated {v['escalated']} | blocked {v['blocked_by_guardrails']} | errors {v['processing_errors']}")
    base = b["baseline_from_history"]["first_contact_resolution"]
    print(f"  FCR {pct(b['first_contact_resolution'])} (baseline {pct(base)}) | "
          f"escalation {pct(b['escalation_rate'])} | cleared {pct(b['cleared_for_answer_rate'])}")
    ic = t["intent_classification"]
    print(f"  intent precision {pct(ic.get('macro_precision'))} | retrieval hit {pct(t['retrieval_hit_rate'])} | "
          f"routing agreement {pct(t['routing_agreement_with_expected_route'])} | "
          f"latency p50/p95 {t['latency_ms']['median']}/{t['latency_ms']['p95']} ms")
    print(f"  log reconciled: {g['log_reconciled']} ({g['tickets_with_logged_decision']}/{v['tickets_processed']})"
          f" | must-not-auto-respond cleared: {g['must_not_auto_respond_cleared']}")
    print(f"  report: {m['run']['output_dir']}/metrics.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
