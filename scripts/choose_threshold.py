"""Choose the routing confidence threshold from data (B-07; Project Brief section 05).

    python -m scripts.choose_threshold                       # uses the newest classifier_oof_*.csv
    python -m scripts.choose_threshold --oof evaluation/results/classifier_oof_<stamp>.csv

Uses the classifier's OUT-OF-FOLD predictions on the development tickets (each ticket predicted
by a model that never saw it). For each candidate threshold it simulates the classifier-based
routing rules (must-escalate intents + confidence threshold) and reports:
  cleared_for_answer      share of tickets passed on to answer drafting
  wrong_intent_cleared    cleared tickets whose predicted intent was wrong
  must_not_cleared        cleared tickets labelled must_not_auto_respond (must be 0)
  cleared_but_expected_escalate  cleared tickets the labels say should go to a person
  routing_agreement       agreement with labels.expected_route
Retrieval and the billing-dispute rule are not simulated here; the harness measures the full chain.
"""
from __future__ import annotations

import argparse
import csv
import glob
from datetime import datetime
from pathlib import Path

from src.config import get_settings

CANDIDATES = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]


def simulate(rows, must_escalate, t):
    cleared = [r for r in rows if r["pred_intent"] not in must_escalate
               and float(r["intent_confidence"]) >= t]
    ids = {r["ticket_id"] for r in cleared}
    agree = sum((("auto_respond" if r["ticket_id"] in ids else "escalate") == r["expected_route"])
                for r in rows)
    return {
        "threshold": t,
        "cleared_for_answer": round(len(cleared) / len(rows), 3),
        "wrong_intent_cleared": sum(r["intent_correct"] == "0" for r in cleared),
        "must_not_cleared": sum(r["must_not_auto_respond"] == "True" for r in cleared),
        "cleared_but_expected_escalate": sum(r["expected_route"] == "escalate" for r in cleared),
        "routing_agreement": round(agree / len(rows), 3),
    }


def recommend(table, max_coverage_loss=0.02):
    """Rule (fixed before looking at the numbers): among safe thresholds (no must-not ticket
    cleared) that lose at most `max_coverage_loss` of the best coverage, take the one with the
    fewest wrong intents cleared; on a tie take the highest threshold (most safety margin)."""
    safe = [r for r in table if r["must_not_cleared"] == 0]
    best_cov = max(r["cleared_for_answer"] for r in safe)
    near = [r for r in safe if r["cleared_for_answer"] >= best_cov - max_coverage_loss - 1e-9]
    fewest = min(r["wrong_intent_cleared"] for r in near)
    return max((r for r in near if r["wrong_intent_cleared"] == fewest), key=lambda r: r["threshold"])


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oof", help="classifier_oof_*.csv (default: newest in evaluation/results)")
    p.add_argument("--output", default="evaluation/results")
    p.add_argument("--max-coverage-loss", type=float, default=0.02,
                   help="coverage (share of tickets) you accept losing to remove wrong intents")
    args = p.parse_args()
    path = args.oof or max(glob.glob("evaluation/results/classifier_oof_*.csv"), default=None)
    if not path:
        raise SystemExit("No classifier_oof_*.csv found; run python -m scripts.train_classifier first")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    table = [simulate(rows, s.must_escalate, t) for t in CANDIDATES]
    rec = recommend(table, args.max_coverage_loss)

    head = ("| Threshold | Cleared for answer | Wrong intent cleared | Must-not cleared | "
            "Cleared but expected escalate | Agreement with expected route |")
    lines = [f"# Routing confidence threshold ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Out-of-fold predictions: `{path}` ({len(rows)} development tickets). "
             f"Must-escalate intents: {', '.join(sorted(s.must_escalate))}.", "",
             head, "|---|---|---|---|---|---|"]
    print(f"{'thr':>5} {'cleared':>8} {'wrongInt':>8} {'mustNot':>7} {'expEsc':>6} {'agree':>6}")
    for r in table:
        lines.append(f"| {r['threshold']:.2f} | {100 * r['cleared_for_answer']:.1f}% | "
                     f"{r['wrong_intent_cleared']} | {r['must_not_cleared']} | "
                     f"{r['cleared_but_expected_escalate']} | {100 * r['routing_agreement']:.1f}% |")
        mark = "  <- recommended" if r is rec else ""
        print(f"{r['threshold']:5.2f} {100 * r['cleared_for_answer']:7.1f}% {r['wrong_intent_cleared']:8} "
              f"{r['must_not_cleared']:7} {r['cleared_but_expected_escalate']:6} "
              f"{100 * r['routing_agreement']:5.1f}%{mark}")
    lines += ["", f"Recommended: **{rec['threshold']:.2f}**. Rule: no must-not-auto-respond ticket cleared; "
              f"give up at most {100 * args.max_coverage_loss:.0f} points of coverage to clear fewer wrong "
              "intents; on a tie take the highest threshold (most safety margin at no measured cost)."]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"threshold_choice_{datetime.now():%Y%m%d_%H%M}.md"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nRecommended CONFIDENCE_THRESHOLD = {rec['threshold']:.2f}   (report: {f})")


if __name__ == "__main__":
    main()
