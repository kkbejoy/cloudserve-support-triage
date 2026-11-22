"""Fairness audit of one harness run (B-14, NFR-06: first-contact resolution should differ by less
than 5 percentage points between customer groups).

    python -m scripts.fairness_audit --run evaluation/results/run_<id> --tickets data/validation_tickets.json

Writes fairness.md into the run folder and prints it. Per group (tier, region, language fluency,
channel) it shows:
  - FCR (answered automatically / tickets) next to the historical FCR of the same tickets
  - answerable-labelled tickets that were answered: a fairer comparison, because groups differ in
    how many of their tickets must go to a person anyway
  - answers sent although labelled escalate, and drafts blocked
Groups with fewer than 20 tickets are marked: their rates move a lot with one ticket.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

GROUPS = ("customer_tier", "customer_region", "language_fluency", "channel")
SMALL = 20
LIMIT_PTS = 5.0


def _pct(a: int, b: int) -> float:
    return round(100 * a / b, 1) if b else 0.0


def audit(rows: List[dict], tickets: Dict[str, dict]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for field in GROUPS:
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(field) or "unknown"].append(r)
        table = []
        for name, rs in sorted(groups.items()):
            lab = [tickets.get(r["ticket_id"], {}).get("labels", {}) for r in rs]
            hist = [tickets.get(r["ticket_id"], {}).get("history", {}) or {} for r in rs]
            sent = [r["final_action"] == "auto_respond" for r in rs]
            answerable = [i for i, x in enumerate(lab) if x.get("expected_route") == "auto_respond"]
            table.append({
                "group": name, "tickets": len(rs),
                "fcr": _pct(sum(sent), len(rs)),
                "historical_fcr": _pct(sum(bool(h.get("first_contact_resolution")) for h in hist), len(rs))
                if any(hist) else None,
                "answerable": len(answerable),
                "answerable_answered": _pct(sum(sent[i] for i in answerable), len(answerable)),
                "sent_but_labelled_escalate": sum(1 for i, x in enumerate(lab)
                                                  if sent[i] and x.get("expected_route") == "escalate"),
                "blocked": sum(r["final_action"] == "block" for r in rs),
                "small": len(rs) < SMALL})
        out[field] = table
    return out


def to_markdown(result: Dict[str, list], run_dir: str) -> str:
    lines = [f"# Fairness audit - {Path(run_dir).name}", "",
             f"NFR-06: FCR should differ by less than {LIMIT_PTS:.0f} percentage points between groups. "
             f"Groups under {SMALL} tickets are marked with *.", ""]
    for field, table in result.items():
        big = [t for t in table if not t["small"]]
        use, scope = (big, f"groups of {SMALL}+") if len(big) >= 2 else (table, "all groups, some small")
        fcr = [t["fcr"] for t in use]
        ans = [t["answerable_answered"] for t in use if t["answerable"]] or [0]
        gap, gap_ans = max(fcr) - min(fcr), max(ans) - min(ans)
        lines += [f"## {field}", "",
                  f"Largest FCR gap {gap:.1f} pts ({'within' if gap < LIMIT_PTS else 'OVER'} the {LIMIT_PTS:.0f}-pt "
                  f"limit); among answerable tickets {gap_ans:.1f} pts ({scope}).", "",
                  "| Group | Tickets | FCR | Historical FCR | Answerable answered | Sent but labelled escalate | "
                  "Blocked |", "|---|---|---|---|---|---|---|"]
        for t in table:
            h = f"{t['historical_fcr']}%" if t["historical_fcr"] is not None else "-"
            lines.append(f"| {t['group']}{'*' if t['small'] else ''} | {t['tickets']} | {t['fcr']}% | {h} | "
                         f"{t['answerable_answered']}% of {t['answerable']} | {t['sent_but_labelled_escalate']} | "
                         f"{t['blocked']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a run folder written by the harness")
    p.add_argument("--tickets", required=True, help="the ticket file that run used (for labels and history)")
    args = p.parse_args()
    rows = [json.loads(ln) for ln in (Path(args.run) / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    tickets = {t.get("ticket_id"): t for t in json.loads(Path(args.tickets).read_text(encoding="utf-8"))
               if isinstance(t, dict)}
    md = to_markdown(audit(rows, tickets), args.run)
    (Path(args.run) / "fairness.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
