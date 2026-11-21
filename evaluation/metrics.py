"""Metrics for an evaluation run (FR-11, acceptance criterion A10; Build Specification section 04).

compute() takes the per-ticket rows written by the harness and returns the four groups the
Build Specification requires - volume, business, technical, governance - plus breakdowns by
channel and customer group for the fairness audit. Labels are optional: on a file without them
(e.g. the hidden set if it had none) the technical accuracy figures are reported as None.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

GROUP_FIELDS = ("channel", "customer_tier", "customer_region", "language_fluency")
# "cleared" = passed routing and (if generation ran) the model judged it answerable; still not sent
CLEARED = ("pending_answer", "drafted", "auto_respond")


def _validation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Guardrails (B-09): what they blocked, and the safety numbers that matter most."""
    val = [r for r in rows if r.get("validation_ran")]
    if not val:
        return {"ran": False}
    sent = [r for r in rows if r["final_action"] == "auto_respond"]
    ans = [r for r in val if r.get("answerable_from_docs") is True]
    not_ans = [r for r in val if r.get("answerable_from_docs") is False]
    return {
        "ran": True,
        "drafts_checked": len(val),
        "passed": sum(1 for r in val if r.get("validation_passed")),
        "blocked_by_check": dict(Counter(g for r in val for g in r.get("guardrails_fired", []))),
        "fact_check_tokens": sum(r.get("grounding_tokens") or 0 for r in val),
        "sent_with_citations_added_by_code": sum(1 for r in sent if r.get("citations_added")),
        # safety: answers that went out although the label says a person should have handled it
        "sent_but_labelled_escalate": sum(1 for r in sent if r.get("expected_route") == "escalate"),
        "sent_but_labelled_not_answerable": sum(1 for r in sent if r.get("answerable_from_docs") is False),
        # how much of the gap the guardrails closed, and what it cost in good answers
        "not_answerable_checked": len(not_ans),
        "not_answerable_blocked": sum(1 for r in not_ans if not r.get("validation_passed")),
        "answerable_checked": len(ans),
        "answerable_blocked": sum(1 for r in ans if not r.get("validation_passed")),
    }


def _generation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How well answer drafting closes the answerability gap (PR-02's answerable flag)."""
    gen = [r for r in rows if r.get("generation_ran")]
    if not gen:
        return {"ran": False}
    ok = [r for r in gen if r.get("answerable_by_model") is not None]
    unans = [r for r in ok if r.get("answerable_from_docs") is False]
    ans = [r for r in ok if r.get("answerable_from_docs") is True]
    drafted = [r for r in ok if r.get("answerable_by_model")]
    cited_ok = [r for r in drafted if r.get("expected_doc_ids")]
    return {
        "ran": True,
        "tickets_sent_to_generation": len(gen),
        "drafted": len(drafted),
        "judged_not_answerable": sum(1 for r in ok if not r.get("answerable_by_model")),
        "generation_errors": len(gen) - len(ok),
        "served_from_cache": sum(1 for r in gen if r.get("llm_cached")),
        # the 20% gap: cleared by routing but labelled not answerable from the docs
        "unanswerable_cleared": len(unans),
        "unanswerable_caught": sum(1 for r in unans if not r.get("answerable_by_model")),
        "answerable_cleared": len(ans),
        "answerable_refused": sum(1 for r in ans if not r.get("answerable_by_model")),
        "drafted_but_labelled_not_answerable": sum(1 for r in drafted if r.get("answerable_from_docs") is False),
        "citation_matches_expected_doc": _pct(
            sum(bool(set(r.get("cited_doc_ids") or []) & set(r["expected_doc_ids"])) for r in cited_ok),
            len(cited_ok)),
    }


def _pct(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 1)


def _per_class(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labelled = [r for r in rows if r.get("true_intent") and r.get("intent")]
    if not labelled:
        return {"labelled_tickets": 0}
    classes = sorted({r["true_intent"] for r in labelled} | {r["intent"] for r in labelled})
    per = {}
    for c in classes:
        tp = sum(1 for r in labelled if r["intent"] == c and r["true_intent"] == c)
        fp = sum(1 for r in labelled if r["intent"] == c and r["true_intent"] != c)
        fn = sum(1 for r in labelled if r["intent"] != c and r["true_intent"] == c)
        per[c] = {"precision": _pct(tp, tp + fp), "recall": _pct(tp, tp + fn), "support": tp + fn}
    with_support = [p for p in per.values() if p["support"]]
    return {
        "labelled_tickets": len(labelled),
        "accuracy": _pct(sum(r["intent"] == r["true_intent"] for r in labelled), len(labelled)),
        "macro_precision": round(statistics.mean(p["precision"] or 0 for p in with_support), 4),
        "macro_recall": round(statistics.mean(p["recall"] or 0 for p in with_support), 4),
        "per_class": per,
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    acts = Counter(r["final_action"] for r in rows)
    lab = [r for r in rows if r.get("true_intent")]
    ans = [r for r in rows if r.get("answerable_from_docs") and r.get("expected_doc_ids") and r.get("retrieval_ran")]
    return {
        "tickets": n,
        "cleared_rate": _pct(sum(acts[a] for a in CLEARED), n),
        "auto_respond_rate": _pct(acts["auto_respond"], n),
        "escalation_rate": _pct(acts["escalate"] + acts["block"], n),
        "intent_accuracy": _pct(sum(r["intent"] == r["true_intent"] for r in lab), len(lab)),
        "retrieval_hit_rate": _pct(sum(r["retrieval_hit"] for r in ans), len(ans)),
    }


def compute(rows: List[Dict[str, Any]], reconciliation: Dict[str, Any], run_info: Dict[str, Any]) -> Dict[str, Any]:
    n = len(rows)
    acts = Counter(r["final_action"] for r in rows)
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    auto = acts["auto_respond"]
    labelled = [r for r in rows if r.get("expected_route")]
    cleared = [r for r in rows if r["final_action"] in CLEARED]

    # retrieval hit rate: answerable tickets that reached retrieval and got an expected article back
    ans = [r for r in rows if r.get("answerable_from_docs") and r.get("expected_doc_ids") and r.get("retrieval_ran")]
    urg = [r for r in rows if r.get("true_urgency") and r.get("urgency")]

    history = [r for r in rows if r.get("hist_escalated") is not None]
    hist_fcr = _pct(sum(bool(r.get("hist_first_contact_resolution")) for r in history), len(history))
    hist_esc = _pct(sum(bool(r.get("hist_escalated")) for r in history), len(history))

    metrics = {
        "run": run_info,
        "volume": {
            "tickets_processed": n,
            "answered_automatically": auto,
            "drafted_awaiting_validation": acts["drafted"],
            "pending_answer_step": acts["pending_answer"],
            "escalated": acts["escalate"],
            "escalated_not_answerable": sum(1 for r in rows if r.get("rule") == "not_answerable"),
            "blocked_by_guardrails": acts["block"],
            "provider_unavailable": sum(1 for r in rows if r.get("provider_unavailable")),
            "processing_errors": sum(1 for r in rows if r.get("error")),
            "degraded_run": sum(1 for r in rows if r.get("provider_unavailable")) > 0.05 * n if n else False,
        },
        "business": {
            "first_contact_resolution": _pct(auto, n),
            "first_contact_resolution_note": (
                "Counts only answers actually sent (passed every guardrail). Blocked drafts, drafts without "
                "guardrails and tickets pending the answer step are not resolved."),
            "cleared_for_answer_rate": _pct(len(cleared), n),
            "escalation_rate": _pct(acts["escalate"] + acts["block"], n),
            "response_time_ms": {"mean": round(statistics.mean(lat), 1) if lat else None,
                                 "median": _percentile(lat, 0.5)},
            "response_time_note": ("System processing time per ticket. First reply to the customer = this time "
                                   "for automatic answers; escalated tickets need an acknowledgement (FR-12)."),
            "baseline_from_history": {"first_contact_resolution": hist_fcr, "escalation_rate": hist_esc,
                                      "tickets_with_history": len(history)},
        },
        "technical": {
            "intent_classification": _per_class(rows),
            "urgency_accuracy": _pct(sum(r["urgency"] == r["true_urgency"] for r in urg), len(urg)),
            "retrieval_hit_rate": _pct(sum(r["retrieval_hit"] for r in ans), len(ans)),
            "retrieval_hit_rate_tickets": len(ans),
            "routing_agreement_with_expected_route": _pct(
                sum((r["final_action"] in CLEARED) ==
                    (r["expected_route"] == "auto_respond") for r in labelled), len(labelled)),
            "cleared_but_expected_escalate": sum(1 for r in cleared if r.get("expected_route") == "escalate"),
            "generation": _generation(rows),
            "validation": _validation(rows),
            "latency_ms": {"median": _percentile(lat, 0.5), "p95": _percentile(lat, 0.95),
                           "max": round(max(lat), 1) if lat else None},
        },
        "governance": {
            "decisions_logged_by_stage": reconciliation.get("records_by_stage", {}),
            "tickets_with_logged_decision": reconciliation.get("tickets_with_final_decision"),
            "log_reconciled": reconciliation.get("reconciled"),
            "missing_from_log": reconciliation.get("missing", []),
            "must_not_auto_respond_cleared": sum(1 for r in cleared if r.get("must_not_auto_respond")),
            "must_not_auto_respond_answered": sum(1 for r in rows if r.get("must_not_auto_respond")
                                                  and r["final_action"] == "auto_respond"),
            "guardrail_activations": dict(Counter(g for r in rows for g in r.get("guardrails_fired", []))),
            "private_data_detections": sum(1 for r in rows if "private_data" in r.get("guardrails_fired", [])),
            "escalations_by_rule": dict(Counter(r["rule"] for r in rows if r["final_action"] == "escalate")),
            "kill_switch_auto_respond_enabled": run_info.get("config", {}).get("auto_respond_enabled"),
        },
        "by_group": {},
    }
    for field in GROUP_FIELDS:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[str(r.get(field) or "unknown")].append(r)
        metrics["by_group"][field] = {g: _summary(rs) for g, rs in sorted(groups.items())}
    return metrics


def _gen_md(gen: Dict[str, Any], p) -> List[str]:
    if not gen.get("ran"):
        return ["| Answer drafting | not run |"]
    return [f"| Sent to answer drafting / drafted / judged not answerable / errors | "
            f"{gen['tickets_sent_to_generation']} / {gen['drafted']} / {gen['judged_not_answerable']} / "
            f"{gen['generation_errors']} |",
            f"| Gap caught: labelled not-answerable that drafting refused | "
            f"{gen['unanswerable_caught']} of {gen['unanswerable_cleared']} |",
            f"| Answerable tickets wrongly refused by drafting | {gen['answerable_refused']} of "
            f"{gen['answerable_cleared']} |",
            f"| Drafted but labelled not answerable (for B-09 grounding check) | "
            f"{gen['drafted_but_labelled_not_answerable']} |",
            f"| Draft cites an expected article | {p(gen['citation_matches_expected_doc'])} |"]


def _val_md(val: Dict[str, Any]) -> List[str]:
    if not val.get("ran"):
        return ["| Guardrails | not run |"]
    return [f"| Drafts checked by guardrails / passed (sent) | {val['drafts_checked']} / {val['passed']} |",
            f"| Blocked, by check | {val['blocked_by_check'] or 'none'} |",
            f"| **Sent but labelled escalate** (safety) | {val['sent_but_labelled_escalate']} |",
            f"| Sent but labelled not answerable | {val['sent_but_labelled_not_answerable']} |",
            f"| Not-answerable drafts blocked | {val['not_answerable_blocked']} of {val['not_answerable_checked']} |",
            f"| Answerable drafts blocked (lost automatic answers) | {val['answerable_blocked']} of "
            f"{val['answerable_checked']} |",
            f"| Sent answers where code added citations | {val['sent_with_citations_added_by_code']} |",
            f"| Fact-check tokens | {val['fact_check_tokens']} |"]


def to_markdown(m: Dict[str, Any]) -> str:
    def p(v):
        return "-" if v is None else f"{100 * v:.1f}%"

    v, b, t, g = m["volume"], m["business"], m["technical"], m["governance"]
    ic = t["intent_classification"]
    L = [f"# Evaluation run {m['run']['run_id']}", "",
         f"Input: `{m['run']['input_file']}` - {v['tickets_processed']} tickets - "
         f"{m['run']['duration_s']} s - stages built: {', '.join(m['run']['stages_built'])}", "",
         "## Volume", "", "| Measure | Value |", "|---|---|"]
    L += [f"| {k.replace('_', ' ').capitalize()} | {val} |" for k, val in v.items()]
    L += ["", "## Business", "", "| Measure | This run | Baseline (history) |", "|---|---|---|",
          f"| First contact resolution | {p(b['first_contact_resolution'])} | "
          f"{p(b['baseline_from_history']['first_contact_resolution'])} |",
          f"| Escalation rate | {p(b['escalation_rate'])} | {p(b['baseline_from_history']['escalation_rate'])} |",
          f"| Cleared for answer | {p(b['cleared_for_answer_rate'])} | - |",
          f"| Response time, mean / median (ms) | {b['response_time_ms']['mean']} / "
          f"{b['response_time_ms']['median']} | hours (Project Brief: 8-12 h) |", "",
          f"_{b['first_contact_resolution_note']} {b['response_time_note']}_", "",
          "## Technical", "", "| Measure | Value |", "|---|---|",
          f"| Intent accuracy / macro precision / macro recall | {p(ic.get('accuracy'))} / "
          f"{p(ic.get('macro_precision'))} / {p(ic.get('macro_recall'))} |",
          f"| Urgency accuracy | {p(t['urgency_accuracy'])} |",
          f"| Retrieval hit rate (answerable tickets, n={t['retrieval_hit_rate_tickets']}) | "
          f"{p(t['retrieval_hit_rate'])} |",
          f"| Routing agreement with expected route | {p(t['routing_agreement_with_expected_route'])} |",
          f"| Cleared but labelled 'escalate' | {t['cleared_but_expected_escalate']} |",
          *_gen_md(t.get("generation") or {}, p),
          *_val_md(t.get("validation") or {}),
          f"| Latency median / p95 / max (ms) | {t['latency_ms']['median']} / {t['latency_ms']['p95']} / "
          f"{t['latency_ms']['max']} |", "",
          "## Governance", "", "| Measure | Value |", "|---|---|",
          f"| Decisions logged by stage | {g['decisions_logged_by_stage']} |",
          f"| Tickets with a logged decision / reconciled | {g['tickets_with_logged_decision']} / "
          f"{g['log_reconciled']} |",
          f"| Must-not-auto-respond tickets cleared / answered | {g['must_not_auto_respond_cleared']} / "
          f"{g['must_not_auto_respond_answered']} |",
          f"| Guardrail activations | {g['guardrail_activations'] or 'none (guardrails not built yet)'} |",
          f"| Private data detections | {g['private_data_detections']} |",
          f"| Escalations by rule | {g['escalations_by_rule']} |",
          f"| Automatic answers enabled (kill switch) | {g['kill_switch_auto_respond_enabled']} |", ""]
    for field, groups in m["by_group"].items():
        L += [f"## By {field.replace('_', ' ')}", "",
              "| Group | Tickets | Cleared | Escalated | Intent accuracy | Retrieval hit rate |",
              "|---|---|---|---|---|---|"]
        L += [f"| {k} | {s['tickets']} | {p(s['cleared_rate'])} | {p(s['escalation_rate'])} | "
              f"{p(s['intent_accuracy'])} | {p(s['retrieval_hit_rate'])} |" for k, s in groups.items()]
        L.append("")
    if ic.get("per_class"):
        L += ["## Intent per class", "", "| Intent | Precision | Recall | Support |", "|---|---|---|---|"]
        L += [f"| {c} | {p(x['precision'])} | {p(x['recall'])} | {x['support']} |"
              for c, x in sorted(ic["per_class"].items())]
    return "\n".join(L) + "\n"
