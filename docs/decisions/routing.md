# Decision: routing rules, confidence threshold and decision log (B-07, B-10)

Status: **decided** - 21 Nov 2026 (threshold measured with `python -m scripts.choose_threshold`)
Serves: FR-04, FR-05, FR-10, acceptance criteria A5, A7 (kill switch), A8; Project Brief section 05;
Governance Framework sections 1 (minimum record), 4 (confidence floor) and 5 (kill switch).
Evidence: `evaluation/results/threshold_choice_20261121_0934.md`, from the classifier's out-of-fold
predictions `evaluation/results/classifier_oof_20261120_1746.csv` (500 development tickets).

## Routing rules (src/route.py), first match decides
| # | Rule | Action | Requirement |
|---|---|---|---|
| 0 | Kill switch off (`AUTO_RESPOND_ENABLED=false`) | escalate | Governance s.5 |
| 1 | Ticket unreadable (empty body, malformed, unknown channel, missing id) | escalate | FR-01, A11 |
| 2 | No reliable classification (fallback, below floor, missing confidence) | escalate | FR-02, Governance s.4 |
| 3 | Must-escalate intent: security incident, compliance request, feature request, unclear request | escalate | FR-05 |
| 4 | Billing ticket mentioning refunds, credits, disputes, double charges, cancellation | escalate | FR-05 |
| 5 | No help article above the relevance threshold (0.35) | escalate | FR-03 |
| 6 | Intent confidence below `CONFIDENCE_THRESHOLD` | escalate | FR-04 |
| 7 | Otherwise: pass to answer drafting (generation + guardrails decide the final action) | answer | FR-06 |

Every decision carries a plain-English reason. No randomness: same input, same decision (A5).
Urgency is never used for routing: it is not predictable from the text (classifier.md, finding 4).

## Confidence threshold: 0.80
Out-of-fold simulation of rules 3 and 6 on development tickets:

| Threshold | Cleared for answer | Wrong intent cleared | Must-not cleared | Cleared but expected escalate | Agreement with expected route |
|---|---|---|---|---|---|
| 0.00 | 82.6% | 7 | 0 | 102 | 79.6% |
| 0.50 | 82.2% | 5 | 0 | 101 | 79.6% |
| 0.60-0.80 | 81.2% | 3 | 0 | 99 | 79.4% |
| 0.85 | 79.6% | 3 | 0 | 98 | 78.2% |
| 0.90 | 79.0% | 2 | 0 | 98 | 77.6% |
| 0.95 | 75.8% | 2 | 0 | 94 | 76.0% |
| 0.99 | 70.0% | 0 | 0 | 87 | 73.0% |

Rule (fixed before choosing): no must-not-auto-respond ticket cleared; accept at most 2 points of
coverage loss to clear fewer wrong intents; on a tie take the highest threshold.
Result: **0.80** - the top of a plateau (0.60-0.80 give identical results), so it adds safety margin at
no measured cost. The Project Brief's illustrative 0.80 is therefore kept, but now measured.
Above 0.85, each wrong intent removed costs 3-6 points of coverage.

## Finding that shapes the rest of the build
The labels say 37.8% of tickets should escalate; classifier + rules catch about 19%. The other
**about 99 tickets (20%) have ordinary intents but no answer in the documentation**
(e.g. database, performance, deployment questions). Neither the classifier, the relevance threshold
(chunking.md: answerable vs unanswerable similarity 0.609 vs 0.598) nor a text model trained to
predict answerability (73% vs a 71% baseline) can identify them.
Consequence: these must be caught at the answer step - PR-02 returning `answerable: false` - and by
the grounding guardrail (B-08, B-09). Routing is necessary but not sufficient; record in the PRD
revision log (Stage 5).

**Outcome (22 Nov, validation run):** the model almost never declines (1 of 63 drafts), so PR-02's
answerability flag does not close this gap. The fact-check blocks drafts that add unsupported claims
(guardrails.md). Most of the remaining "sent but labelled escalate" tickets have an identical-text
twin labelled "answer", and text patterns match the escalate label only at its base rate: the labels
behave like noise for these tickets, which caps agreement for any text-only system.

## Decision log (src/logging_store.py)
- SQLite (`DATABASE_URL`), one row per decision per stage, fields as the Governance Framework's
  minimum record (decision_id, timestamp, ticket_id, stage, input_summary, model name/version,
  prediction + confidence, alternatives, sources_used, threshold_applied, action_taken, reason,
  guardrail_results, prompt_version, requirement_ids) plus run_id, rule and error.
- Customer text is never stored: input_summary = channel, length and a SHA-256 prefix.
- Written for every ticket, including failures (processing errors are escalated and logged).
- Reconciliation: every ticket in a run has a routing decision (`DecisionLog.reconcile`, A8).

## Kill switch
`AUTO_RESPOND_ENABLED=false` in `.env` makes rule 0 escalate every ticket on the next run or API
restart, with no code change or deployment. Who may use it, timing and testing: Governance Framework
section 5 (B-21).

## Limits
- The threshold simulation does not include retrieval (rule 5) or the billing rule (rule 4); the
  harness (B-05) measures the full chain.
- Out-of-fold predictions on templated development data; real tickets may be classified less confidently.
