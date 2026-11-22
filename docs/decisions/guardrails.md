# Decision: guardrails before any answer is sent (B-09)

**Requirements:** FR-09 (check every outgoing answer for private data, unsupported claims and
commitments), NFR-03 (hallucination ≤ 5%, citation accuracy ≥ 95%), NFR-04 (zero private data in
released answers), NFR-10 (ticket text cannot change behaviour), Governance Framework section 4.

## Why

Answer drafting (B-08) almost never declines: on the development set PR-02 v1.0 judged 0 of 374
drafts "not answerable", and v1.1 0 of 29 in the first comparison. "The model wrote an answer" is
therefore not a reason to send it. Every draft passes five checks; the first failure blocks it and
the ticket goes to a person with the draft attached to the hand-off note.

## Before drafting: instruction integrity (on the ticket)

Ticket text that tries to change the system's instructions ("ignore previous instructions", "print
your system prompt", "you are now…", fake `</ticket>` tags) is blocked before anything is drafted,
escalated, and the ticket is appended to `storage/review_queue.jsonl` for review, as the Governance
Framework requires ("block, escalate and record the input for review"). The decision log keeps only a
fingerprint of customer text (NFR-04), so the review queue is the one place the flagged text is kept;
it is git-ignored. The patterns match none of the 580 development and validation tickets.

Secrets in a ticket (keys, emails, card numbers, phone numbers, "password is <value>") are replaced
with `[redacted …]` in the text sent to the model (NFR-04: only the data needed). Answers are never
redacted: a reply that contains private data is blocked.

## The checks on a draft (cheapest first; stop at the first failure)

| # | Check | How | Blocks |
|---|---|---|---|
| 1 | policy | settings | kill switch off; intent confidence below the routing threshold (repeated here as a second line of defence) |
| 2 | private_data | patterns in code | email address, API key/token, "password is …", phone number, card number (Luhn-checked) |
| 3 | scope_tone | patterns in code | refunds/credits, guarantees and promises, "this will/should fix it", dates, waived charges, unprofessional wording, talk about the system itself |
| 4 | citations | code | citation of an article the model was not given |
| 5 | grounding | PR-04 v1.2, one model call per draft | any sentence the fact-checker marks "no", "partial" or skips |
| 6 | citation_coverage | code | a claim that has no citation and whose supporting article cannot be identified |

Only check 5 costs tokens, and it runs only for drafts that passed 1–4.

**Citations are attached by code (FR-06).** The first run with guardrails (22 Nov, 20 tickets) sent
answers where several sentences had no citation (three answers had none inline). The model forgets
them; the fact-check has just matched each sentence to a passage. So PR-04 v1.2 also returns the
supporting article per sentence, and code adds `[DOC-…]` to any sentence the model left uncited (one
article sent = that article). The model's own citations are kept; greetings need none. A claim that
still cannot be cited blocks the draft. Blocking uncited answers instead would have cut automatic
answers on that sample from 14 to about 9.

## Choices and trade-offs

- **Fail closed.** If the fact-check cannot run (provider down, daily limit, unreadable reply) or a
  check crashes, the draft is blocked. Cost: fewer automatic answers during outages. Benefit: an
  unchecked answer never reaches a customer.
- **"Partial" blocks by default** (`GROUNDING_ALLOW_PARTIAL=false`). A partly supported sentence is
  how invented steps and promises get in (development ticket DEV-0011). Can be relaxed if
  measurement shows too many good answers lost.
- **One fact-check call per draft, not per sentence.** PR-04 v1.0 checked one sentence against one
  passage per call; with ~3 sentences and 1–2 articles that is 3–6 calls per draft, beyond the free
  tier (200K tokens/day measured, ~1,000 tokens per call). v1.1 checks all sentences in one call.
- **Separate fact-check model is optional** (`GROUNDING_PROVIDER`, `GROUNDING_MODEL`). On Groq's
  free plan limits are per model, so a different model gives the check its own daily allowance.
- **Refund wording blocks even when the article mentions refunds** (DOC-BILL-002). FR-05 sends
  refund requests to a person anyway; a sent answer talking about refunds is the riskier error.
- **Pattern lists were checked for false alarms**: none of the 29 help articles triggers the
  private-data patterns, only DOC-BILL-002 triggers the refund pattern, and none of the 200
  reference responses in `ground_truth_responses.json` triggers either list.

## What the checks cannot catch

A draft that is fully supported by the articles but does not resolve the customer's situation
(development ticket DEV-0002: "key returns 401, nothing changed on our side") passes every check.
Many such tickets carry contradictory labels in the data (the same text labelled answerable once and
not answerable once), which caps any text-only system; this is reported, not tuned away.

## How it is measured (harness, every run)

- Drafts checked / passed (sent) / blocked by check.
- **Sent but labelled escalate** — the main safety number; must-not-auto-respond tickets answered
  must be 0.
- Not-answerable drafts blocked (gap closed) and answerable drafts blocked (automatic answers lost).
- Fact-check tokens.

Every check result is written to the decision log (`validation` stage, `guardrail_results`).

## To revisit after the validation run

Whether "partial" should pass; whether any scope pattern blocks good answers too often (read the
blocked drafts); fact-check token cost against the daily allowance.
