# Fairness audit (B-14, NFR-06)

**Run:** validation set, 80 tickets, `run_20261122T024310-e4330f` (22 Nov 2026). Regenerate for
any run with `python -m scripts.fairness_audit --run <run folder> --tickets <ticket file>`.

**Result in one line:** the system raises first-contact resolution (FCR) for every group, and it
closes the largest historical gap (language fluency). NFR-06 as written (raw FCR within 5 points) is
not met for tier, region and channel. Most of the tier gap comes from the mix of tickets each group
sends, not from how the system treats them.

## By design

The classifier, retrieval and routing read only the ticket text. Tier, region and fluency are never
inputs (docs/decisions/classifier.md), so the system cannot treat a group differently because of who
the customer is. Differences come from what the tickets ask and how they are worded.

## What the run shows

| Group | Tickets | FCR (system) | FCR (history, same tickets) | Answerable tickets answered |
|---|---|---|---|---|
| Tier: business | 30 | 86.7% | 53.3% | 95.2% of 21 |
| Tier: enterprise* | 8 | 87.5% | 62.5% | 100% of 6 |
| Tier: standard | 42 | 59.5% | 38.1% | 81.0% of 21 |
| Fluency: fluent | 61 | 70.5% | 54.1% | 87.5% of 40 |
| Fluency: non-fluent* | 19 | 78.9% | 21.1% | 100% of 8 |
| Region: Latin America* | 7 | 28.6% | 28.6% | 66.7% of 3 |

\* fewer than 20 tickets: one ticket moves the rate by 5–14 points. Full tables per tier, region,
fluency and channel are in the run folder (`fairness.md`).

1. **Language fluency.** Historically non-fluent customers were resolved at first contact 21% of the
   time against 54% for fluent customers: a 33-point gap. The system answers 79% of non-fluent
   tickets and every answerable one. The gap is closed and reversed. Caveat: 7 of the 15 answers
   sent to non-fluent customers were labelled "escalate" (see the label-noise finding, KI-06/KI-18).
2. **Tier: a 27-point gap, mostly ticket mix.** Half of standard-tier tickets are labelled
   "escalate" (21 of 42, including 5 unclear requests, 3 security incidents, 2 feature requests and
   1 compliance request, which must go to a person), against 9 of 30 for business. Among answerable
   tickets the gap shrinks to 14 points: standard lost 4, of which 3 were fact-check blocks on
   data-residency and data-export answers and 1 had no matching article. Those losses follow the
   topic, not the tier. The historical gap between standard and enterprise was 24 points.
3. **Channel.** Raw FCR differs by 14 points, but among answerable tickets by 1 point: channel does
   not change how answerable tickets are handled.
4. **Region.** Latin America has 7 tickets; its rate is not reliable on this sample.

## Against NFR-06

NFR-06 compares raw FCR between groups. That number mixes two things: how many of a group's tickets
must go to a person (set by what customers ask), and how well the system handles the rest. The
first is outside the system's control and the second is what fairness is about.

Proposed revision (for the Stage 5 PRD revision log): measure the gap on answerable tickets, for
groups of at least 20 tickets, and report it next to the historical gap. On this run, under that
measure, channel meets the 5-point limit and tier does not (14 points, driven by three fact-check
blocks on one topic).

## Follow-ups

- Re-run this audit on the gate run; validation groups are small.
- Read the three blocked data-residency/data-export drafts: if the fact-check is too strict for that
  article, it costs the standard tier most.
- Watch answers sent to non-fluent customers that were labelled escalate (7 of 15).
