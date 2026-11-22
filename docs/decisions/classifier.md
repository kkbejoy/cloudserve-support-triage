# Decision: intent and urgency classifier (B-06)

Status: **decided** - 20 Nov 2026
Serves: FR-02, acceptance criterion A3, NFR-03 (precision >= 85%, calibration within 5 points), NFR-08.
Evidence: `evaluation/results/classifier_comparison_20261120_1746.md` (+ `.json`) and out-of-fold
predictions `evaluation/results/classifier_oof_20261120_1746.csv`.

## Constraints
- Confidence must match observed accuracy within 5 points (Project Brief section 07) - it drives routing.
- Same input, same output (A5); free; works when the model provider is down (A11).
- Text only: customer tier, region and language fluency are never features (NFR-06 fairness).
- 500 development tickets, 22 intents (13-29 each), 3 urgency levels. 500 tickets but only 343 unique
  texts, so cross-validation keeps identical texts in the same fold (grouped, stratified, 5 folds).

## Options compared
| Model | Features | Why considered |
|---|---|---|
| logreg_emb | MiniLM embeddings | Main candidate: real probabilities, reuses retrieval's model |
| knn_emb | MiniLM embeddings, 10 neighbours (distance-weighted, cosine) | Simple and explainable ("looks like these past tickets") |
| tfidf_lr | word + bigram counts | Baseline: shows whether embeddings help |
| LLM (PR-07) | prompt | Not run: no calibrated confidence, one API call per ticket, free-tier limits |

Selection rule (set before the run): highest intent macro precision, tie-break macro F1.
Calibration: one temperature per target fitted on out-of-fold predictions (minimum log-loss).

## Results - development tickets, 5-fold grouped cross-validation, run 20 Nov 2026
Intent:
| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Top-3 | ECE raw | Temperature | ECE calibrated |
|---|---|---|---|---|---|---|---|---|
| logreg_emb | 98.2% | 98.6% | 98.2% | 98.3% | 99.0% | 13.4% | 0.35 | 1.1% |
| **knn_emb** | 98.6% | 98.7% | 98.6% | 98.6% | 99.8% | 3.9% | 0.5 | 0.9% |
| tfidf_lr | 98.2% | 98.5% | 98.3% | 98.2% | 98.6% | 16.1% | 0.45 | 1.0% |

Urgency (majority-class baseline "medium" = 45.2%):
| Model | Accuracy | Macro precision | Macro recall |
|---|---|---|---|
| logreg_emb | 47.8% | 45.8% | 44.9% |
| knn_emb | 47.8% | 45.0% | 44.9% |
| tfidf_lr | 50.0% | 47.6% | 46.9% |

## Decision
Chosen: **knn_emb** (MiniLM embeddings + 10 nearest neighbours), per the selection rule.
- Highest intent macro precision (98.7%) and recall (98.6%); top-3 accuracy 99.8%.
- Best-calibrated before correction (ECE 3.9%); 0.9% after temperature - within the 5-point requirement.
- Explainable: a prediction can be shown as "similar to these past tickets".
- The lead over the other two is 0.1-0.2 points (about one ticket): all three exceed the 85% target, so the
  choice is low-risk; logreg_emb is the fallback (`--model logreg_emb`).

Floor (`CLASSIFIER_FLOOR`): **0.30** kept. No out-of-fold prediction scored below it (lowest 0.48), so it only
catches broken or truly unknown input. The operating cut-off for answering is the routing confidence
threshold (B-07), chosen from the out-of-fold predictions.

## Findings
1. **Intent is highly predictable** from the wording (about 98.6% for every method), well above the 85% target.
2. **Errors are confined to neighbouring topics** - 7 of 500: deployment_failure -> integration_help (3),
   deployment_failure -> rollback_request (2), rate_limit -> performance_degradation (1),
   api_key_issue -> account_access (1). None involve security, compliance or other must-escalate intents.
3. **Calibration holds where it matters**: 477 predictions above 0.9 confidence were 99.6% correct.
   Most errors carry lower confidence (0.48-0.58) and can be caught by the routing threshold;
   two were confident (0.97, e.g. DEV-0001, non-fluent wording) - these rely on the answer-step and guardrail checks.
4. **Urgency is not predictable from text**: 48-50% accuracy against a 45% majority baseline, for every method.
   Consequence: predicted urgency must not drive routing; FR-13 (urgency-ordered queue) needs revisiting -
   to record in the PRD revision log (Stage 5): options are keyword rules, treating urgency as advisory, or
   ordering by age with urgency as a tie-break.

## Limits
- Calibrated ECE is measured on the same predictions the temperature was fitted on (slightly optimistic).
- Development data is highly templated (343 unique texts in 500); scores on differently-worded tickets may be lower.
- Measured on development data only; validation set not used for choices.
