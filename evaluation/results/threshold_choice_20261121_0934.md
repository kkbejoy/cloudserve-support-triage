# Routing confidence threshold (2026-11-21 09:34)

Out-of-fold predictions: `evaluation/results\classifier_oof_20261120_1746.csv` (500 development tickets). Must-escalate intents: compliance_request, feature_request, security_incident, unclear_request.

| Threshold | Cleared for answer | Wrong intent cleared | Must-not cleared | Cleared but expected escalate | Agreement with expected route |
|---|---|---|---|---|---|
| 0.00 | 82.6% | 7 | 0 | 102 | 79.6% |
| 0.50 | 82.2% | 5 | 0 | 101 | 79.6% |
| 0.60 | 81.2% | 3 | 0 | 99 | 79.4% |
| 0.70 | 81.2% | 3 | 0 | 99 | 79.4% |
| 0.75 | 81.2% | 3 | 0 | 99 | 79.4% |
| 0.80 | 81.2% | 3 | 0 | 99 | 79.4% |
| 0.85 | 79.6% | 3 | 0 | 98 | 78.2% |
| 0.90 | 79.0% | 2 | 0 | 98 | 77.6% |
| 0.95 | 75.8% | 2 | 0 | 94 | 76.0% |
| 0.97 | 74.6% | 2 | 0 | 93 | 75.2% |
| 0.99 | 70.0% | 0 | 0 | 87 | 73.0% |

Recommended: **0.80**. Rule: no must-not-auto-respond ticket cleared; give up at most 2 points of coverage to clear fewer wrong intents; on a tie take the highest threshold (most safety margin at no measured cost).
