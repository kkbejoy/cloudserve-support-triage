# Chunking comparison (2026-11-16 23:49)

Tickets: `data\development_tickets.json` - development data. Model: all-MiniLM-L6-v2.

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR | Median best score (answerable) | Median best score (unanswerable) |
|---|---|---|---|---|---|---|
| section | 90.5% | 95.8% | 98.6% | 0.936 | 0.609 | 0.598 |
| whole | 87.4% | 95.8% | 97.2% | 0.921 | 0.566 | 0.537 |
| fixed | 88.8% | 95.0% | 97.2% | 0.926 | 0.583 | 0.56 |

## Threshold table - section

| Threshold | Answerable keep correct article | Unanswerable return nothing |
|---|---|---|
| 0.20 | 100.0% | 4.9% |
| 0.25 | 98.9% | 6.3% |
| 0.30 | 95.8% | 11.2% |
| 0.35 | 94.1% | 14.7% |
| 0.40 | 93.6% | 14.7% |
| 0.45 | 90.5% | 15.4% |
| 0.50 | 81.8% | 23.1% |
| 0.55 | 70.0% | 35.7% |
| 0.60 | 51.3% | 50.3% |
| 0.65 | 30.0% | 69.2% |
| 0.70 | 15.7% | 87.4% |

## Threshold table - whole

| Threshold | Answerable keep correct article | Unanswerable return nothing |
|---|---|---|
| 0.20 | 97.8% | 7.0% |
| 0.25 | 95.8% | 9.1% |
| 0.30 | 94.7% | 11.9% |
| 0.35 | 93.6% | 13.3% |
| 0.40 | 91.6% | 14.7% |
| 0.45 | 84.0% | 22.4% |
| 0.50 | 73.7% | 36.4% |
| 0.55 | 50.4% | 58.0% |
| 0.60 | 32.2% | 72.0% |
| 0.65 | 22.1% | 84.6% |
| 0.70 | 13.2% | 90.9% |

## Threshold table - fixed

| Threshold | Answerable keep correct article | Unanswerable return nothing |
|---|---|---|
| 0.20 | 100.0% | 4.9% |
| 0.25 | 96.6% | 6.3% |
| 0.30 | 96.1% | 11.2% |
| 0.35 | 93.8% | 13.3% |
| 0.40 | 92.2% | 14.7% |
| 0.45 | 89.1% | 14.7% |
| 0.50 | 78.7% | 25.9% |
| 0.55 | 61.9% | 45.5% |
| 0.60 | 41.7% | 67.8% |
| 0.65 | 23.0% | 78.3% |
| 0.70 | 10.1% | 93.0% |
