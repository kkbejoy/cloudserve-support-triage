# Classifier comparison (20261120_1746)

Development tickets: 500 (343 unique texts), 5-fold grouped cross-validation. Embeddings: all-MiniLM-L6-v2. Chosen: **knn_emb**.

## Intent

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Top-3 | ECE raw | Temperature | ECE calibrated |
|---|---|---|---|---|---|---|---|---|
| logreg_emb | 98.2% | 98.6% | 98.2% | 98.3% | 99.0% | 13.4% | 0.35 | 1.1% |
| knn_emb | 98.6% | 98.7% | 98.6% | 98.6% | 99.8% | 3.9% | 0.5 | 0.9% |
| tfidf_lr | 98.2% | 98.5% | 98.3% | 98.2% | 98.6% | 16.1% | 0.45 | 1.0% |

## Urgency

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Top-3 | ECE raw | Temperature | ECE calibrated |
|---|---|---|---|---|---|---|---|---|
| logreg_emb | 47.8% | 45.8% | 44.9% | 45.1% | 100.0% | 17.2% | 2.85 | 3.0% |
| knn_emb | 47.8% | 45.0% | 44.9% | 44.6% | 100.0% | 18.4% | 6.0 | 4.1% |
| tfidf_lr | 50.0% | 47.6% | 46.9% | 47.1% | 100.0% | 19.3% | 3.4 | 4.6% |

## Chosen model (knn_emb): intent per class

| Intent | Precision | Recall | Support |
|---|---|---|---|
| integration_help | 87.5% | 100.0% | 21 |
| rollback_request | 93.3% | 100.0% | 28 |
| account_access | 95.7% | 100.0% | 22 |
| performance_degradation | 95.8% | 100.0% | 23 |
| api_key_issue | 100.0% | 94.4% | 18 |
| api_usage_question | 100.0% | 100.0% | 24 |
| authentication_failure | 100.0% | 100.0% | 20 |
| billing_query | 100.0% | 100.0% | 24 |
| compliance_request | 100.0% | 100.0% | 26 |
| configuration_help | 100.0% | 100.0% | 17 |
| data_export | 100.0% | 100.0% | 29 |
| data_residency | 100.0% | 100.0% | 29 |
| database_issue | 100.0% | 100.0% | 26 |
| deployment_failure | 100.0% | 81.5% | 27 |
| feature_request | 100.0% | 100.0% | 20 |
| onboarding | 100.0% | 100.0% | 22 |
| quota_or_overage | 100.0% | 100.0% | 23 |
| rate_limit | 100.0% | 92.3% | 13 |
| security_incident | 100.0% | 100.0% | 26 |
| sso_configuration | 100.0% | 100.0% | 26 |
| unclear_request | 100.0% | 100.0% | 15 |
| webhook_issue | 100.0% | 100.0% | 21 |

Most confused (true -> predicted):

- deployment_failure -> integration_help: 3
- deployment_failure -> rollback_request: 2
- rate_limit -> performance_degradation: 1
- api_key_issue -> account_access: 1

## Calibration (knn_emb, intent, after temperature)

Temperature is fitted on the same out-of-fold predictions it is measured on, so the calibrated ECE is slightly optimistic.

| Confidence bin | Tickets | Mean confidence | Accuracy |
|---|---|---|---|
| 0.4-0.5 | 2 | 0.481 | 0.0 |
| 0.5-0.6 | 7 | 0.529 | 0.714 |
| 0.8-0.9 | 14 | 0.851 | 0.929 |
| 0.9-1.0 | 477 | 0.993 | 0.996 |
