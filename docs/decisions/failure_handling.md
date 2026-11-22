# Failure handling (B-18, A11, NFR-02)

Rule: a failure never crashes the run and never sends an unchecked answer. The affected ticket goes
to a person, the reason is written to the decision log, and the run carries on.

| Failure | Where it is caught | What happens to the ticket | Shown in | Test |
|---|---|---|---|---|
| Malformed ticket (missing body, unknown channel, bad field, duplicate id) | `src/ingest.py` | kept, flagged; routing escalates it (`unreadable`) | `problems_in_input` in metrics; decision log | `test_ingest`, `test_api` T16d |
| No relevant help article (retrieval below threshold) | `src/route.py` rule `no_relevant_docs` | escalated with acknowledgement and hand-off note | decision log, `rule` | `test_pipeline_log` |
| Classifier unsure (confidence below floor) | `src/classify.py` → `unclear_request` | escalated | decision log | `test_classify` |
| Bad API key / unknown model name | harness start-up check (`_preflight`) | run stops before processing, exit code 2, message says what to fix | console | `test_harness` T11d |
| Provider timeout or 5xx | `src/llm.py`: retries with backoff | if still failing: escalated (`provider_unavailable`) | `provider_unavailable` count | `test_generate` T18c |
| Rate limit (429) | `src/llm.py`: waits (Retry-After, capped), paces calls | as above if it does not recover | same | `test_generate` |
| Provider down for the whole run | circuit breaker after 3 consecutive failures: no more calls | every cleared ticket escalated quickly; run completes and reconciles; summary warns **degraded run** (>5% provider errors) | `degraded_run` flag | `test_harness` T18f |
| Daily token limit reached mid-run | same as above (429 that does not recover) | escalated; cached replies still used | same | T18f |
| Model reply not readable JSON | `src/generate.py` / `src/guardrails.py`: one retry | escalated (`generation_failed`) or blocked (fact-check unreadable) | decision log | `test_generate` T06c, `test_guardrails` T09g |
| Fact-check cannot run | `src/guardrails.py` | blocked, escalated ("blocked to be safe") | `blocked_by_check` | T09g |
| Bug inside a guardrail | `Guardrails.check` catch-all | blocked, never sent | `failed_check = error` | T09j |
| Unexpected error while processing one ticket | `src/pipeline.py` catch-all | escalated (`processing_error`); other tickets continue | `processing_errors` | `test_harness` |
| Index or classifier missing | start-up of harness / API | harness exits 2; API starts, `/health` says `not_ready` and how to fix (`python -m scripts.setup`) | console, `/health` | `test_api` T16e, `test_setup` |
| Ticket tries to change the system's instructions | `Guardrails.check_input` | blocked before drafting, kept in the review queue | `blocked_instruction_integrity` | T09n, T16c |
| Kill switch (`AUTO_RESPOND_ENABLED=false`) | `src/route.py` rule 0 and guardrail policy check | nothing sent; everything escalated | decision log | `test_route`, T09b |

Two rules hold throughout: nothing is counted as resolved unless it was sent after every check
passed, and every ticket in the input file has a decision in the log (reconciled at the end of each
run, A8).
