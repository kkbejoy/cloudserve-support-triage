# Example tickets

One ticket per channel, two tickets built to trigger the instruction-integrity guardrail, and one
that must always go to a person. Use them to try the API or the harness.

| File | Channel | What should happen |
|---|---|---|
| `tickets/01_email_login.json` | email | answered automatically with citations, or escalated if a check fails |
| `tickets/02_chat_rollback.json` | chat | answered automatically (non-fluent wording) or escalated |
| `tickets/03_docs_comment_webhook.json` | docs_comment | answered automatically or escalated |
| `tickets/04_forum_api_key.json` | forum | answered automatically or escalated |
| `tickets/05_guardrail_injection.json` | email | **blocked**: `blocked_instruction_integrity`, kept in `storage/review_queue.jsonl` |
| `tickets/06_guardrail_reveal_prompt.json` | chat | **blocked**: `blocked_instruction_integrity` |
| `tickets/07_must_escalate_security.json` | email | **escalated**: security incidents always go to a person (FR-05) |

Whether 01–04 are answered depends on the model's draft passing every guardrail; the response says
which check decided and why.

## Through the API

Start the server: `uvicorn src.api:app`. Then either open http://127.0.0.1:8000/docs and paste a
file's contents into `POST /tickets`, or:

Windows PowerShell
```
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tickets -ContentType "application/json" -InFile examples/tickets/05_guardrail_injection.json
Invoke-RestMethod http://127.0.0.1:8000/decisions/EX-GUARD-1
```

macOS / Linux / Git Bash
```
curl -s -X POST http://127.0.0.1:8000/tickets -H "Content-Type: application/json" -d @examples/tickets/05_guardrail_injection.json
curl -s http://127.0.0.1:8000/decisions/EX-GUARD-1
```

## Through the harness (all seven in one run)

```
python -m evaluation.harness --input examples/all_examples.json --output evaluation/results/
```
