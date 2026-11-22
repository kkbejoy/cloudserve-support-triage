# CloudServe support triage

[![CI](https://github.com/kkbejoy/cloudserve-support-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/kkbejoy/cloudserve-support-triage/actions/workflows/ci.yml)

An intelligent support triage system for CloudServe's support queue. Every ticket, from any of the
four channels, is classified, matched against the 29 help-centre articles, and either **answered
automatically with citations** or **escalated to a person** with a hand-off note. An answer is sent
only if it passes every guardrail. Every decision is written to a permanent log.

Built for the Forward Deployed Engineering capstone (CloudServe brief). Python 3.11, runs locally,
free-tier model provider (Groq).

**Contents:** [Quick start](#quick-start) · [Results](#results-validation-set) ·
[How it works](#how-it-works) · [Configuration](#configuration) ·
[Repository layout](#repository-layout) · [Troubleshooting](#troubleshooting) ·
[Known limitations](#known-limitations)

---

## Quick start

These steps follow the assessment procedure in order. Commands are shown for **Windows
PowerShell**; macOS/Linux differences are noted underneath.

### 1. Clone into an empty folder

```
git clone https://github.com/kkbejoy/cloudserve-support-triage.git
cd cloudserve-support-triage
```

### 2. Create the environment and install the pinned dependencies

Needs **Python 3.10 or newer** (built and tested on 3.11).

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`.
If PowerShell refuses to run the activate script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

The install is large (PyTorch, sentence-transformers) and takes several minutes.

### 3. Configure: copy `.env.example` and add a key

```
copy .env.example .env
```
macOS/Linux: `cp .env.example .env`

Open `.env` and replace `your_key_here` in **`GROQ_API_KEY`** with a Groq API key (free at
https://console.groq.com/keys). Nothing else has to change.

Recommended on the free tier: also set `GROUNDING_MODEL=openai/gpt-oss-120b`. The fact-check then
uses its own daily allowance, separate from answer drafting (Groq limits are per model).

To use OpenRouter instead, fill `OPENROUTER_API_KEY`, leave `GROQ_API_KEY` as the placeholder, and
set `MODEL_NAME` to an OpenRouter model id.

### 4. One-time setup

```
python -m scripts.setup
```
This builds what is not stored in git: the search index over the documentation and the
intent/urgency classifier, both from the files in `data/`. Then it runs a self-test. The first run
downloads the embedding model `all-MiniLM-L6-v2` (about 90 MB) from Hugging Face and takes a few
minutes. It ends with **`Ready.`** Running it again skips work that is already done.
`python -m scripts.setup --check-llm` also confirms the API key works (one tiny call).

### 5. Start the system

```
uvicorn src.api:app
```
Leave it running and use a second terminal (activate the venv there too). Check
http://127.0.0.1:8000/health: it should say `"status": "ok"`. The interactive page at
http://127.0.0.1:8000/docs lets you submit tickets from the browser.

### 6. Submit one ticket from each channel

Ready-made tickets are in [`examples/tickets/`](examples/README.md).

```
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tickets -ContentType "application/json" -InFile examples/tickets/01_email_login.json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tickets -ContentType "application/json" -InFile examples/tickets/02_chat_rollback.json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tickets -ContentType "application/json" -InFile examples/tickets/03_docs_comment_webhook.json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tickets -ContentType "application/json" -InFile examples/tickets/04_forum_api_key.json
```
macOS/Linux:
`curl -s -X POST http://127.0.0.1:8000/tickets -H "Content-Type: application/json" -d @examples/tickets/01_email_login.json`

Each response carries `intent` and `intent_confidence`, the retrieved `sources`, the `final_action`
(`auto_respond`, `escalate` or `block`) with its `rule` and `reason`, then the `answer` (with
`[DOC-…]` citations on every sentence) or the `acknowledgement` and `escalation_note`, and the
ticket's `decision_log` entries.

### 7. Submit a ticket engineered to trigger a guardrail

```
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tickets -ContentType "application/json" -InFile examples/tickets/05_guardrail_injection.json
```
Expected: `"final_action": "block"`, `"rule": "blocked_instruction_integrity"`, no answer, an
acknowledgement for the customer, and the ticket kept in `storage/review_queue.jsonl` for review.
`06_guardrail_reveal_prompt.json` is a second example. `07_must_escalate_security.json` shows a
security incident that always goes to a person.

### 8. Full unattended run on a ticket file

```
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/
```
The command takes any ticket file with the same format, for example a file the system has never
seen. The 80 validation tickets take about 6 minutes on Groq's free tier, because answer drafting
and the fact-check wait for the provider's per-minute limits. The run never stops for input. If the
provider fails, affected tickets are escalated and the summary says **degraded run**. A wrong key or
model name stops it at the start with a clear message (exit code 2).

Without an API key the run stops at the start with a message saying which key to set (exit code 2).
To try the pipeline without one, add `--no-generate`: tickets are classified, retrieved and routed,
but nothing is drafted or sent.

The run writes a folder `evaluation/results/run_<run id>/` with:

| File | What it is |
|---|---|
| `metrics.md` | The metrics report: volume, business, technical, governance, and results by customer group |
| `metrics.json` | The same, machine-readable |
| `responses.jsonl` | One line per ticket: decision, answer or escalation, sources, guardrails, latency |

`--limit 20` runs only the first 20 tickets. `--no-generate` stops after routing (no model calls).

### 9. Reconcile the metrics report with the decision log

The decision log is the SQLite file **`storage/decisions.db`** (schema:
[`docs/decision_log_schema.sql`](docs/decision_log_schema.sql)). The *Governance* section of
`metrics.md` already reports whether every ticket in the run has a decision (`Log reconciled`). To
check it yourself, count decisions per run:

```
python -c "import sqlite3; c=sqlite3.connect('storage/decisions.db'); print(c.execute('select run_id, count(distinct ticket_id), count(*) from decisions group by run_id order by run_id desc limit 3').fetchall())"
```
The harness run's id matches its folder name (`run_<run id>`), and its distinct ticket count equals
*tickets processed* in `metrics.md`. The API server also writes a run of its own (the tickets
submitted in steps 6 and 7), so it appears in the list too. Each ticket has one record per stage it went through (`classification`, `routing`,
then `generation` and `validation` when an answer was drafted), with the action, the rule, a
plain-English reason, the sources and the guardrail results. Customer text is not stored in the
log, only its channel, length and a fingerprint.

Any SQLite viewer works too, for example DB Browser for SQLite.

### 10. Run the tests

```
python -m pytest
```
88 tests, about 1 minute, no API key or network needed: a fake model and a fake embedder stand in.
Coverage report: `python -m pytest --cov` (89% of the system code). Style check: `flake8`.

Every push runs the same checks on GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):
lint and tests, then a smoke run of steps 3, 4 and 8 on a clean machine (setup from
`.env.example`, then the unattended run with `--no-generate`, so no API key is needed).

---

## Results (validation set)

Full run on the 80 validation tickets, 22 Nov 2026 (Groq free tier: `openai/gpt-oss-20b` drafting,
`openai/gpt-oss-120b` fact-check).

| Measure | Result |
|---|---|
| First-contact resolution (answered automatically, every check passed) | **72.5%** (historical baseline for the same tickets: 46.2%) |
| Tickets that must never be auto-answered, answered | **0** |
| Sent answers with a citation on every sentence | **58 of 58** |
| Citations pointing at the expected article | **97.7%** (43 of 44 with an expected article) |
| Retrieval finds the expected article | 96.2% |
| Drafts blocked by the fact-check | 5 (each added a claim the article does not make) |
| Decision log reconciled with tickets processed | 80 of 80 |
| Latency per ticket, median / 95th percentile | 4.1 s / 10.4 s (two model calls plus free-tier rate limits) |

Read with care: 57 of the 80 validation tickets repeat a development ticket's text, so the
classifier's validation precision (100%) is optimistic; the cross-validated figure on development
data (grouped 5-fold, identical texts kept together) is 98.7% macro precision. 15 of the 58 sent
answers are labelled "escalate" in the data; 10 of those have an identical development ticket
labelled "answer". Fairness by customer group: [`docs/fairness_audit.md`](docs/fairness_audit.md).

## How it works

```
ticket (email | chat | docs_comment | forum)
  -> ingest       one internal format; malformed records flagged, never dropped
  -> classify     intent (22 classes) + urgency, each with a confidence (kNN on MiniLM embeddings, calibrated)
  -> retrieve     relevant help-article sections from the Chroma index; the full article is used
  -> route        deterministic rules: must-escalate intents, billing disputes, no relevant article,
                  confidence below 0.80 -> escalate; otherwise go on
  -> input check  ticket text that tries to change the system's instructions -> block
  -> draft        LLM writes an answer from the retrieved articles only (prompt PR-02)
  -> guardrails   private data, commitments/tone, citations, sentence-by-sentence fact-check
                  (prompt PR-04); code attaches a citation to every claim
  -> send (auto_respond) or block/escalate with acknowledgement + hand-off note
  -> decision log every stage, every ticket
```

Design decisions, each with the measurements behind it, are in [`docs/decisions/`](docs/decisions/):
chunking, classifier, routing threshold, guardrails, failure handling. The fairness audit is in
[`docs/fairness_audit.md`](docs/fairness_audit.md). Prompts are versioned in
[`prompts/`](prompts/README.md).

## Configuration

All settings live in `.env`; every one has a working default in `.env.example`, with a comment.
The ones you are most likely to touch:

| Setting | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` / `OPENROUTER_API_KEY` | placeholder | provider key (one is enough) |
| `MODEL_NAME` | provider default (`openai/gpt-oss-20b` on Groq) | answer-drafting model |
| `GROUNDING_MODEL` | same as drafting | fact-check model; `openai/gpt-oss-120b` recommended on Groq |
| `AUTO_RESPOND_ENABLED` | `true` | **kill switch**: `false` sends nothing, everything goes to a person |
| `CONFIDENCE_THRESHOLD` | `0.80` | below this intent confidence a ticket is escalated |
| `RELEVANCE_THRESHOLD` | `0.35` | below this retrieval score there is "no relevant article" |
| `MUST_ESCALATE_INTENTS` | security, compliance, feature requests, unclear | never answered automatically |

Model replies are cached in `storage/llm_cache.sqlite`, so re-running the same tickets makes no new
calls. Delete the file to force fresh ones.

## Repository layout

```
src/          the system: ingest, classify, retrieve, route, generate, guardrails, pipeline, api, decision log
evaluation/   harness (one command, any input file) and metrics
scripts/      setup, index build, classifier training, fairness audit, and the experiments behind docs/decisions
prompts/      versioned prompts (PR-01 ... PR-07)
data/         the supplied tickets, documentation and reference responses
examples/     example tickets for the API and the harness
docs/         design decisions, decision-log schema, fairness audit
tests/        pytest suite (offline)
storage/      built and runtime files - not in git, created by setup and by runs
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Windows fatal exception: access violation` while running tests | Fixed in `tests/conftest.py`. If it appears elsewhere, install the latest Microsoft Visual C++ Redistributable (x64). |
| `Vector index ... not found` or `Classifier model not found` | Run `python -m scripts.setup`. |
| `/health` says `not_ready` | Same: run setup, then restart `uvicorn`. |
| `LLM setup problem ... HTTP 401` | The key in `.env` is wrong or still the placeholder. |
| `LLM setup problem ... HTTP 404 model_not_found` | The provider no longer offers that model. Set `MODEL_NAME` to a current one from the provider's model list. |
| `HTTP 429 ... tokens per day` in the log, run marked degraded | The free daily allowance is used up. Affected tickets were escalated safely. Wait for it to recover, set `GROUNDING_MODEL` to a different model, or re-run later (finished replies come from the cache). |
| First setup run seems stuck | It is downloading the embedding model (about 90 MB) and PyTorch is loading. |

## Known limitations

- Answer drafting and the fact-check need a model provider. With none, tickets are escalated, not
  answered.
- End-to-end latency is about 4 s per ticket (median) on Groq's free tier, dominated by two model
  calls and rate-limit waits. The system's own processing takes well under a second.
- Many tickets in the supplied data carry contradictory labels (the same text labelled "answer" once
  and "escalate" once). This caps agreement with the labels for any system that reads only the
  text. See the report.

## Attribution

All code in this repository was written for this project. The datasets in `data/` and the project
brief are from the course pack. Open-source libraries are listed in `requirements.txt`.
