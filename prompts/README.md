# Prompt library

Versioned prompts from the Stage 3 Prompt Library. File name = `PR<id>_<name>_v<version>.txt`.
The first line (`# PR-0X ...`) is a header and is not sent to the model. `src/prompts.py` loads the
newest version of an id; the id and version (e.g. `PR-02 v1.0`) are written to the decision log
(`prompt_version`) so every drafted answer can be traced to the exact prompt text.

Pin a version with `PROMPT_PR02_VERSION=1.0` in `.env`. Compare versions on the development set
with `python -m scripts.compare_prompts --versions 1.0,1.1`.

To change a prompt: add a new file with a higher version (never edit a released version), run the
tests and the harness, and record the change and its measured effect in the Prompt Library.

| Id | File | Used by | Status |
|---|---|---|---|
| PR-01 | dev/PR01_spec_from_requirement_v1.0.txt | development (specs from requirements) | dev aid |
| PR-02 | build/PR02_answer_with_citations_v1.1.txt | `src/generate.py` - answer drafting (FR-06, FR-07) | in use |
| PR-02 | build/PR02_answer_with_citations_v1.0.txt | first version; never refused (0 of 374 drafts, development set) | superseded |
| PR-03 | build/PR03_escalation_summary_v1.0.txt | `src/generate.py` - escalation note when `ESCALATION_NOTE_LLM=true` (FR-08) | optional |
| PR-04 | build/PR04_claim_support_check_v1.2.txt | `src/guardrails.py` - fact-check of every sentence of a draft, one call, names the supporting article so code can cite it (FR-06, FR-09) | in use |
| PR-04 | build/PR04_claim_support_check_v1.1.txt | one-call fact-check without the supporting article | superseded |
| PR-04 | build/PR04_claim_support_check_v1.0.txt | one sentence x one passage per call; too many calls for the free tier | superseded |
| PR-05 | evaluation/PR05_answer_quality_eval_v1.0.txt | answer-quality evaluation | planned |
| PR-06 | dev/PR06_code_review_v1.0.txt | development (code review) | dev aid |
| PR-07 | dev/PR07_llm_intent_baseline_v1.0.txt | classifier comparison baseline | dev aid |

Placeholders are `{name}`; JSON braces in a prompt are left alone. Customer text is wrapped in
`<ticket>` tags and any tag-like text inside it is neutralised first (prompt-injection defence).
