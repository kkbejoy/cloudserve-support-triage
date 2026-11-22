# Decision: chunking strategy and relevance threshold (B-03 / B-04)

Status: **decided** - 16 Nov 2026
Serves: FR-03, acceptance criterion A4. Setup Guide section 04: "try at least two configurations, measure, record".
Evidence: `evaluation/results/chunking_comparison_20261116_2349.md` (and `.json`)

## Facts about the corpus
- 29 articles, 10 categories, 138-214 words each (about 177 on average).
- Every article has the same four sections: Symptoms, Common causes, Resolution, Notes.
- The longest articles (about 280-300 tokens) exceed all-MiniLM-L6-v2's 256-token input, so a
  whole-article embedding loses the end (usually Notes).
- Dataset Guide warns that splitting inside a resolution sequence gives passages that "retrieve well
  but read as incomplete".

## Options compared
| Option | Chunks | Description |
|---|---|---|
| section | 116 | One chunk per section, prefixed with title + applies-to; hits grouped by `doc_id`, full article returned ("small-to-big") |
| whole | 29 | One chunk per article |
| fixed | 59 | 800 characters, 120 overlap (Setup Guide baseline, LangChain RecursiveCharacterTextSplitter) |

All use all-MiniLM-L6-v2, cosine similarity and Chroma (embedded, local). Citations are article-level `doc_id`s in every case.

## Results - development tickets (357 answerable, 143 unanswerable), run 16 Nov 2026
| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR | Median best score, answerable | Median best score, unanswerable |
|---|---|---|---|---|---|---|
| **section** | **90.5%** | **95.8%** | **98.6%** | **0.936** | 0.609 | 0.598 |
| whole | 87.4% | 95.8% | 97.2% | 0.921 | 0.566 | 0.537 |
| fixed | 88.8% | 95.0% | 97.2% | 0.926 | 0.583 | 0.560 |

Relevance threshold table (section):
| Threshold | Answerable keep correct article | Unanswerable return nothing |
|---|---|---|
| 0.30 | 95.8% | 11.2% |
| **0.35** | **94.1%** | **14.7%** |
| 0.40 | 93.6% | 14.7% |
| 0.45 | 90.5% | 15.4% |
| 0.50 | 81.8% | 23.1% |
| 0.60 | 51.3% | 50.3% |

## Decision 1 - chunking: section
Best or joint best on every measure; ranks the correct article first most often (+3.1 points Hit@1
over whole) and has the highest MRR. Keeps each resolution sequence intact and every chunk within the
model's 256-token limit. Hit@3 of 95.8% exceeds the FR-03 target of 80%.

## Decision 2 - relevance threshold: 0.35 (`RELEVANCE_THRESHOLD`)
0.35 rejects as many unanswerable tickets as 0.40-0.45 (about 15%) while keeping the most answerable
tickets' correct article (94.1%). Above 0.45 the cost rises steeply: at 0.50, 18% of answerable tickets
lose their article; at 0.60, about half do.

## Finding that shapes the design
Similarity scores barely separate answerable from unanswerable tickets (median 0.609 vs 0.598).
Unanswerable tickets - feature requests, unclear requests, questions the docs do not cover - are on the
same topic as an article without being answered by it. Similarity measures topic, not answerability.

Consequence: the relevance threshold only filters clearly off-topic tickets. Whether a ticket can be
answered is decided by the other layers:
1. must-escalate intents (FR-05) via the classifier (B-06);
2. the classifier confidence threshold (B-07, still to be measured);
3. the answer step returning `answerable: false` (PR-02, FR-06);
4. the unsupported-claim guardrail (FR-09).
To record in the PRD revision log (Stage 5): FR-03's role narrowed to topic filtering; FR-02 carries more of the screening.

## Limits
- Measured on development data only; the validation set was not used to choose these values.
- `answerable_from_docs` labels taken as correct (PRD assumption).
- 4.2% of answerable tickets do not get the correct article in the top 3; to inspect during evaluation.
