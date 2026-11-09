# Decision: chunking strategy for the documentation index (B-03)

Status: **pending measurement** (fill in after `scripts/compare_chunking.py` on development tickets)
Serves: FR-03, acceptance criterion A4. Setup Guide §04: "try at least two configurations, measure, record".

## Facts about the corpus
- 29 articles, 10 categories, 138–214 words each (≈177 average).
- Every article has the same four sections: Symptoms, Common causes, Resolution, Notes.
- Longest articles (~280–300 tokens) exceed all-MiniLM-L6-v2's 256-token input, so whole-article
  embeddings lose the end (usually Notes).
- Dataset Guide warns that splitting inside a resolution sequence gives passages that "retrieve well
  but read as incomplete".

## Options compared
| Option | Chunks | Description |
|---|---|---|
| section | 116 | One chunk per section, prefixed with title + applies-to; retrieval groups hits by `doc_id` and returns the full article ("small-to-big") |
| whole | 29 | One chunk per article |
| fixed | 58 | 800 characters, 120 overlap (Setup Guide baseline) |

All use all-MiniLM-L6-v2, cosine similarity, Chroma. Citations are article-level `doc_id`s in every case.

## Results (development tickets, answerable = 357, unanswerable = 143)
| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR | Median best score answerable | Median best score unanswerable |
|---|---|---|---|---|---|---|
| section | | | | | | |
| whole | | | | | | |
| fixed | | | | | | |

Run date: ____ · results file: `evaluation/results/chunking_comparison_____.md`

## Decision
Chosen: ____ because ____.
Relevance threshold (B-04): ____, chosen from the threshold table because ____ (trade-off: answerable
tickets that lose their article vs unanswerable tickets that wrongly return something).
