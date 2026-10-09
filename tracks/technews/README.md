# Track `technews` — Tech News Analyst (MultiHop-RAG)

## Scenario

You build the research assistant of a newsroom. Editors ask questions like

> *"Did the TechCrunch report on October 7, 2023 about Flexport and the later TechCrunch article from October 30
> describe the same kind of event?"*
> *"Which company is the antagonist in a legal case covered by The Verge and is also accused by news publishers
> in a class action lawsuit reported by TechCrunch?"*

Answering them needs **one to four different articles**, and the questions refer to those articles by their
**source** and **publication date**. Some questions ask about articles that are not in the archive at all —
then the assistant must say so instead of making something up.

## Data (`data/`)

**`articles.jsonl`** — 609 news articles (Sept 26 – Dec 25, 2023), one JSON object per line:

| field | example |
|---|---|
| `id` | `"news-0042"` (stable; use it for citations) |
| `title` | `"The FTX trial is bigger than Sam Bankman-Fried"` |
| `source` | `"The Verge"` (49 different sources: TechCrunch, Sporting News, The Verge, Polygon, Fortune, …) |
| `published_at` | `"2023-09-28T12:00:00+00:00"` (UTC) |
| `category` | `technology`, `sports`, `entertainment`, `business`, `science`, `health` |
| `author` | string or `null` |
| `url` | original URL |
| `body` | full article text (median ~1,300 words, max ~12k words; ~1.06M words in total) |

**`test.jsonl`** — 156 public questions **with** answers for development. Your pipeline only sees
`{"id", "question"}`; the other fields are gold:
`type`, `answer`, `answer_kind` (`entity` / `yesno` / `abstain`), `evidence_ids` (the articles the answer is
built from), `evidence_facts` (the sentences that matter in each evidence article).

The leaderboard uses 157 **hidden** questions from the same distribution (disjoint from test).

### Question types (40 per split each)

| type | what it asks | gold answer |
|---|---|---|
| `inference_query` | "Which person/company … according to *A* and *B*?" — combine facts from 2-4 articles | a name (`"Sam Altman"`) |
| `comparison_query` | "Does article *A* say X while article *B* says Y?" | `Yes` / `No` (balanced 50/50) |
| `temporal_query` | "Between the report of *A* on date d1 and the later report of *B*, did … change / stay consistent?" | `Yes` / `No` (balanced 50/50) |
| `null_query` | looks just like the others, but the needed articles/sources are **not in the archive** | `Insufficient information.` |

## Pipeline contract

```python
class Pipeline:
    def __init__(self, data_dir: pathlib.Path): ...   # load articles, build your index (<~5 min)
    def answer(self, question: dict) -> dict:          # called from 8 threads at once
        return {"answer": "Yes", "citations": ["news-0012", "news-0031"]}
```

Answer format: **`Yes`/`No`** for yes/no questions, **just the name** for who/which questions, and exactly
**`Insufficient information.`** to abstain (the abstention token; case and final period don't matter).

## Metrics

The leaderboard metric is **`correct`** (mean over questions):

* **null_query**: correct iff the answer abstains, i.e. the normalised answer (lowercase, no punctuation/articles)
  *starts with* `insufficient information`. Abstaining on any other question is always wrong.
* **Yes/No gold**: the *first word* of your answer must be the right `yes`/`no` (`"Yes, because …"` is fine,
  `"It seems yes"` is not).
* **Name gold**: correct if the normalised answer equals the gold, or contains it and is at most 12 tokens long
  (`"Sam Bankman-Fried (FTX founder)"` is fine). Otherwise an LLM judge (claude-haiku) compares the first 30
  words of your answer with the gold (handles `"SBF"`, `"Alphabet's Google"`). Listing several candidates is wrong.

Secondary metrics:

* **`exact`** — normalised exact match with the gold string (strict; shows how much the judge/containment rule matters).
* **`evidence_recall@5`** — fraction of the gold evidence articles among your first 5 *distinct* cited ids.
  For `null_query` there is no evidence: 1 if you cite nothing, else 0 (don't cite what you didn't use).
* **`evidence_all@5`** — 1 if *all* gold evidence articles are among your first 5 citations (same null rule).
  Multi-hop questions need all hops; this is much harder than recall.
* **`abstained`** — fraction of answers that abstain. Look at it per type: ideally 1.0 on null, 0.0 elsewhere.

Only the first 5 distinct citations count, so citing everything does not pay.

Background: the original paper (Tang & Yang 2024, arXiv:2401.15391) scores generation with plain accuracy on
these short answers (GPT-4 with retrieved chunks: 0.56; with the gold evidence: 0.89, null queries excluded)
and retrieval with Hits@K / MAP@K / MRR@K over *chunks* (best Hits@4 ≈ 0.66 with voyage-02 + a reranker). Our
`evidence_*@5` is at the article level.

## How to run

```bash
# in the repo root, with the venv active and your .env set up (see START_HERE.md)
mkdir -p mine && cp tracks/technews/skeleton/pipeline.py mine/pipeline.py
python -m ragkit.run --track technews --pipeline mine/pipeline.py --limit 20            # quick check
python -m ragkit.run --track technews --pipeline mine/pipeline.py --out report.json     # whole test set
python scripts/failures.py report.json                                         # what went wrong
python -m studio --pipeline mine/pipeline.py                                   # every step, in the browser

# submit to the leaderboard (hidden questions); a folder is zipped, so helper modules are fine
python scripts/submit.py --track technews --pipeline mine/ --label decompose-v1 \
    --note "split multi-hop questions into one search per article; expect comparison questions to improve" --wait
```

The skeleton (`skeleton/pipeline.py`) runs end to end (BM25 over 1000-character windows, top 3, a bare prompt)
and scores low. The TODOs in it are starting points.

Rules: use only `data/` (no web search, no other copies of the dataset). The LLM must answer from retrieved
articles; the model's own world knowledge is not the point (see "closed book" below).

## Questions to think about

1. Questions say "according to TechCrunch … published on October 7". Can your retriever *see* the source and
   the date? Should metadata be **text in the chunk**, a **filter**, or both?
2. One embedding for a question that mentions three articles — what does it actually point to?
   How many *different* articles end up in your context?
3. What does `evidence_all@5` tell you that `evidence_recall@5` doesn't? Which question type fails it most?
4. When should the assistant abstain? How do you tell "the archive doesn't have it" from "my retriever missed it"?
   What does a prompt that says "abstain if unsure" do to the *answerable* questions?
5. Yes/No questions are 50/50. How much better than a coin are you, per type? Is your model biased to "Yes"?
6. Is every extra LLM call worth it? Compare tokens/question with the gain per question type.
7. Run your LLM *without* retrieval. How much does it already know about late-2023 news?
