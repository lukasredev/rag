# Track `papers` — Ask the Paper (Qasper)

**Scenario.** You are building a study assistant for an NLP course. A student is reading a research paper
and asks questions about it — *"Which datasets do they evaluate on?"*, *"Do they compare against BERT?"*,
*"How was the corpus annotated?"* — and the assistant must answer **from that paper**, point to the
paragraphs it used, and say `UNANSWERABLE` when the paper simply does not say.

Every question tells you which paper it is about (`paper_id`, `title`). So this is not "find the right
document among thousands" — it is **"find the right 2 % of one long document"**: how you cut a paper into
pieces, what context you keep around each piece, and how you prompt for the answer matter much more than
which vector database you use.

The questions and answers come from **Qasper** (Dasigi et al., NAACL 2021): NLP practitioners wrote the
questions after reading only the title and abstract; *other* annotators then read the full paper, answered,
and marked the evidence paragraphs. License: **CC BY 4.0**, https://allenai.org/data/qasper.

## Data (`data/`)

| file | what |
|---|---|
| `papers.jsonl` | 200 NLP papers (arXiv 2016–2020) — 9,883 paragraphs, 1,404 figure/table captions. Median paper ≈ 3,300 words (≈ 5k tokens), longest 18k words |
| `test.jsonl` | 150 public questions **with** gold answers and gold evidence — develop on these |

The hidden evaluation set (150 questions) is about **different papers** than `test.jsonl` (the papers are in
`papers.jsonl` too — 58 of the 200 papers are distractors with no questions).

One paper per line:

```json
{"paper_id": "1912.01214", "title": "Cross-lingual Pre-training Based Transfer ...", "abstract": "...",
 "sections": [{"idx": 0, "name": "Introduction",
               "paragraphs": [{"id": "1912.01214:0:0", "text": "Although Neural Machine Translation ..."}, ...]},
              {"idx": 4, "name": "Approach ::: Bridge Language Model Pretraining", "paragraphs": [...]}, ...],
 "captions": [{"id": "1912.01214:cap:2", "text": "Table 1: Data Statistics."}, ...]}
```

* Paragraph ids are `<paper_id>:<section_idx>:<paragraph_idx>`; caption ids are `<paper_id>:cap:<k>`.
  These ids are what you cite.
* Section names are hierarchical: `"Experiments ::: Setup ::: Datasets."` is a subsection of a subsection.
* The text comes from the LaTeX source: citations are `BIBREF12`, tables `TABREF5`, figures `FIGREF3`, math
  is `INLINEFORM0` / `EQREF7`. **Tables and figures themselves are not available** — only their captions.

Question record (your pipeline only sees `id`, `question`, `paper_id`, `title`):

```json
{"id": "bdf930...", "question": "What was their system's performance?", "paper_id": "1706.01875",
 "title": "Measuring Offensive Speech in Online Political Discourse",
 "type": "extractive", "answers": ["accuracy and F1-score of 89.6% and 89.2%, respectively"],
 "evidence": [["1706.01875:4:2"], ["1706.01875:4:2"]], "evidence_in_caption": false, "n_annotators": 2}
```

* `answers`: one reference per annotator (duplicates removed). Many questions have 2–3 annotators who
  phrase the answer differently — any of them counts.
* `evidence`: one list of paragraph/caption ids **per annotator**.
* `yes_no` (`"yes"`/`"no"`) on boolean questions; `evidence_in_caption` = some annotator pointed at a table
  or figure (≈ 18 % of the questions — often the actual number is in the table, which you do not have!).

### Question types

| type | test | eval | what it is | what a good answer looks like |
|---|---:|---:|---|---|
| `extractive` | 71 | 71 | the answer is a span (or list of spans) in the paper | `"BLEU, METEOR"`, `"220 human-human dialogs"` |
| `abstractive` | 36 | 37 | the annotators wrote a free-form answer | one short sentence |
| `boolean` | 25 | 26 | yes/no question | `"Yes"` / `"No"` |
| `unanswerable` | 18 | 16 | all annotators agree the paper does not answer it | exactly `UNANSWERABLE` |

The type is the annotators' majority answer type (your pipeline does not get it). Questions where annotators
disagreed on *whether* the paper answers the question, or on yes vs. no, were dropped.

## Pipeline contract

```python
class Pipeline:
    def __init__(self, data_dir: pathlib.Path): ...           # build your index (<= ~5 min on CPU)
    def answer(self, question: dict) -> dict:                  # called from 8 threads at once
        return {"answer": "Yes", "citations": ["1912.01214:4:2", "1912.01214:cap:3"]}
```

* **Abstention:** if the paper does not answer the question, the answer must be exactly `UNANSWERABLE`
  (case-insensitive; surrounding whitespace/quotes and a trailing period are ignored).
  `"UNANSWERABLE because the paper ..."` is **not** an abstention.
* **Yes/no:** only the first word counts — `"Yes, they do ..."` is read as *yes*.
* **Citations:** paragraph or caption ids. **Only the first 5 count**, so order them by confidence.

## Metrics (`track.py`)

| metric | definition |
|---|---|
| **`correct`** (primary) | `unanswerable`: 1 iff the answer is exactly the abstention token. `boolean`: 1 iff the answer starts with the right yes/no. `extractive` / `abstractive`: 0 for an abstention; 1 if the token F1 against some reference is ≥ 0.8; otherwise an **LLM judge** (`ragkit.metrics.llm_judge`, claude-haiku) decides whether the answer agrees in substance with one of the references |
| `answer_f1` | the official-style Qasper metric: max token F1 over the references (SQuAD normalisation). Boolean → 1/0 on yes/no, unanswerable → 1/0 on abstention. Rewards *short, reference-like* answers; a correct but chatty answer gets a low F1 |
| `evidence_recall@5` | of one annotator's evidence paragraphs, which fraction is among your first 5 citations — best over annotators. Unanswerable questions have no evidence → 1.0 (ragkit convention), so **read the per-type rows** |
| `abstained` | 1 if the answer is `UNANSWERABLE`. On the `unanswerable` row you want 1.0; on every other row it is the false-abstention rate |

The overall row is the mean over all questions. The leaderboard shows `correct` per type.

## Rules

* Use `ragkit.llm.chat` (default model) for generation. Any retrieval/reranking you like, with the
  embedding models and rerankers listed in the main README ("Models available on the server"); the sandbox is
  offline except for the LLM.
* Do not hard-code answers or use gold fields at answer time.
* Every submission needs a `--note` with your hypothesis: what did you change, what do you expect?

## How to run

```bash
# in the repo root, with the venv active and your .env set up (see START_HERE.md)
mkdir -p mine && cp tracks/papers/skeleton/pipeline.py mine/pipeline.py
python -m ragkit.run --track papers --pipeline mine/pipeline.py --limit 20            # quick check
python -m ragkit.run --track papers --pipeline mine/pipeline.py --out report.json     # whole test set
python scripts/failures.py report.json                                         # what went wrong
python -m studio --pipeline mine/pipeline.py                                   # every step, in the browser

# submit to the leaderboard (hidden questions); a folder is zipped, so helper modules are fine
python scripts/submit.py --track papers --pipeline mine/ --label sections-v1 \
    --note "paragraph chunks with section titles; expect higher evidence recall on 'which dataset' questions" --wait
```

`--out` writes every question with your prediction and scores. Read your failures, not just the mean.

## Questions to think about

1. Open one paper in `papers.jsonl`. What is a natural unit of retrieval? What does a paragraph lose when you
   cut it out of its section ("Experiments ::: Datasets")? What does a 1000-character window lose?
2. The question tells you the paper. How many candidate pieces does that leave? Does retrieving from 200
   papers instead of 1 ever make sense?
3. "What datasets do they use?" — where in a paper is that answered? Is it always in one paragraph?
4. ~18 % of the evidence points at a table or figure. What do you actually have of them? What should the
   model say when the number it needs is in a table you cannot see?
5. When should the model say `UNANSWERABLE`? Look at the `abstained` column on the *answerable* rows — what
   does an over-cautious prompt cost you?
6. Compare `answer_f1` and `correct`. Why can an answer be judged correct but have a low F1, and what does
   that tell you about the answer style the references have?
7. A whole paper is ~5k tokens. Why not just give the model the whole paper? Try it — and compare the cost.
8. The model may have read these papers during pre-training. How would you find out how much of your score
   comes from its memory rather than from your retrieval?
9. Is your retriever or your generator the bottleneck? (Compare `evidence_recall@5` with `correct` per type.)
