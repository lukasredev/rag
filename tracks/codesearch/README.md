# Track `codesearch` — "Where is the code that…?"

You are building the search box of a code-search assistant. A developer types a natural-language
description ("Convert a UNIX-style path into a platform specific directory spec", or "function that
retries an HTTP request with exponential backoff") and your pipeline must return **the one Python
function** in a pile of 10,000 functions from 609 open-source GitHub projects that the description was
written for — as a ranked list, best first.

The core difficulty is the **vocabulary mismatch**: questions are English, the corpus is code. The words
of a description ("obtain the dataset as a python-local object") rarely appear in the code
(`def as_data_frame(self, use_pandas=True): ... pandas.read_csv(...)`), and identifiers like
`getUserName` are not English words. This track is retrieval-heavy: the LLM is there to *help retrieval*
(rewrite the query into code vocabulary, imagine what the code looks like, rerank candidates), not to write
a long answer.

## Data

Source: **CodeSearchNet** (Husain et al., 2019, *CodeSearchNet Challenge: Evaluating the State of Semantic
Code Search*, arXiv:1909.09436), Python subset, `test` split, from the Hugging Face parquet files of
`code-search-net/code_search_net`. **License:** every function comes from a GitHub repository that had a
license permitting redistribution when the dataset was built; each repository keeps its own license
(the dataset card lists no per-example license — see the `repo` field and the repository on GitHub).
Use the data for this workshop only.

`data/corpus.jsonl` — 10,000 functions, one JSON object per line:

```json
{"id": "csn-000123", "repo": "owner/project", "path": "pkg/module.py", "func_name": "Class.method",
 "code": "def method(self, x):\n    # comments are kept\n    ...", "url": "https://github.com/.../module.py#L10-L30"}
```

- The **docstring has been removed** from `code` (with Python's `ast`, falling back to `tokenize` for
  Python-2 code). Comments and docstrings of *nested* functions remain. The docstrings are what the
  questions are made of, so leaving them in would make this a trivial text-matching task.
- `func_name` contains the class for methods (`Class.method`); the `def` line in `code` does not.
- Exact duplicate functions were removed. Many near-duplicates (forks, copies, sibling methods) remain —
  that is realistic.

`data/test.jsonl` — 180 public questions **with** gold answers. The 180 hidden `eval` questions on the
server come from the same process (disjoint functions, same mix of types).

```json
{"id": "test-000", "question": "Updates the process stats with the information from the processes",
 "type": "semantic", "gold_id": "csn-008081", "gold_repo": "...", "gold_path": "...",
 "gold_func_name": "NextflowInspector._update_process_stats", "bm25_rank": 28}
```

Your pipeline only sees `id` and `question`.

## Question types

| type | n (test / eval) | how it was made |
|---|---|---|
| `lexical` | 93 / 94 | the function's own docstring (first paragraph, cleaned) — and a good **identifier-aware BM25** baseline ranks the gold function in its top 10 |
| `semantic` | 47 / 46 | same kind of docstring query, but that BM25 baseline does **not** find the function in its top 10: the words of the description are not the words of the code |
| `paraphrase` | 40 / 40 | written by Claude from the docstring + code, phrased like a developer searching an unfamiliar codebase ("where is the code that…"), with synonyms instead of the docstring wording and **no** identifiers, project or class names |

Docstring queries: first paragraph of the docstring; parameter/return/example sections, URLs, doctests
and reference markers are dropped; queries shorter than 6 words, trivial ones ("Constructor", "Initialize
…"), ones that literally contain the function name (also as split words, `get user name` for
`get_user_name`), and docstrings that occur more than once in the corpus are excluded. The labelling
BM25 indexes `func_name ×3 + path + code` with camelCase/snake_case splitting (see `prepare_data.py`;
`bm25_rank` in the question file is that rank). **The mix is stratified**: in the natural distribution
~76% of the docstring queries are `lexical`; here `semantic` is over-represented on purpose, so that you
can see where each retriever wins.

## Answer format and metric

```python
{"answer": "csn-000123",                      # your best guess (may just repeat citations[0])
 "citations": ["csn-000123", "csn-004711", ...]}   # RANKED function ids, best first
```

Only the **first 10 citations** are scored (duplicates are removed first). If `citations` is empty, the
`answer` id is used as a one-element ranking.

- **`mrr@10` (primary)** — reciprocal rank of the gold function: 1.0 if it is first, 0.5 if second,
  0.33 if third, …, 0.1 if tenth, 0 if it is not in the first 10. Averaged over questions. It rewards
  putting the right function *at the top*, not just somewhere in the list.
- **`hit@1`** — fraction of questions where the first citation is the gold function.
- **`hit@10`** — fraction where the gold function is anywhere in the first 10 (a recall measure: an
  upper bound for what a reranker on your top 10 could reach).

There is exactly one gold function per question. A near-duplicate of the gold function (same code in a
fork) does not count — rare, but it happens.

## Rules

- Your `Pipeline.__init__` gets `data_dir` (corpus + test.jsonl); build your index there. It must finish
  within ~5 minutes on CPU on the server (the embedding cache helps from the second submission on).
- `answer()` is called from 8 threads; do not mutate shared state in it.
- Use `ragkit` (LLM, embeddings, BM25). No internet other than the LLM API; no hard-coding of test answers.
- Do not use `test.jsonl` gold ids in your pipeline. Tuning on test is fine — that is what it is for — but
  expect the hidden eval to be lower (only 40–94 questions per type: ±0.05 is noise).

## How to run

```bash
# in the repo root, with the venv active and your .env set up (see START_HERE.md)
mkdir -p mine && cp tracks/codesearch/skeleton/pipeline.py mine/pipeline.py
python -m ragkit.run --track codesearch --pipeline mine/pipeline.py --limit 20            # quick check
python -m ragkit.run --track codesearch --pipeline mine/pipeline.py --out report.json     # whole test set
python scripts/failures.py report.json                                         # what went wrong
python -m studio --pipeline mine/pipeline.py                                   # every step, in the browser

# submit to the leaderboard (hidden questions); a folder is zipped, so helper modules are fine
python scripts/submit.py --track codesearch --pipeline mine/ --label bge-v1 \
    --note "bge-small instead of MiniLM; expect better ranking on paraphrased descriptions" --wait
```

Embedding 10,000 functions takes ~3 min with `all-MiniLM-L6-v2` and ~5–11 min with `bge-small-en-v1.5`
on a shared CPU; the result is cached, so only the first run pays. Use `--limit` while you iterate.

## Questions to think about

1. Look at 10 questions and their gold functions side by side. Which words of the question appear in the
   code? Which parts of a function carry the most meaning: name, arguments, path, body, comments?
2. What does `ragkit.text.tokenize` do to `getUserName`, `get_user_name`, `HTTPResponse`? What *should*
   happen to them?
3. Is a general-purpose sentence-embedding model any good at embedding code? What exactly do you embed —
   the whole function? Does the model see all of it (token limits)? Does it see the class name?
4. BM25 and dense retrieval fail on different questions. Is combining them always better? What does
   Reciprocal Rank Fusion assume about the two rankings?
5. Docstrings and paraphrase questions are written in different styles. Which retriever suffers more from the
   style change, and why?
6. An LLM knows what code for "retry with exponential backoff" usually looks like. How can you use that
   knowledge for *retrieval* (and not just for answering)?
7. An LLM can judge "does this code do X?" much better than a cosine similarity. How many candidates can
   you afford to show it per question, and what does that cost (tokens, seconds, money)?
8. The LLM may have *seen* these GitHub projects during training. How would you find out how much it
   knows without any retrieval — and is using that knowledge cheating?
