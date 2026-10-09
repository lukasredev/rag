# RAG Workshop Kit

> **New here? Read [START_HERE.md](START_HERE.md) first.** This page is the reference. The repo contains the
> corpora and the public test sets; the hidden evaluation questions live only on the server.

Five self-contained RAG challenges for CS undergraduates. Each one has a public dataset, a public test set with
answers, a hidden evaluation set, a starter skeleton, and a
shared evaluation server with leaderboards.

Students write one Python class (`Pipeline`), check their score locally, and submit it to the server, which runs it
in a sandbox on the hidden questions. Every submission must include a **hypothesis** ("what did I change, what do I
expect?"), and scores are broken down **by question type**. So the competition rewards experiments and error
analysis, not pasting together the biggest pipeline.

## The five tracks

| track | dataset (public) | the pipeline gets | must return | what it teaches |
|---|---|---|---|---|
| [`papers`](tracks/papers/README.md) | **Qasper**: 200 NLP research papers (CC BY 4.0) | question + paper id | short answer or `UNANSWERABLE`, paragraph ids | chunking long structured documents, captions/tables, abstaining, "do we even need RAG?" (long-context baseline) |
| [`technews`](tracks/technews/README.md) | **MultiHop-RAG**: 609 tech/business news articles, 2023 (ODC-BY) | question | short answer / Yes-No / "Insufficient information", article ids | multi-hop questions, metadata (source, date) as filters, query decomposition, temporal reasoning |
| [`rfc`](tracks/rfc/README.md) | **IETF RFCs**: 44 raw protocol specs with version chains (IETF Trust) | question | answer or `UNANSWERABLE`, RFC (section) ids | parsing messy raw text, section-aware chunking, BM25 for exact terms, superseded documents, *building your own eval set* |
| [`shell`](tracks/shell/README.md) | **tldr / DocPrompting**: man pages of 337 commands (MIT / CC BY 4.0) | natural-language request | one bash command, man-page paragraph ids | RAG for code generation, two-stage retrieval (command, then flags), few-shot retrieval, when the model already knows |
| [`codesearch`](tracks/codesearch/README.md) | **CodeSearchNet (Python)**: 10k functions, docstrings stripped | description of a function | ranked function ids | vocabulary mismatch NL↔code, embedding model choice, hybrid fusion pitfalls, HyDE, LLM reranking |

Each track folder has a `README.md` (scenario, data format, metric, questions to think about), `track.py`
(the metric), `skeleton/` (your starting point) and `data/` (the corpus and the public test set).

## Repository layout

```
ragkit/            small, readable helper library students use
  llm.py           chat()/chat_json() for Claude or OpenAI models, disk-cached, token accounting
  embed.py         embed() with on-disk cache (local sentence-transformers or OpenAI)
  bm25.py          BM25 in ~50 lines
  text.py          tokenize, chunk_chars, chunk_words, split_paragraphs
  metrics.py       EM, token F1, recall@k, MRR, llm_judge
  run.py           local runner: identical scoring to the server
studio/            RAG Studio: a local web app to see what a pipeline does (below)
scripts/submit.py  submit a pipeline to the evaluation server
scripts/failures.py  print the questions a run got wrong, next to the expected answers
START_HERE.md      the students' guide: setup, first hour, the loop, rules
tracks/<name>/     the five tracks (see above)
tests/             unit tests for ragkit
```

## For students

The step-by-step guide is [START_HERE.md](START_HERE.md); this section is the short version.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --torch-backend cpu -e ".[local,studio]"   # ragkit + sentence-transformers (CPU torch: ~1 GB, not ~4)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env                 # the key you got for the workshop

# develop on the public test set (fast iterations with --limit)
python -m ragkit.run --track technews --pipeline tracks/technews/skeleton/pipeline.py --limit 20
python -m ragkit.run --track technews --pipeline my_pipeline.py --out report.json   # per-question details

# submit to the hidden evaluation set (max 100 per hour per track)
python scripts/submit.py --server http://<server>:8000 --team <team> --token <token> \
    --track technews --pipeline my_pipeline.py --label hybrid-v1 \
    --note "Added source/date headers to chunks; expect temporal questions to improve" --wait
```

Your pipeline is one file (or a folder with `pipeline.py` + helpers):

```python
class Pipeline:
    def __init__(self, data_dir): ...            # load the corpus, build the index
    def answer(self, question: dict) -> dict:    # called from 8 threads at once
        return {"answer": "...", "citations": ["doc-id", ...]}
```

On the server, your pipeline runs **without internet access**: call the LLM through `ragkit.llm.chat` (the server
passes it through, with `claude-haiku-5-5` only and a budget per team), and use local embedding models
(`sentence-transformers`), not API embeddings.

**Only `claude-haiku-5-5` is allowed**, locally and on the server: `chat(..., model="claude-opus-5-5")` raises
`ModelNotAllowedError`. Haiku is cheap enough that a full test run costs cents, and good retrieval matters more
than a bigger model (see the per-track READMEs).

### Spend cap

Every LLM call through `ragkit` is priced and counted on your machine. Once the total would go over
`RAG_MAX_USD` (default **$5**), `ragkit` refuses further calls and runs stop with a clear message; cached
answers are free. `python -m ragkit.spend` shows what you have spent; set `RAG_MAX_USD=20` (or `off`) in `.env`
to change the cap. Still give every key a spend limit in the provider console: that is the only hard limit.

### RAG Studio: see what your pipeline does

```bash
uv pip install -e ".[local,studio]"
python -m studio                                  # open http://127.0.0.1:8765
python -m studio --pipeline my_pipeline.py        # list your own pipeline too (or add it in the UI)
```

A local web app with five views:

1. **Pipeline:** a diagram of the RAG steps that lights up with the steps your pipeline actually runs, the build
   time and its source code.
2. **Corpus:** what is in the data: number and length of documents (histogram), facets such as source,
   category, month, RFC status or repo to filter by, full-text search, and a reader for every document that lists
   the test questions needing it. Every document id in a trace is clickable and opens here, scrolled to the cited
   paragraph or RFC section. A second tab, *Test questions & their evidence*, shows how questions link to documents: a link map of all
   test questions and the documents they need (multi-hop questions have several lines, hub documents many), and per
   question the expected answer, the evidence passages or quotes, and a graph of the documents it needs and the
   other questions that need the same documents.
3. **One question:** pick a test question (or type your own) and see every step: BM25 query terms with their idf,
   the query embedding, the retrieved chunks (gold evidence marked ✓), the exact prompt sent to the LLM and its
   response, the score and the cost.
4. **Test run:** run the pipeline on N test questions (by type), with a cost estimate first and a live progress
   bar; then scores per type and every question with its full trace.
5. **Compare runs:** your runs next to the reference results, per question type, plus which questions one run
   fixed and the other broke.

The pipeline's index is built once and rebuilt automatically when you edit its code. `ragkit` records LLM calls,
embeddings and BM25 searches by itself; add your own steps with `trace.log("retrieve", hits=[{"id", "score", "text"}])`
(the skeletons show how). Studio runs are stored in `.studio/runs/`. It only uses the public test set.

Use `ragkit` or anything else installed in the environment (the server has the same packages).

### Models available on the server

The server runs your pipeline offline, so it can only load these models (all from Hugging Face). Locally,
`sentence-transformers` downloads them the first time you use one.

| model | kind | size | notes |
|---|---|---:|---|
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | embedding | 90 MB | the default in `ragkit.embed`: small and fast |
| [`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) | embedding | 90 MB | same size, but trained on question → answer pairs |
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | embedding | 130 MB | strong for its size; for queries, prefix `Represent this sentence for searching relevant passages: ` |
| [`intfloat/e5-small-v2`](https://huggingface.co/intfloat/e5-small-v2) | embedding | 130 MB | needs `query: ` before questions and `passage: ` before documents |
| [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | embedding | 440 MB | bigger bge-small: better, but several times slower to embed a corpus on CPU |
| [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | embedding | 440 MB | strong general-purpose model, also several times slower than MiniLM |
| [`cross-encoder/ms-marco-TinyBERT-L-2-v2`](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2-v2) | reranker | 18 MB | very fast, rough |
| [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | reranker | 90 MB | good speed/quality balance |
| [`cross-encoder/ms-marco-MiniLM-L-12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) | reranker | 130 MB | a bit better, twice as slow |
| [`mixedbread-ai/mxbai-rerank-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | reranker | 150 MB | |
| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | reranker | 1.1 GB | the strongest here, and the slowest |

Embedding models: `embed(texts, model="local:BAAI/bge-small-en-v1.5")`. Rerankers:
`sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict([(query, passage), ...])`.
The first run with a new embedding model embeds the whole corpus (minutes on CPU, on the server too); after that
it comes from the cache.
