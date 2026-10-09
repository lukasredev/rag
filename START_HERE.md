# Start here

Today you build a **RAG system**: a program that answers questions by first *finding* the right pieces of a
document collection and then letting an LLM answer from them. You work alone or in a team on one **track** (one
dataset and one kind of question). You get a starting pipeline that works but scores badly. Your job is to make it
better, one measured change at a time.

Your score comes from **hidden questions** on the evaluation server, shown on a leaderboard. But the score is
not the only thing that counts: every submission needs a hypothesis, and the best experiments and the best
error analysis win too, including experiments that didn't work.

**Get the code:** [github.com/lukasredev/rag](https://github.com/lukasredev/rag) — clone it as shown in step 1.

All commands below are for the `papers` track. Replace `papers` with your track everywhere.

## 1. Set up (about 15 minutes; do it before the workshop if you can)

You need [git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/);
uv fetches Python 3.12 itself. Get the workshop repository (no git? "Code → Download ZIP" on the GitHub page):

```bash
git clone https://github.com/lukasredev/rag.git rag-workshop
cd rag-workshop
uv venv --python 3.12
source .venv/bin/activate                                   # Windows: .venv\Scripts\activate
uv pip install --torch-backend cpu -e ".[local,studio]"     # downloads about 1 GB
```

Every new terminal needs `source .venv/bin/activate` again before the commands below work.

Create a file called `.env` in the `rag-workshop` folder with what the organisers gave you. If you work alone,
the "team" is just you: one name and one token, the same as for a team.

```
ANTHROPIC_API_KEY=sk-ant-...
RAG_TEAM=your-team-name          # or your own name if you work alone
RAG_TOKEN=your-token
RAG_SERVER=https://viscon-eval.lukasre.ch
```

Keep the key and token to yourself (or your team). In a team, everyone uses the same name and token, so
all your submissions count for the team. `.env` is in `.gitignore`, so git won't pick it up.

Check that everything works with 5 questions:

```bash
python -m ragkit.run --track papers --pipeline tracks/papers/skeleton/pipeline.py --limit 5
```

It ends with a small table like this (your numbers may differ):

```
=== papers: 5 questions ===
overall: abstained=0.000  answer_f1=0.266  correct=0.600  evidence_recall@5=0.500

type                        n     correct
abstractive                 1       1.000
...
llm usage: {'calls': 5, 'cached_calls': 0, 'input_tokens': 3980, 'output_tokens': 1220}
```

This takes a few seconds: all starting pipelines use keyword search. Once you add embeddings, the first run
embeds the whole corpus, which takes 3 to 10 minutes on a laptop; after that it is cached.

## 2. Read your track (15 minutes)

Open `tracks/papers/README.md`. It explains the scenario, what the data looks like, the types of questions,
how answers are scored and what your pipeline must return. Then look at the data yourself: open the corpus and
a few questions in `tracks/papers/data/test.jsonl`. These test questions are public, with the expected answers.

| track | your pipeline gets | it returns | starting pipeline scores about |
|---|---|---|---|
| `papers` | a question about one research paper | a short answer or `UNANSWERABLE`, paragraph ids | 0.20 |
| `technews` | a question about 2023 tech news | a short answer, article ids | 0.30 |
| `rfc` | a question about internet protocols | an answer or `UNANSWERABLE`, RFC section ids | 0.82 |
| `shell` | "how do I …" in plain English | one bash command, man-page paragraph ids | 0.00 (fix the first TODO) |
| `codesearch` | a description of a Python function | a ranking of function ids | 0.28 (MRR) |

Our own best pipelines reach 0.80 to 0.95 on every track, with the same small model you have. So there is a
lot of room, and getting there is about retrieval and prompting, not about a bigger model.

## 3. Your first real run (30 minutes)

Copy the starting pipeline into a folder of your own, and run it on all test questions:

```bash
mkdir -p mine && cp tracks/papers/skeleton/pipeline.py mine/pipeline.py
python -m ragkit.run --track papers --pipeline mine/pipeline.py --out report-v1.json
```

For `papers` this takes about a minute and costs about $0.04. It prints the score per question type, and
`report-v1.json` keeps every question with your answer and its scores.

Now **look at what went wrong** before you change anything:

```bash
python scripts/failures.py report-v1.json
```

For each wrong answer it shows the question, your answer, the expected answer, what you cited and where the
evidence actually is:

```
--- extractive  1bdc990c7e948724ab04e70867675a334fdd3051
Q:        Where do they get the recipes from?
answer:   Based on the provided context, the paper does not explicitly say where the recipes were sourced from...
expected: ['from Food.com']
cited:    1909.00105:4:7, 1909.00105:5:0, 1909.00105:4:2
evidence: 1909.00105:3:0  (0 of 1 among your first 5 citations)
```

Here the right paragraph (`3:0`) was never retrieved, so the model couldn't answer. That's a retrieval
problem, not a prompting problem.

To see every step for one question, start RAG Studio:

```bash
python -m studio --pipeline mine/pipeline.py      # then open http://127.0.0.1:8765
```

At the top, choose your track and then `mine/pipeline.py` as the pipeline (it is listed under every track).
**One question** shows what was searched, what was found (the right evidence is marked ✓), the exact prompt
the LLM got and what it answered. Studio keeps running until you press Ctrl+C, so use a second terminal for
everything else (and activate the venv there too).

For 10 to 20 wrong answers, write down *why* each one failed. Usually it is one of these:

- **retrieval:** the right passage was not found (no ✓ among your citations)
- **chunking:** it was found, but cut so the important part is missing
- **generation:** the right passage was there, but the answer is wrong, too long, or in the wrong format
- **the question or the scoring:** the expected answer looks wrong to you (tell us!)

Count them. The most common reason is where your first change should go.

## 4. The loop (the rest of the day)

1. **Hypothesis.** For example: "The starting pipeline searches all 200 papers, but each question says which
   paper it is about. If we search only that paper, we should find the evidence much more often."
2. **Change** one thing in `mine/pipeline.py`. The `TODO`s in the starting pipeline are good first ideas.
3. **Measure** on the same questions as before. While you work, `--limit 30` is quick (it always takes the
   first 30 questions, so runs are comparable). Before you decide, run the whole test set again with a new
   report name:

   ```bash
   python -m ragkit.run --track papers --pipeline mine/pipeline.py --limit 30          # quick check
   python -m ragkit.run --track papers --pipeline mine/pipeline.py --out report-v2.json
   ```

   Compare the table per question type with the previous run, not only the overall number. A change can help
   one type and hurt another. (The example above took `correct` from 0.20 to 0.40, and `unanswerable` stayed
   at 0: the next thing to fix.)
4. **Submit** when it is better:

   ```bash
   python scripts/submit.py --track papers --pipeline mine/ --label per-paper-bm25 \
       --note "Search only inside the paper the question names; expect evidence recall and correct to roughly double" --wait
   ```

   The server runs your pipeline on the hidden questions. With `--wait` the result appears in your terminal
   after a minute or two (longer if other teams are in the queue, or if your pipeline embeds a lot):

   ```
   submitted #38: https://viscon-eval.lukasre.ch/submission/38
   status: done
   overall: abstained=0.000  answer_f1=0.210  correct=0.520  evidence_recall@5=0.567
     abstractive            n=37   correct=0.595
     ...
   ```

   Your score on the hidden questions will differ from your test score, often by up to 0.1 either way
   (here 0.52 against 0.40). The sets are small, so differences below about 0.05 between two submissions are
   noise. The leaderboard is at `https://viscon-eval.lukasre.ch`.

Keep every version that worked (`cp -r mine mine-v2`, or commit it with git: the folder is a git repository).
You will want to go back.

## Rules

- **Model:** only `claude-haiku-5-5`, through `ragkit.llm.chat` (the default). Other models are refused.
- **Embeddings and rerankers:** only the 11 models listed under
  [Models available on the server](README.md#models-available-on-the-server) in the README (6 embedding models,
  5 rerankers, with links to Hugging Face). The server has no internet, so it can't load any other model.
- **On the server, your pipeline has no internet access.** It can read only its track's `data/` folder.
- **Submissions:** at most 100 per hour per track, each with a `--note` of at least 20 characters saying what you
  changed and what you expect. A pipeline is one `pipeline.py` or a folder with `pipeline.py` and helper files
  (`.py`, `.json`, `.txt`, `.yaml`, `.md`); keep big files such as reports out of that folder.
- **Budget:** every team, and everyone working alone, has the same fixed LLM budget on the server. Answers are cached, so re-running an unchanged
  pipeline costs nothing. `python -m ragkit.spend` shows what you've spent locally.
- **Fair play:** don't hard-code answers, don't use the expected answers at answer time, and don't use other
  copies of the dataset.

## When something goes wrong

- **`No module named 'ragkit'`:** the venv isn't active in this terminal; run `source .venv/bin/activate`
  in the `rag-workshop` folder.
- **Your submission failed:** the line `why:` says what happened. If your pipeline raised exceptions, the
  server shows only their type, because the hidden questions must stay hidden. Run the same pipeline locally on
  the test set to see the full error. If it says the problem is on the server's side, tell the organisers.
- **"model … is not allowed":** you asked for a model other than Haiku; remove the `model=` argument.
- **"spend cap" or "budget is used up":** tell the organisers.
- **Everything is slow:** use `--limit 30` while you work. A new embedding model embeds the whole corpus once,
  which takes minutes; after that it is cached.

More detail on `ragkit` (the helper library), RAG Studio and the metrics is in [README.md](README.md).
