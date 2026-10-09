# Track `shell` — Shell Command Helper

> "How do I list only the hosts that are alive in 192.168.1.1–254, pinging them all in parallel?"
> → `fping -a -g 192.168.1.1 192.168.1.254`

You build the retrieval-augmented brain of a **shell command helper**: it gets a natural-language request
and must answer with **one bash command**. The servers it runs on have exactly the tools whose man pages
are in `data/docs.jsonl` installed — and many of them are not the tools you (or the LLM) would reach for first.
Retrieval is *code-generation RAG*: you look up documentation (which command? which options?) and example
code, not facts.

The data comes from the tldr benchmark of **DocPrompting** (Zhou et al., *DocPrompting: Generating Code by
Retrieving the Docs*, ICLR 2023). As in the paper, every command in the questions is **unseen** in the
training examples you get; many of them are niche (`fping`, `timidity`, `sacct`, `pdftocairo`, `debsecan`, ...).

## Data (`data/`)

| file | what | size |
|---|---|---|
| `docs.jsonl` | man-page paragraphs `{"id": "fping_12", "cmd": "fping", "text": "-a, --alive  Show systems that are alive."}` | 27,000 paragraphs, 337 commands |
| `commands.jsonl` | one line per command: `{"cmd", "n_paragraphs", "summary"}` (summary = the NAME line of the man page, e.g. "fping - send ICMP ECHO_REQUEST packets to network hosts") | 337 |
| `train_examples.jsonl` | 4,000 tldr pairs `{"nl", "cmd", "cmd_name", "has_docs"}` for **other** commands (training split of DocPrompting). `nl` is the original terse, lower-cased tldr caption. `has_docs`: whether that command's man page is in `docs.jsonl` | 4,000 |
| `test.jsonl` | 119 public questions **with** gold answers — develop on these | 119 |

Paragraph ids are the dataset's own ids `<cmd>_<n>` (n = position in the man page; `_0` is the header line,
`_1` usually the NAME line, `_2` usually the SYNOPSIS). Paragraphs are short (≈18 words on average): one option
description each, roughly. The corpus contains the man pages of all question commands (test **and** hidden eval)
plus ~100 distractor commands, so you cannot guess the answer command from "what's in the corpus".

A question record:

```json
{"id": "tldr-6950", "question": "I want to ping every address from 192.168.1.1 through 192.168.1.254 in one go and get back only the hosts that are alive. How can I do that?",
 "type": "rare_flags", "answer": "fping -a -g 192.168.1.1 192.168.1.254", "cmd_name": "fping",
 "oracle_man": ["fping_3", "fping_6", "fping_7", "..."], "oracle_text": ["..."], "tldr_nl": "list alive hosts within a subnet generated from a netmask"}
```

Your pipeline only sees `id` and `question`. Everything else is gold.

**Where the questions come from.** tldr captions are written *under a page that names the command* ("close the
current connection", "list all ports"), so on their own they are hopelessly ambiguous. We rewrote each caption
with an LLM into a self-contained request that describes the task and the kind of tool **without naming the
program** (ecosystem names like git, Docker, GitHub CLI, Kubernetes, OpenSSL may appear). The original caption
is kept as `tldr_nl`. A few rewrites are imperfect — that is real life.

### Question types (leaderboard shows the primary metric per type)

| type | meaning | test | eval |
|---|---|---|---|
| `common_simple` | well-known command (`find`, `scp`, `journalctl`, `git tag`, `docker`, ...), ≤ 1 option | 32 | 33 |
| `common_flags` | well-known command, ≥ 2 options or a pipe | 33 | 29 |
| `rare_simple` | niche command (`fping`, `timidity`, `sacct`, `xetex`, ...), ≤ 1 option | 25 | 25 |
| `rare_flags` | niche command, ≥ 2 options or a pipe | 29 | 32 |

"Common" is a hand-curated list (in `prepare_data.py`) — a judgement call, not a law of nature. Test and eval use
**disjoint sets of commands** (117 vs 116 commands), so tuning on the test commands does not transfer for free.

## Answer format and placeholders

Return `{"answer": "<one bash command line>", "citations": ["<doc id>", ...]}`.

- The scorer takes the **first non-empty line** of `answer` (a surrounding ``` fence and a leading `$ ` are
  stripped, nothing else). Explanations or alternatives after the first line are ignored — but if your first line
  is prose, it is wrong.
- Values the user must fill in are written as tldr-style placeholders in double curly braces:
  `scp {{path/to/local_file}} {{user}}@{{host}}:{{path/to/dir}}`. For scoring, **every `{{...}}` is replaced by
  the same token `<ph>`** in gold and prediction, so the *name* inside the braces never matters
  (`{{file}}` = `{{path/to/file.tgz}}`). Values the request states literally (port 22, "50 MB", `eth0`) can be
  written literally; in tldr they are sometimes placeholders (`{{22}}`) — exact match then fails, but the judge
  (below) accepts it.
- Cite the paragraph ids you used (first 5 count).

## Metrics

| metric | definition |
|---|---|
| **`correct`** (PRIMARY) | `exact` OR (`installed` AND LLM judge says the prediction does the same job as the reference) |
| `exact` | the normalised token sequences are identical (placeholders → `<ph>`, quotes removed, leading `sudo`/`env VAR=..` dropped) |
| `installed` | the main (first) command of the prediction has a man page in `docs.jsonl` (`docker start` counts as `docker-start` or `docker`) |
| `cmd_acc` | the main command equals the gold command — DocPrompting's *command name accuracy* |
| `token_f1` | token-level F1 of normalised prediction vs. gold — DocPrompting's main metric |
| `doc_recall@5` | fraction of the gold man-page paragraphs (`oracle_man`) among your first 5 citations |
| `cmd_recall@5` | 1 if any of your first 5 citations is a paragraph of the gold command's man page |

The **judge** (claude-haiku-5-5) sees the request, the reference command, the reference command's gold man-page
paragraphs and your command. It accepts short/long option spellings, different option order, harmless extras,
concrete values instead of placeholders — and also a *different installed tool* if it clearly achieves the same
result (`netstat -rn` for "routing table, numeric" when the reference uses `netstat --route --numeric`).
It rejects wrong/missing options, wrong subcommands and several alternatives in one answer.

Why not just exact match? Shell commands have many correct spellings (`-r` = `--recursive`, option order,
`-o x` vs `-o=x`); on our test set fewer than half of the judged-correct answers are exact matches.
Why the `installed` gate? The scenario is "help me on *this* machine": `stress-ng --io 2` is a fine answer in
general, but here only `stress` is installed.

**Weaknesses (be aware of them):** the judge is an LLM and can be fooled by plausible-looking options for rare
commands (it gets the gold man-page paragraphs to reduce this); `installed` only checks the first command of a
pipeline; tldr references are sometimes not the only or best solution; and some man pages in the corpus are old or
even describe a different program of the same name (e.g. `stack`, `gh`) — the tldr answer then uses options the
man page doesn't mention. `token_f1` rewards copying the placeholder structure more than correctness — use it for
diagnosis only.

## Rules

- Use the default LLM through `ragkit.llm.chat` (you may make several calls per question; we report tokens).
- No access to the internet, the original dataset or tldr-pages at answer time; the sandbox only sees `data/`.
- Don't hard-code answers or command names for specific test questions. Hand-written general knowledge
  (e.g. a list of shell builtins) is fine.
- Index build ≤ ~5 min on CPU (the embedding cache helps), answering 119 questions ≤ ~10 min.

## How to run

```bash
# in the repo root, with the venv active and your .env set up (see START_HERE.md)
mkdir -p mine && cp tracks/shell/skeleton/pipeline.py mine/pipeline.py
python -m ragkit.run --track shell --pipeline mine/pipeline.py --limit 20            # quick check
python -m ragkit.run --track shell --pipeline mine/pipeline.py --out report.json     # whole test set
python scripts/failures.py report.json                                         # what went wrong
python -m studio --pipeline mine/pipeline.py                                   # every step, in the browser

# submit to the leaderboard (hidden questions); a folder is zipped, so helper modules are fine
python scripts/submit.py --track shell --pipeline mine/ --label format-v1 \
    --note "system prompt asking for one command only; expect the skeleton's 0 to jump" --wait
```

Start from `skeleton/pipeline.py` (BM25 over paragraphs, top 3, minimal prompt; runs, but scores low).

## Questions to think about

1. Run a closed-book baseline (no retrieval) first. Per type: where does the LLM already know the answer, and
   where does it fail? What *kind* of failures are they (wrong tool? invented option? not installed?)
2. A paragraph like `-a  Show systems that are alive.` — which command is it from? What does your retriever
   see? How could you give paragraphs context, or retrieve at a different granularity?
3. "Which command?" and "which options?" are different retrieval problems. Should they be the same index?
4. The request says *what* the user wants, the man page says *how*. Can the LLM help bridge the vocabulary gap
   before retrieval (query rewriting, "what would the man page say")?
5. `train_examples.jsonl` contains no answer for any question command. Why might retrieving *examples* still help?
   What do they teach the model that man pages don't?
6. Can you detect that an option in your answer does not exist? What do you do then?
7. When does retrieval *hurt*? Look for questions the closed-book model gets right and your RAG pipeline gets wrong.
8. Look at 10 answers the judge accepted and 10 it rejected. Do you agree? What does that tell you about your
   offline evaluation?
