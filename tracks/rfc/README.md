# Track `rfc` — Ask the RFCs

**Scenario.** You are building the "ask the spec" assistant for a networking course. Students and engineers
ask precise questions about Internet standards — *"Which alert does a TLS 1.3 server send when …?"*,
*"What is the minimum value of X in HTTP/2?"*, *"What did RFC 2616 say about …?"* — and the assistant must
answer from the **actual RFC text**, cite where it found it, use the **current** version of a standard unless
the question asks about an old one, and say `UNANSWERABLE` when the corpus does not answer the question.

The LLM you use has probably read these RFCs many times. So the questions target specific details: exact
values, conditions, the version of a document, things only the text settles. Try a closed-book baseline first
(the LLM answers without any retrieval) to see how much your model knows on its own.

## Data (`data/`)

| file | what |
|---|---|
| `rfcs/rfcNNNN.txt` | 44 RFCs as **raw plain text** from rfc-editor.org (6.1 MB) — page headers/footers, form feeds, tables of contents, ASCII tables and diagrams included |
| `index.json` | one record per RFC: `id` (`"rfc9110"`), `number`, `title`, `date`, `year`, `status`, `obsoletes`, `obsoleted_by`, `updates`, `updated_by` (all RFCs) and the same four lists restricted to this corpus (`*_in_corpus`) |
| `test.jsonl` | public questions **with** gold answers — develop on these |

The corpus contains **version chains on purpose**:
HTTP/1.1 2616 → 7230/7231 → 9110/9112, HTTP/2 7540 → 9113, TLS 1.2 5246 → 1.3 8446, TCP 793 → 9293,
IPv6 2460 → 8200, SMTP 2821 → 5321, JSON 7159 → 8259, BCP 14 2119 (+ 8174). "Current" always means
current *within this corpus* (`obsoleted_by_in_corpus` is empty). Outside the corpus, the world moved on:
`index.json` shows that RFC 8446 has itself been obsoleted by RFC 9846, which is not included. Plus: HTTP/3 9114, QUIC 9000,
UDP 768, IP 791, ICMP 792, ARP 826, DNS 1034/1035, DHCP 2131, NTP 5905, CIDR 4632, NAT 3022, IMF 5322,
MIME 2045, URI 3986, UTF-8 3629, Base64 4648, WebSocket 6455, OAuth 2.0 6749, JWT 7519, Cookies 6265,
robots.txt 9309, TLS extensions/SNI 6066, Telnet 854, FTP 959.

Question record (what your pipeline sees is only `id` and `question`):

```json
{"id": "rfc-1a2b3c4d", "question": "In HTTP/2, ...", "type": "fact",
 "answers": ["..."], "gold_docs": ["rfc9113"], "gold_sections": ["rfc9113#6.5.2"],
 "source_sections": ["rfc9113 §6.5.2 Defined Settings"], "evidence": ["<verbatim quote>"],
 "closed_book_known": null, "validation": "reviewer", "human_reviewed": true}
```

`version` questions also have `target` (`"current"` or `"old"`) and `other_version_answer` (what the other
version says, as claimed by the generator, not reviewed). `unanswerable` questions have
`answers: ["UNANSWERABLE"]` and a `why_unanswerable`. `gold_sections` is empty when the gold passage has no
section number (RFC 768/792/826/854 have none). `closed_book_known` / `validation` / `human_reviewed`
describe how the question was validated (see the instructor section); they are not used for scoring.

### Question types

| type | what it tests |
|---|---|
| `fact` | one specific value: a limit, default, constant, code, field, named item |
| `definition` | what something means / what an implementation MUST or SHOULD do in a specific situation |
| `version` | the answer depends on the document version: "according to the current specification …" or explicitly about an old RFC ("In RFC 2616 …") |
| `multi_doc` | needs two RFCs (e.g. WebSocket + HTTP semantics, JWT + Base64, HTTP/3 + QUIC). After the 2026-10 audit only 2 test / 5 eval remain: most generated ones were two unrelated facts or one fact plus framing, and became single-RFC `fact` questions |
| `unanswerable` | plausible question the corpus does not answer → the answer must be `UNANSWERABLE` |

## Pipeline contract

```python
class Pipeline:
    def __init__(self, data_dir: pathlib.Path): ...           # build your index (≤ ~5 min on CPU)
    def answer(self, question: dict) -> dict:                  # called from 8 threads
        return {"answer": "...", "citations": ["rfc9110#15.5.1", "rfc9112"]}
```

* **Abstention:** if the corpus does not answer the question, the answer must contain the token
  `UNANSWERABLE` (exactly this word, any case). An answer containing it counts as an abstention.
* **Citations:** RFC ids, either document level (`"rfc9110"`) or section level (`"rfc9110#4.2.1"`;
  `"RFC 9110 §4.2.1"` and `"RFC 9110, Section 4.2.1"` are also understood). Appendix sections are letters
  (`"rfc9112#C.3"`). Section numbers are the RFC's own numbers. **Only the first 5 citations count.**

## Metrics (`track.py`)

| metric | definition |
|---|---|
| **`correct`** (primary) | answerable question: 1 if the answer agrees with the gold answer — exact match after normalisation, otherwise an LLM judge (`ragkit.metrics.llm_judge`, claude-haiku) decides; an abstention scores 0. Unanswerable question: 1 iff the answer contains `UNANSWERABLE` |
| `citation_recall@5` | fraction of the gold RFCs (`gold_docs`) among the documents of your first 5 citations. For `version` questions the gold RFC is the *target* version, so this row says "did you cite the right version". For unanswerable questions there is nothing to find → 1.0 (ragkit convention) |
| `section_hit@5` | 1 if one of your first 5 citations names a gold section (or its parent/child: citing §4.2 for gold §4.2.1 counts). Gold passages without section numbers (RFC 768/792/826/854) fall back to the document; unanswerable → 1.0 |
| `abstained` | fraction of answers that abstain. On the `unanswerable` row you want 1.0, on all other rows 0.0 |

The overall row is the mean over all questions; look at the **per-type rows** — that is where the story is.

## Rules

* Use `ragkit.llm.chat` (default model) for generation. You may use any retrieval you like, with the
  embedding models and rerankers listed in the main README ("Models available on the server").
* Do not hard-code answers or use the `test.jsonl` gold fields at answer time. Do not fetch anything from the
  internet at answer time (the sandbox only lets the LLM API through anyway).
* Every submission needs a `--note` with your hypothesis: what did you change, what do you expect?

## How to run

```bash
# in the repo root, with the venv active and your .env set up (see START_HERE.md)
mkdir -p mine && cp tracks/rfc/skeleton/pipeline.py mine/pipeline.py
python -m ragkit.run --track rfc --pipeline mine/pipeline.py --limit 20            # quick check
python -m ragkit.run --track rfc --pipeline mine/pipeline.py --out report.json     # whole test set
python scripts/failures.py report.json                                         # what went wrong
python -m studio --pipeline mine/pipeline.py                                   # every step, in the browser

# submit to the leaderboard (hidden questions); a folder is zipped, so helper modules are fine
python scripts/submit.py --track rfc --pipeline mine/ --label sections-v1 \
    --note "section chunks + BM25; expect better fact accuracy because header names match exactly" --wait
```

`--out` writes every question with your prediction and scores — read your failures, not just the mean.

## Questions to think about

1. Open `data/rfcs/rfc793.txt` and `rfc9110.txt` in an editor. What in there is *not* content? What happens
   to a 1000-character window that straddles a page break?
2. RFCs are structured documents. What is a natural unit of retrieval? How do you find section boundaries
   reliably — and how do you know your parser is right?
3. Questions say "SETTINGS_MAX_CONCURRENT_STREAMS", "Retry-After", "0x1c", "DHCPNAK". What does a sentence
   embedding model do with such tokens? What does BM25 do?
4. RFC 2616 and RFC 9110 both answer "what does status 413 mean" — differently. How does your pipeline know
   which one to trust? When should it use the old one?
5. When should the model say `UNANSWERABLE`? What does it cost you on the other types if it says it too often?
   Look at the `abstained` column per type.
6. The model knows a lot about RFCs by heart. When is that helpful, and when does it hurt (hint: old
   versions, exact constants)? How can your prompt make the model trust the passages over its memory?
7. Your citations are measured separately from the answer. Is your retriever or your generator the bottleneck?
   (Compare `citation_recall@5` with `correct` per type.)
8. `multi_doc` questions need two documents. Does one query find both?
