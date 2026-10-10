"""Shell Command Helper (tldr / DocPrompting): natural-language request -> ONE bash command.

Scoring (see README for the reasoning and the weaknesses):
    cmd_acc      the predicted command name equals the gold one (DocPrompting's "command name accuracy")
    exact        normalised token sequence equal (placeholders {{...}} -> <ph>)
    token_f1     token-level F1 between normalised prediction and gold (DocPrompting's main metric)
    installed    the predicted main command has a man page in data/docs.jsonl ("is installed on our servers")
    correct      PRIMARY: exact OR (installed AND an LLM judge, given the request, the gold command and its
                 man-page paragraphs, says the prediction does the same job)
    doc_recall@5 recall of the gold man-page paragraphs (oracle_man) among the first 5 citations
    cmd_recall@5 1 if any of the first 5 citations is a paragraph of the gold command's man page
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ragkit import config
from ragkit.metrics import recall_at_k

NAME = "shell"
TITLE = "Shell Command Helper (tldr / DocPrompting)"
PRIMARY = "correct"
PUBLIC_FIELDS = ["id", "question"]
MAX_CITATIONS = 5

_PLACEHOLDER = re.compile(r"\{\{.*?\}\}")
_TOKEN = re.compile(r"<ph>|[|;&<>]+|[^\s|;&<>=]+=?|=")
_PREFIXES = {"sudo", "doas", "env", "command", "time", "nohup"}


def extract_command(answer: str) -> str:
    """First non-empty line, without markdown code fences or a leading '$ ' prompt."""
    text = (answer or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if line.startswith("$ "):
            line = line[2:].strip()
        if line:
            return line
    return ""


def normalize_command(cmd: str) -> list[str]:
    """Tokens of a command with every {{placeholder}} replaced by '<ph>' and quotes removed."""
    cmd = _PLACEHOLDER.sub(" <ph> ", cmd)
    cmd = cmd.replace('"', " ").replace("'", " ")
    tokens = _TOKEN.findall(cmd)
    while tokens and (tokens[0] in _PREFIXES or re.match(r"^[A-Z_][A-Z0-9_]*=$", tokens[0])):
        tokens = tokens[1:] if not tokens[0].endswith("=") else tokens[2:]
    # "<ph> <ph>" next to each other (e.g. {{a}}{{b}} or {{x}}.mp4) collapses to one placeholder
    out: list[str] = []
    for t in tokens:
        if t == "<ph>" and out and out[-1] == "<ph>":
            continue
        out.append(t)
    return out


def command_names(prediction: str) -> set[str]:
    """Possible names of the main (first) command: {'docker', 'docker-start'} for 'docker start x'."""
    words = prediction.split()
    while words and (words[0] in _PREFIXES or re.match(r"^[A-Z_][A-Z0-9_]*=", words[0])):
        words = words[1:]
    if not words:
        return set()
    first = words[0].rsplit("/", 1)[-1].lower()  # /usr/bin/foo -> foo; MP4Box == mp4box
    names = {first}
    if len(words) > 1:
        names.add(f"{first}-{words[1].lower()}")
    return names


_installed: set[str] | None = None


def installed_commands() -> set[str]:
    global _installed
    if _installed is None:
        path = Path(__file__).resolve().parent / "data" / "commands.jsonl"
        with open(path) as f:
            _installed = {json.loads(line)["cmd"].lower() for line in f if line.strip()}
    return _installed


def _f1(pred: list[str], gold: list[str]) -> float:
    if not pred or not gold:
        return float(pred == gold)
    from collections import Counter

    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(gold)
    return 2 * p * r / (p + r)


JUDGE_SYSTEM = """You grade a shell-command assistant. You get a user's request, the reference bash command
(from tldr-pages), excerpts of the reference command's man page, and the predicted command.
Decide whether the predicted command does what the request asks, equivalently to the reference.
- A different command or tool than the reference is fine only if it clearly achieves the same result.
- Placeholders like {{path/to/file}} or <...> stand for user-supplied values; their names and example values
  do not matter. A concrete example value in place of a placeholder is fine.
- Short and long forms of the same option (-r / --recursive), different option order, harmless extra
  options and equivalent syntax (e.g. -o value vs -o=value) are fine.
- Wrong or missing options, wrong subcommand, a different behaviour, or several alternative commands
  in one answer are NOT fine. If you are unsure whether an option exists or means what is needed, use the
  man-page excerpts; if they do not settle it, answer false.
Reply with JSON only: {"equivalent": true} or {"equivalent": false}."""


def _judge(question: dict, prediction: str) -> float:
    from ragkit.llm import chat_json

    docs = "\n".join(f"- {t}" for t in question.get("oracle_text", [])[:8])
    prompt = (
        f"Request: {question['question']}\n\nReference command: {question['answer']}\n\n"
        f"Man-page excerpts for `{question['cmd_name']}`:\n{docs}\n\nPredicted command: {prediction[:1000]}"
    )
    verdict = chat_json(prompt, system=JUDGE_SYSTEM, model=config.get("RAG_JUDGE_MODEL", "claude-haiku-5-5"), max_tokens=2000, cache=True)
    return float(isinstance(verdict, dict) and verdict.get("equivalent") is True)


def score(question: dict, prediction: dict) -> dict[str, float]:
    pred_cmd = extract_command(prediction.get("answer", ""))
    gold_cmd = question["answer"]
    pred_toks, gold_toks = normalize_command(pred_cmd), normalize_command(gold_cmd)

    names = command_names(pred_cmd)
    cmd_ok = float(question["cmd_name"].lower() in names)
    installed = float(bool(names & installed_commands()))
    exact = float(pred_toks == gold_toks)
    correct = exact
    if not exact and installed:
        correct = _judge(question, pred_cmd)

    citations = [str(c) for c in (prediction.get("citations") or [])][:MAX_CITATIONS]
    gold_docs = question.get("oracle_man", [])
    prefix = question["cmd_name"] + "_"
    return {
        "correct": correct,
        "cmd_acc": cmd_ok,
        "exact": exact,
        "installed": installed,
        "token_f1": round(_f1(pred_toks, gold_toks), 4),
        "doc_recall@5": recall_at_k(citations, gold_docs, MAX_CITATIONS),
        "cmd_recall@5": float(any(c.startswith(prefix) and c[len(prefix):].isdigit() for c in citations)),
    }
