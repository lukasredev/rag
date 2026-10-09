"""Run a pipeline on a track's questions and score it -- locally exactly like the server does.

    python -m ragkit.run --track papers --pipeline tracks/papers/skeleton/pipeline.py --limit 20

A pipeline is a Python file that defines:

    class Pipeline:
        def __init__(self, data_dir: pathlib.Path): ...       # load the corpus, build your index
        def answer(self, question: dict) -> dict:              # question has at least "id" and "question"
            return {"answer": "...", "citations": ["doc-id", ...]}

`answer()` is called from several threads at once, so don't mutate shared state in it.

A track is a folder tracks/<name>/ with
    track.py          NAME, TITLE, PRIMARY metric, PUBLIC_FIELDS, score(question, prediction) -> dict
    data/             the corpus + test.jsonl (public questions with answers)
    hidden/eval.jsonl hidden evaluation questions (only on the server)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from . import config
from .spend import SpendCapError

ROOT = Path(__file__).resolve().parent.parent


def tracks_dir() -> Path:
    return Path(config.get("RAG_TRACKS_DIR", str(ROOT / "tracks")))


def _import_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))  # lets a pipeline import helper files next to it
    spec.loader.exec_module(module)
    return module


def load_track(name: str):
    track = _import_file(tracks_dir() / name / "track.py", f"track_{name}")
    track.DIR = tracks_dir() / name
    return track


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    with open(path) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return questions[:limit] if limit else questions


def public_view(track, question: dict) -> dict:
    """Only the fields a pipeline is allowed to see (no gold answers)."""
    return {k: question[k] for k in track.PUBLIC_FIELDS if k in question}


def predict(pipeline_path: Path, data_dir: Path, questions: list[dict], workers: int = 8, show_progress: bool = True):
    """Build the pipeline once, then answer every question. Returns ({id: prediction}, timing info)."""
    module = _import_file(pipeline_path.resolve(), "student_pipeline")
    t0 = time.time()
    pipeline = module.Pipeline(data_dir)
    build_seconds = time.time() - t0

    def one(q):
        start = time.time()
        try:
            out = pipeline.answer(dict(q))
            if not isinstance(out, dict):
                out = {"answer": str(out), "citations": []}
            out.setdefault("answer", "")
            out.setdefault("citations", [])
            out["answer"] = "" if out["answer"] is None else str(out["answer"])
            out["citations"] = [str(c) for c in (out["citations"] or [])]
        except SpendCapError:
            raise
        except Exception:
            out = {"answer": "", "citations": [], "error": traceback.format_exc(limit=3)}
        out["seconds"] = round(time.time() - start, 3)
        return q["id"], out

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(one, questions), total=len(questions), disable=not show_progress, desc="answering"))
    timing = {"build_seconds": round(build_seconds, 1), "answer_seconds": round(time.time() - t1, 1)}
    return dict(results), timing


def evaluate(track, questions: list[dict], predictions: dict, workers: int = 8) -> dict:
    """Score predictions against gold questions. Returns overall + per-type means and per-question scores."""

    scoring_errors: list[str] = []

    def one(q):
        pred = predictions.get(q["id"], {"answer": "", "citations": []})
        try:
            return q, pred, track.score(q, pred)
        except SpendCapError:
            raise
        except Exception:
            # e.g. the LLM judge could not be reached: the score of this question is unknown, not 0
            scoring_errors.append(traceback.format_exc(limit=2))
            return q, pred, {track.PRIMARY: 0.0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, questions))

    def mean_table(selected):
        # Each metric is averaged over the questions that report it, so a metric that only applies to
        # some question types is not diluted. The primary metric must be reported for every question.
        keys = sorted({k for _, _, s in selected for k in s} | {track.PRIMARY})
        table = {}
        for k in keys:
            values = [s.get(k, 0.0) if k == track.PRIMARY else s[k] for _, _, s in selected if k in s or k == track.PRIMARY]
            table[k] = round(sum(values) / max(len(values), 1), 4)
        return table

    by_type: dict[str, list] = {}
    for row in rows:
        by_type.setdefault(row[0].get("type", "all"), []).append(row)
    return {
        "track": track.NAME,
        "primary": track.PRIMARY,
        "n": len(rows),
        "pipeline_errors": sum(1 for _, p, _ in rows if p.get("error")),
        "scoring_errors": len(scoring_errors),
        "scoring_error_samples": scoring_errors[:3],
        "overall": mean_table(rows),
        "by_type": {t: {"n": len(r), **mean_table(r)} for t, r in sorted(by_type.items())},
        "per_question": [
            {"id": q["id"], "type": q.get("type"), "question": q["question"], "prediction": p, "scores": s}
            for q, p, s in rows
        ],
    }


def print_report(report: dict, timing: dict | None = None, usage: dict | None = None) -> None:
    primary = report["primary"]
    print(f"\n=== {report['track']}: {report['n']} questions ===")
    print("overall: " + "  ".join(f"{k}={v:.3f}" for k, v in report["overall"].items()))
    print(f"\n{'type':<24}{'n':>5}  {primary:>10}")
    for t, row in report["by_type"].items():
        print(f"{t:<24}{row['n']:>5}  {row.get(primary, 0):>10.3f}")
    if report["pipeline_errors"]:
        print(f"\nWARNING: {report['pipeline_errors']} questions raised an exception in your pipeline "
              f"(see 'error' in the per-question output); they count as wrong")
    if report["scoring_errors"]:
        print(f"\nWARNING: scoring failed for {report['scoring_errors']} questions (judge unreachable?); "
              f"these scores are NOT valid:\n{report['scoring_error_samples'][0]}")
    if timing:
        print(f"\ntiming: {timing}")
    if usage:
        print(f"llm usage: {usage}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    parser.add_argument("--track", help="track name, e.g. papers")
    parser.add_argument("--pipeline", type=Path, help="path to your pipeline.py")
    parser.add_argument("--split", default="test", help="question file in tracks/<track>/data/<split>.jsonl")
    parser.add_argument("--questions", type=Path, help="explicit questions file (overrides --split)")
    parser.add_argument("--limit", type=int, help="only the first N questions (fast iteration)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, help="write the full report (with per-question details) here")

    p_pred = sub.add_parser("predict", help="only produce predictions (used by the server sandbox)")
    p_pred.add_argument("--track", required=True)
    p_pred.add_argument("--pipeline", type=Path, required=True)
    p_pred.add_argument("--questions", type=Path, required=True)
    p_pred.add_argument("--out", type=Path, required=True)
    p_pred.add_argument("--workers", type=int, default=8)

    args = parser.parse_args(argv)
    if not args.track or not args.pipeline:
        parser.error("--track and --pipeline are required")
    track = load_track(args.track)
    data_dir = track.DIR / "data"

    if args.cmd == "predict":
        questions = [public_view(track, q) for q in load_questions(args.questions)]
        predictions, timing = predict(args.pipeline, data_dir, questions, args.workers, show_progress=False)
        from .llm import usage

        args.out.write_text(json.dumps({"predictions": predictions, "timing": timing, "usage": usage()}))
        return

    qpath = args.questions or data_dir / f"{args.split}.jsonl"
    questions = load_questions(qpath, args.limit)
    predictions, timing = predict(args.pipeline, data_dir, [public_view(track, q) for q in questions], args.workers)
    from .llm import usage

    pipeline_usage = usage()  # snapshot before scoring, so LLM-judge calls are not counted
    report = evaluate(track, questions, predictions, args.workers)
    report["timing"], report["usage"] = timing, pipeline_usage
    print_report(report, timing, report["usage"])
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    try:
        main()
    except SpendCapError as e:
        sys.exit(f"\nSTOPPED: {e}")
