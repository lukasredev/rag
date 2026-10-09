"""RAG Studio: a local web app that shows what your pipeline does, question by question.

    python -m studio                       # then open http://127.0.0.1:8765
    python -m studio --pipeline my_pipeline.py --pipeline other/pipeline.py

It only ever uses the public test set (tracks/<name>/data/test.jsonl). Runs are stored in .studio/runs/.

Shared, read-only deployment (see studio/__main__.py --public): RAG_STUDIO_PUBLIC=1 shows only the guide, the
starter pipelines' code and the corpus. Nothing can be loaded, run or changed, so no code runs and no LLM is
called. Every page needs the shared password in RAG_STUDIO_PASSWORD.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ragkit import config, spend, trace
from ragkit.build import footer_html
from studio.corpus import ADAPTERS as CORPUS_ADAPTERS
from studio.corpus import length_histogram
from studio.corpus import search as corpus_search
from studio.corpus import snippet as corpus_snippet
from ragkit.llm import usage
from ragkit.run import evaluate, load_questions, load_track, public_view, tracks_dir

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
STATE = Path(config.get("RAG_STUDIO_DIR", str(ROOT / ".studio")))
RUNS = STATE / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

# Question fields that hold the ids a good pipeline should retrieve/cite, per track.
GOLD_ID_FIELDS = ["evidence", "evidence_ids", "gold_docs", "gold_sections", "gold_id", "oracle_man"]

app = FastAPI(title="RAG Studio")
PUBLIC = config.get("RAG_STUDIO_PUBLIC") == "1"
EXTRA_PIPELINES: list[Path] = []  # set by __main__ from --pipeline
_lock = threading.Lock()
_tracks: dict[str, object] = {}
_loaded: dict[str, "Loaded"] = {}
_jobs: dict[str, dict] = {}


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────


def track(name: str):
    with _lock:
        if name not in _tracks:
            if not (tracks_dir() / name / "track.py").exists():
                raise HTTPException(404, f"unknown track {name}")
            _tracks[name] = load_track(name)
        return _tracks[name]


def questions_of(name: str) -> list[dict]:
    return load_questions(track(name).DIR / "data" / "test.jsonl")


def gold_ids(q: dict) -> list[str]:
    out: list[str] = []
    for field in GOLD_ID_FIELDS:
        value = q.get(field)
        stack = [value] if isinstance(value, str) else list(value or [])
        while stack:
            v = stack.pop(0)
            if isinstance(v, (list, tuple)):
                stack[:0] = list(v)
            elif isinstance(v, str) and v and " " not in v.strip() and v not in out:  # ids, not quoted evidence text
                out.append(v)
    return out


def gold_answer(q: dict):
    return q.get("answers") or q.get("answer") or q.get("gold_func_name")


def price(model: str):
    return spend.price(model)


def cost_of(events: list[dict]) -> dict:
    """Tokens and estimated USD of the LLM calls in a trace. Cached calls are free; for them we also
    estimate what they would have cost (about 4 characters per token)."""
    out = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "tokens_uncached": 0, "usd": 0.0,
           "usd_uncached": 0.0, "priced": True}
    for e in events:
        if e["step"] != "llm":
            continue
        d = e["data"]
        out["calls"] += 1
        n_in, n_out = d.get("input_tokens", 0), d.get("output_tokens", 0)
        if d.get("cached"):
            out["cached"] += 1
            chars_in = len(json.dumps(d.get("messages"))) + len(d.get("system") or "")
            n_in_est, n_out_est = chars_in / 4, len(d.get("response") or "") / 4 + 200
        else:
            n_in_est, n_out_est = n_in, n_out
        out["input_tokens"] += n_in
        out["output_tokens"] += n_out
        out["tokens_uncached"] += int(n_in_est + n_out_est)
        p = price(d.get("model", ""))
        if p is None:
            out["priced"] = False
            continue
        out["usd"] += (n_in * p[0] + n_out * p[1]) / 1e6
        out["usd_uncached"] += (n_in_est * p[0] + n_out_est * p[1]) / 1e6
    out["usd"], out["usd_uncached"] = round(out["usd"], 6), round(out["usd_uncached"], 6)
    return out


def pipelines_of(name: str) -> list[dict]:
    tdir = tracks_dir() / name
    found = [("skeleton", tdir / "skeleton" / "pipeline.py")]
    if PUBLIC:  # the shared Studio shows only the starter code, never solutions or files from the server's disk
        found = [(label, path) for label, path in found if path.exists()]
        return [{"label": label, "path": str(path.resolve()), "doc": _docstring(path), "custom": False} for label, path in found]
    examples = sorted((tdir / "examples").iterdir()) if (tdir / "examples").exists() else []  # not in the student repo
    found += [(d.name, d / "pipeline.py") for d in examples if (d / "pipeline.py").exists()]
    found += [(f"{p.parent.name}/{p.name}", p) for p in EXTRA_PIPELINES + _user_pipelines()]
    out = []
    for label, path in found:
        doc = _docstring(path)
        out.append({"label": label, "path": str(path.resolve()), "doc": doc, "custom": not path.is_relative_to(tdir)})
    return out


def _user_pipelines() -> list[Path]:
    f = STATE / "pipelines.json"
    return [Path(p) for p in json.loads(f.read_text())] if f.exists() else []


def _docstring(path: Path) -> str:
    import ast

    try:
        return ast.get_docstring(ast.parse(path.read_text())) or ""
    except Exception:  # noqa: BLE001
        return ""


def _signature(path: Path) -> str:
    """Changes whenever a .py file next to the pipeline changes, so edits trigger a rebuild."""
    parts = [f"{p.name}:{p.stat().st_mtime_ns}" for p in sorted(path.parent.glob("*.py"))]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


# ── loaded pipelines (index built once, rebuilt when the code changes) ──────────────────────────


class Loaded:
    def __init__(self, track_name: str, path: Path):
        self.track, self.path = track_name, path
        self.signature = _signature(path)
        self.status, self.error, self.obj = "building", None, None
        self.build_seconds, self.build_trace = None, []
        self.started = time.time()
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self):
        try:
            module = _import_pipeline(self.path)
            with trace.collect() as events:
                obj = module.Pipeline(track(self.track).DIR / "data")
            self.obj, self.build_trace = obj, events
            self.status = "ready"
        except BaseException:  # noqa: BLE001 - show the student any failure, even SystemExit
            self.error, self.status = traceback.format_exc(limit=6), "error"
        self.build_seconds = round(time.time() - self.started, 1)

    def info(self) -> dict:
        return {"track": self.track, "path": str(self.path), "status": self.status, "error": self.error,
                "build_seconds": self.build_seconds if self.build_seconds is not None
                else round(time.time() - self.started, 1),
                "build_trace": self.build_trace, "stale": self.signature != _signature(self.path)}


def _import_pipeline(path: Path):
    folder = str(path.parent.resolve())
    # forget helper modules imported earlier from this folder, so edits to them are picked up too
    for name, mod in list(sys.modules.items()):
        if (getattr(mod, "__file__", None) or "").startswith(folder + os.sep):
            del sys.modules[name]
    name = "studio_pipeline_" + hashlib.sha1(folder.encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if folder in sys.path:
        sys.path.remove(folder)
    sys.path.insert(0, folder)
    spec.loader.exec_module(module)
    return module


def get_loaded(track_name: str, path: str, rebuild_if_stale: bool = True) -> Loaded:
    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(404, f"{p} does not exist")
    key = f"{track_name}|{p}"
    with _lock:
        cur = _loaded.get(key)
        if cur is None or cur.status == "error" or (rebuild_if_stale and cur.signature != _signature(p)):
            if cur is None or cur.status != "building":
                cur = _loaded[key] = Loaded(track_name, p)
        return cur


def run_one(loaded: Loaded, q: dict, score: bool) -> dict:
    """Answer one question under a trace; score it if it has gold answers."""
    t = track(loaded.track)
    start = time.time()
    with trace.collect() as events:
        try:
            out = loaded.obj.answer(public_view(t, q) if q.get("id") else dict(q))
            if not isinstance(out, dict):
                out = {"answer": str(out), "citations": []}
            out = {"answer": "" if out.get("answer") is None else str(out.get("answer")),
                   "citations": [str(c) for c in (out.get("citations") or [])]}
        except Exception:  # noqa: BLE001
            out = {"answer": "", "citations": [], "error": traceback.format_exc(limit=4)}
    out["seconds"] = round(time.time() - start, 3)
    result = {"prediction": out, "trace": events, "cost": cost_of(events)}
    if score and q.get("id"):
        with trace.collect() as judge_events:
            try:
                result["scores"] = t.score(q, out)
            except Exception:  # noqa: BLE001
                result["scores"], result["scoring_error"] = {}, traceback.format_exc(limit=3)
        result["judge"] = judge_events
        result["judge_cost"] = cost_of(judge_events)
    return result


# ── API ─────────────────────────────────────────────────────────────────────────────────────────


@app.get("/api/info")
def info():
    model = config.llm_model()
    has_key = bool(config.get("ANTHROPIC_API_KEY") or config.get("OPENAI_API_KEY"))
    tracks = []
    for d in sorted(tracks_dir().iterdir()):
        if not (d / "track.py").exists() or not (d / "data" / "test.jsonl").exists():
            continue
        t = track(d.name)
        qs = questions_of(d.name)
        types: dict[str, int] = {}
        for q in qs:
            types[q.get("type", "all")] = types.get(q.get("type", "all"), 0) + 1
        tracks.append({"name": d.name, "title": t.TITLE, "primary": t.PRIMARY, "n": len(qs), "types": types,
                       "pipelines": pipelines_of(d.name)})
    return {"tracks": tracks, "model": model, "judge_model": config.get("RAG_JUDGE_MODEL", "claude-haiku-5-5"),
            "embed_model": config.embed_model(), "models": config.allowed_models(), "prices": spend.PRICES,
            "has_key": has_key, "spent": round(spend.total(), 4), "cap": spend.cap(), "public": PUBLIC,
            "footer": footer_html()}


@app.get("/api/guide")
def guide():
    """START_HERE.md, with the server address filled in when RAG_SERVER is set."""
    path = ROOT / "START_HERE.md"
    text = path.read_text() if path.exists() else "# Start here\n\nSTART_HERE.md is missing from this folder."
    server = config.get("RAG_SERVER")
    if server:
        text = text.replace("http://<server address>:8000", server.rstrip("/"))
    return {"markdown": text}


@app.get("/api/spend")
def spent():
    return {"spent": round(spend.total(), 4), "cap": spend.cap()}


class ModelIn(BaseModel):
    model: str


@app.post("/api/model")
def set_model(body: ModelIn):
    """Switch the default chat model for every pipeline that doesn't pick its own."""
    if body.model.strip() not in config.allowed_models():
        raise HTTPException(400, f"model not allowed; allowed: {', '.join(config.allowed_models())}")
    os.environ["RAG_LLM_MODEL"] = body.model.strip()
    return {"model": config.llm_model()}


@app.get("/api/questions/{track_name}")
def questions(track_name: str):
    qs = questions_of(track_name)
    return [{**q, "gold_ids": gold_ids(q), "gold_answer": gold_answer(q)} for q in qs]


class PipelineIn(BaseModel):
    path: str


@app.post("/api/pipelines")
def add_pipeline(body: PipelineIn):
    p = Path(body.path).expanduser().resolve()
    if p.is_dir():
        p = p / "pipeline.py"
    if not p.exists() or p.suffix != ".py":
        raise HTTPException(400, f"{p} is not a Python file")
    saved = [str(x) for x in _user_pipelines()]
    if str(p) not in saved:
        (STATE / "pipelines.json").write_text(json.dumps(saved + [str(p)], indent=1))
    return {"path": str(p)}


@app.get("/api/source")
def source(path: str):
    p = Path(path).resolve()
    allowed = [x["path"] for d in tracks_dir().iterdir() if (d / "track.py").exists() for x in pipelines_of(d.name)]
    if str(p) not in allowed:
        raise HTTPException(403, "only listed pipelines can be shown")
    files = {f.name: f.read_text() for f in sorted(p.parent.glob("*.py"))}
    return {"main": p.name, "files": files}


class LoadIn(BaseModel):
    track: str
    path: str


@app.post("/api/load")
def load(body: LoadIn):
    return get_loaded(body.track, body.path).info()


@app.get("/api/load")
def load_status(track: str, path: str):
    return get_loaded(track, path, rebuild_if_stale=False).info()


class AskIn(BaseModel):
    track: str
    path: str
    question_id: str | None = None
    question: str | None = None


@app.post("/api/ask")
def ask(body: AskIn):
    loaded = get_loaded(body.track, body.path)
    if loaded.status != "ready":
        raise HTTPException(409, f"pipeline is {loaded.status}")
    if body.question_id:
        q = next((q for q in questions_of(body.track) if q["id"] == body.question_id), None)
        if q is None:
            raise HTTPException(404, "unknown question id")
    else:
        q = {"question": (body.question or "").strip()}
        if not q["question"]:
            raise HTTPException(400, "empty question")
    result = run_one(loaded, q, score=True)
    result["question"] = {**q, "gold_ids": gold_ids(q), "gold_answer": gold_answer(q)} if q.get("id") else q
    return result


class RunIn(BaseModel):
    track: str
    path: str
    label: str = ""
    limit: int | None = 20
    types: list[str] | None = None
    workers: int = 4


@app.post("/api/runs")
def start_run(body: RunIn):
    loaded = get_loaded(body.track, body.path)
    if loaded.status != "ready":
        raise HTTPException(409, f"pipeline is {loaded.status}")
    qs = [q for q in questions_of(body.track) if not body.types or q.get("type") in body.types]
    qs = qs[: body.limit] if body.limit else qs
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    job = {"id": run_id, "track": body.track, "path": str(loaded.path), "label": body.label or Path(loaded.path).parent.name,
           "status": "running", "done": 0, "total": len(qs), "cost": 0.0, "cancel": False, "created": time.time(),
           "model": config.llm_model()}
    _jobs[run_id] = job
    threading.Thread(target=_run_job, args=(job, loaded, qs, max(1, min(body.workers, 8))), daemon=True).start()
    return {k: v for k, v in job.items() if k != "cancel"}


def _run_job(job: dict, loaded: Loaded, qs: list[dict], workers: int):
    results: dict[str, dict] = {}

    def one(q):
        if job["cancel"]:
            return
        r = run_one(loaded, q, score=False)
        results[q["id"]] = r
        if "SpendCapError" in r["prediction"].get("error", ""):
            job["cancel"], job["error"] = True, r["prediction"]["error"].strip().splitlines()[-1]
        job["done"] += 1
        job["cost"] = round(job["cost"] + r["cost"]["usd"], 6)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, qs))
        qs = [q for q in qs if q["id"] in results]
        job["status"] = "scoring"
        before = usage()
        report = evaluate(track(loaded.track), qs, {i: r["prediction"] for i, r in results.items()}, workers=workers)
        after = usage()
        judge_tokens = {"input_tokens": after["input_tokens"] - before["input_tokens"],
                        "output_tokens": after["output_tokens"] - before["output_tokens"]}
        by_id = {q["id"]: q for q in qs}
        for row in report["per_question"]:
            q, r = by_id[row["id"]], results[row["id"]]
            row.update(trace=r["trace"], cost=r["cost"], gold_ids=gold_ids(q), gold_answer=gold_answer(q))
        totals = {k: sum(r["cost"][k] for r in results.values()) for k in
                  ("calls", "cached", "input_tokens", "output_tokens", "tokens_uncached", "usd", "usd_uncached")}
        report.update({k: job[k] for k in ("id", "track", "path", "label", "created", "model")},
                      cancelled=job["cancel"], totals=totals, judge_tokens=judge_tokens,
                      seconds=round(time.time() - job["created"], 1))
        (RUNS / f"{job['id']}.json").write_text(json.dumps(report))
        job["status"] = "cancelled" if job["cancel"] else "done"
    except spend.SpendCapError as e:
        job["status"], job["error"] = "error", str(e)
    except Exception:  # noqa: BLE001
        job["status"], job["error"] = "error", traceback.format_exc(limit=5)


@app.get("/api/jobs/{run_id}")
def job_status(run_id: str):
    job = _jobs.get(run_id)
    if not job:
        raise HTTPException(404)
    return {k: v for k, v in job.items() if k != "cancel"}


@app.post("/api/jobs/{run_id}/cancel")
def cancel(run_id: str):
    if run_id in _jobs:
        _jobs[run_id]["cancel"] = True
    return {"ok": True}


def _summary(r: dict, source: str) -> dict:
    return {"id": r.get("id"), "source": source, "track": r["track"], "label": r.get("label"), "model": r.get("model"),
            "created": r.get("created"), "n": r["n"], "primary": r["primary"], "overall": r["overall"],
            "by_type": r["by_type"], "totals": r.get("totals"), "usage": r.get("usage"),
            "pipeline_errors": r.get("pipeline_errors", 0), "scoring_errors": r.get("scoring_errors", 0),
            "cancelled": r.get("cancelled", False)}


def _is_opus(model) -> bool:
    return "opus" in str(model or "").lower()


@app.get("/api/runs")
def list_runs(track: str | None = None):
    out = []
    for f in sorted(RUNS.glob("*.json"), reverse=True):
        r = json.loads(f.read_text())
        if (not track or r["track"] == track) and not _is_opus(r.get("model")):
            out.append(_summary(r, "studio"))
    # reference results that ship with the tracks (no traces)
    for d in sorted(tracks_dir().iterdir()):
        if track and d.name != track:
            continue
        results = d / "results"
        # top-level files are the old Opus runs: not shown, students use Haiku
        files = sorted(results.glob("*/*.test.json")) if results.exists() else []
        for f in files:
            r = json.loads(f.read_text())
            if _is_opus(r.get("model")):
                continue
            r.setdefault("track", d.name)
            tag = f.parent.name
            rel = f.relative_to(results).as_posix()
            out.append({**_summary(r, "reference"), "id": f"ref:{d.name}:{rel}", "model": r.get("model") or tag,
                        "label": f"{f.name.removesuffix('.test.json')} ({tag})"})
    return out


@app.get("/api/runs/{run_id:path}")
def get_run(run_id: str):
    if run_id.startswith("ref:"):
        _, t, name = run_id.split(":", 2)
        f = tracks_dir() / t / "results" / name
        parts = name.split("/")
        if len(parts) != 2 or ".." in parts or not name.endswith(".test.json") or not f.exists():
            raise HTTPException(404)
        r = json.loads(f.read_text())
        if _is_opus(r.get("model")):
            raise HTTPException(404)
        r.setdefault("track", t)
        qs = {q["id"]: q for q in questions_of(t)}
        for row in r["per_question"]:
            q = qs.get(row["id"], {})
            row.update(gold_ids=gold_ids(q), gold_answer=gold_answer(q))
        tag = parts[0]
        return {**r, "id": run_id, "label": f"{parts[-1].removesuffix('.test.json')} ({tag})", "source": "reference",
                "model": r.get("model") or tag}
    f = RUNS / f"{run_id}.json"
    if "/" in run_id or not f.exists():
        raise HTTPException(404)
    return {**json.loads(f.read_text()), "source": "studio"}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    f = RUNS / f"{run_id}.json"
    if "/" in run_id or not f.exists():
        raise HTTPException(404)
    f.unlink()
    return {"ok": True}


# ── corpus explorer ─────────────────────────────────────────────────────────────────────────────

_corpora: dict[str, dict] = {}


def corpus(track_name: str) -> dict:
    """The track's documents (loaded once) plus which test questions point at which document."""
    with _lock:
        if track_name in _corpora:
            return _corpora[track_name]
    adapter = CORPUS_ADAPTERS.get(track_name)
    if adapter is None:
        raise HTTPException(404, f"no corpus explorer for track {track_name}")
    c = adapter(track(track_name).DIR / "data")
    refs: dict[str, list[dict]] = {}
    q_docs: dict[str, list[str]] = {}
    for q in questions_of(track_name):
        docs = [e["doc"] for e in c.evidence(q)]
        for cid in gold_ids(q) + [q[k] for k in ("paper_id", "cmd_name") if q.get(k)]:
            doc, _ = c.resolve(cid)
            if doc and doc not in docs:
                docs.append(doc)
        q_docs[q["id"]] = docs
        for doc in docs:
            refs.setdefault(doc, []).append({"id": q["id"], "type": q.get("type"), "question": q["question"]})
    entry = {"corpus": c, "refs": refs, "q_docs": q_docs}
    with _lock:
        _corpora[track_name] = entry
    return entry


def _doc_summary(d: dict, refs: dict, q: str = "") -> dict:
    return {"id": d["id"], "title": d["title"], "meta": {k: v for k, v in d["meta"].items() if k != "url"},
            "words": d["words"], "questions": len(refs.get(d["id"], [])), "snippet": corpus_snippet(d, q)}


@app.get("/api/corpus-overview")
def corpus_overview():
    """One line per track for the corpus switcher."""
    out = []
    for name in sorted(CORPUS_ADAPTERS):
        if not (tracks_dir() / name / "data" / "test.jsonl").exists():
            continue
        c = corpus(name)["corpus"]
        out.append({"track": name, "title": track(name).TITLE, "unit": c.unit, "n": len(c.docs),
                    "words": sum(d["words"] for d in c.docs), "questions": len(questions_of(name))})
    return out


@app.get("/api/corpus/{track_name}/stats")
def corpus_stats(track_name: str):
    entry = corpus(track_name)
    c, refs = entry["corpus"], entry["refs"]
    words = sorted(d["words"] for d in c.docs)
    facets = {f: Counter(str(d["meta"].get(f)) for d in c.docs).most_common(40) for f in c.facets}
    return {"unit": c.unit, "note": c.note, "n": len(c.docs), "passages": sum(len(d["parts"]) for d in c.docs),
            "words": sum(words), "median_words": words[len(words) // 2] if words else 0,
            "max_words": words[-1] if words else 0, "histogram": length_histogram(words), "facets": facets,
            "referenced": len(refs), "files": sorted(p.name for p in (track(track_name).DIR / "data").iterdir()
                                                     if p.name != "test.jsonl")}


@app.get("/api/corpus/{track_name}/docs")
def corpus_docs(track_name: str, q: str = "", facet: str = "", sort: str = "", offset: int = 0, limit: int = 50):
    entry = corpus(track_name)
    filters = dict(f.split(":", 1) for f in facet.split("|") if ":" in f)
    hits = [d for _, d in corpus_search(entry["corpus"], q, filters)]
    if sort == "longest":
        hits.sort(key=lambda d: -d["words"])
    elif sort == "shortest":
        hits.sort(key=lambda d: d["words"])
    elif sort == "questions":
        hits.sort(key=lambda d: -len(entry["refs"].get(d["id"], [])))
    return {"total": len(hits), "docs": [_doc_summary(d, entry["refs"], q) for d in hits[offset:offset + limit]]}


@app.get("/api/corpus/{track_name}/doc")
def corpus_doc(track_name: str, id: str):
    entry = corpus(track_name)
    doc_id, anchor = entry["corpus"].resolve(id)
    if doc_id is None:
        raise HTTPException(404, f"{id} is not in the corpus")
    d = entry["corpus"].by_id[doc_id]
    return {"id": d["id"], "title": d["title"], "meta": d["meta"], "parts": d["parts"], "words": d["words"],
            "anchor": anchor, "questions": entry["refs"].get(d["id"], [])}


@app.get("/api/corpus/{track_name}/questions")
def corpus_questions(track_name: str):
    """Every test question with the documents it needs, for the link map."""
    entry = corpus(track_name)
    c = entry["corpus"]
    qs = [{"id": q["id"], "type": q.get("type"), "question": q["question"], "docs": entry["q_docs"].get(q["id"], [])}
          for q in questions_of(track_name)]
    docs = {d: {"title": c.by_id[d]["title"], "n": len(r)} for d, r in entry["refs"].items()}
    return {"unit": c.unit, "questions": qs, "docs": docs}


@app.get("/api/corpus/{track_name}/question")
def corpus_question(track_name: str, id: str):
    entry = corpus(track_name)
    c = entry["corpus"]
    q = next((q for q in questions_of(track_name) if q["id"] == id), None)
    if q is None:
        raise HTTPException(404, "unknown question")
    evidence = c.evidence(q)
    listed = {e["doc"] for e in evidence}
    evidence += [{"doc": d, "role": "gold", "passages": []} for d in entry["q_docs"].get(id, []) if d not in listed]
    for e in evidence:
        d = c.by_id[e["doc"]]
        e.update(title=d["title"], words=d["words"],
                 meta={k: v for k, v in d["meta"].items() if k not in ("url", "month")},
                 others=[r for r in entry["refs"].get(e["doc"], []) if r["id"] != id])
    return {"question": {**q, "gold_answer": gold_answer(q)}, "evidence": evidence, "unit": c.unit}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ── shared deployment: password and read-only ───────────────────────────────────────────────────────
# Requests the read-only Studio answers (all GET). Everything else (loading or running a pipeline, asking a
# question, adding pipelines, changing the model, runs) would run code or spend LLM budget.
PUBLIC_GET = ("/", "/static/", "/api/info", "/api/guide", "/api/questions/", "/api/source", "/api/corpus-overview",
              "/api/corpus/")
COOKIE = "rag_studio"
_secret = os.urandom(32)  # a restart logs everyone out


def _password() -> str:
    return config.get("RAG_STUDIO_PASSWORD") or ""


def _cookie_value() -> str:
    return hmac.new(_secret, _password().encode(), hashlib.sha256).hexdigest()


def _logged_in(request: Request) -> bool:
    return hmac.compare_digest(request.cookies.get(COOKIE, ""), _cookie_value())


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>RAG Studio</title><style>
body {{ font: 15px/1.5 system-ui, sans-serif; background:#F4F1EA; color:#1F1D1A; display:grid; place-items:center;
        min-height:100vh; margin:0; padding:0 16px; }}
form {{ background:#FBFAF6; border:1px solid #DDD6C8; border-radius:4px; padding:28px; width:100%; max-width:340px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} p {{ color:#6B655B; margin:0 0 16px; }}
input, button {{ font:inherit; width:100%; box-sizing:border-box; padding:9px 11px; border-radius:3px; }}
input {{ border:1px solid #CFC7B7; margin-bottom:10px; }} button {{ border:0; background:#B4532A; color:#fff; cursor:pointer; }}
.err {{ color:#A3341C; }} body {{ grid-template-rows:1fr auto; }}
footer {{ color:#6B655B; font-size:13px; padding:16px 0; text-align:center; }} footer a {{ color:inherit; }}</style></head><body><form method="post" action="/login">
<h1>RAG Studio</h1><p>{message}</p>
<input type="password" name="password" placeholder="workshop password" autofocus required>
<button>Open</button></form><footer>{footer}</footer></body></html>"""


@app.middleware("http")
async def public_guard(request: Request, call_next):
    if not PUBLIC:
        return await call_next(request)
    path = request.url.path
    if path == "/login":
        return await call_next(request)
    if not _logged_in(request):
        if path.startswith("/api/"):
            return JSONResponse({"detail": "log in first"}, status_code=401)
        return RedirectResponse("/login", status_code=303)
    allowed = request.method in ("GET", "HEAD") and any(
        path == p if not p.endswith("/") or p == "/" else path.startswith(p) for p in PUBLIC_GET)
    if not allowed:
        return JSONResponse({"detail": "the shared RAG Studio is read-only; run Studio on your laptop for this"},
                            status_code=403)
    return await call_next(request)


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return LOGIN_PAGE.format(message="Enter the workshop password.", footer=footer_html())


@app.post("/login")
async def login(request: Request):
    if not PUBLIC:
        raise HTTPException(404)
    form = urllib.parse.parse_qs((await request.body()).decode(errors="replace"))
    given = (form.get("password") or [""])[0]
    if not _password() or not hmac.compare_digest(given.encode(), _password().encode()):
        await asyncio.sleep(1)  # slows down guessing
        return HTMLResponse(LOGIN_PAGE.format(message='<span class="err">Wrong password.</span>', footer=footer_html()),
                            status_code=401)
    response = RedirectResponse("/", status_code=303)
    https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(COOKIE, _cookie_value(), httponly=True, samesite="lax", secure=https, max_age=7 * 24 * 3600)
    return response


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
