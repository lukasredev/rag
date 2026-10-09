"""Submit a pipeline to the evaluation server.

    python scripts/submit.py --track papers --pipeline my_pipeline.py \
        --note "Switched from 500-char chunks to section-based chunks; expect better recall on long papers" \
        --wait

Team name, token and server URL can also come from the env vars RAG_TEAM, RAG_TOKEN, RAG_SERVER
(or a .env file). If your pipeline needs helper modules, pass a directory: it is zipped automatically.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv


def pack(path: Path) -> tuple[str, bytes]:
    if path.is_file():
        return path.name if path.suffix == ".zip" else "pipeline.py", path.read_bytes()
    if not (path / "pipeline.py").exists():
        sys.exit(f"{path} has no pipeline.py")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in path.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix in {".py", ".txt", ".json", ".yaml", ".md"}:
                zf.write(f, f.relative_to(path))
    return "submission.zip", buf.getvalue()


def main():
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=os.environ.get("RAG_SERVER", "http://localhost:8000"))
    p.add_argument("--team", default=os.environ.get("RAG_TEAM"))
    p.add_argument("--token", default=os.environ.get("RAG_TOKEN"))
    p.add_argument("--track", required=True)
    p.add_argument("--pipeline", type=Path, required=True, help="pipeline.py, a .zip, or a directory with pipeline.py")
    p.add_argument("--note", required=True, help="your hypothesis: what did you change, what do you expect?")
    p.add_argument("--label", default="", help="short name for this version, e.g. 'hybrid-v2'")
    p.add_argument("--split", default="eval", help="staff only: 'test' evaluates on the public questions")
    p.add_argument("--wait", action="store_true", help="wait for the result and print it")
    args = p.parse_args()
    if not args.team or not args.token:
        sys.exit("set --team/--token or RAG_TEAM/RAG_TOKEN")

    filename, content = pack(args.pipeline)
    r = requests.post(
        f"{args.server}/api/submit",
        data={"team": args.team, "token": args.token, "track": args.track, "note": args.note,
              "label": args.label, "split": args.split},
        files={"file": (filename, content)},
        timeout=60,
    )  # fmt: skip
    if r.status_code != 200:
        sys.exit(f"error {r.status_code}: {r.text}")
    sub = r.json()
    print(f"submitted #{sub['id']}: {args.server}{sub['url']}")
    if not args.wait:
        return
    while True:
        s = requests.get(f"{args.server}/api/submissions/{sub['id']}", timeout=30,
                         headers={"x-team": args.team, "x-token": args.token}).json()
        if s["status"] not in ("queued", "running", "uploading"):
            break
        time.sleep(10)
    print(f"status: {s['status']}")
    if (s["report"] or {}).get("failure"):
        print(f"why: {s['report']['failure']}")
    rep = s["report"] or {}
    if "overall" in rep:
        print("overall: " + "  ".join(f"{k}={v:.3f}" for k, v in rep["overall"].items()))
        for t, row in rep["by_type"].items():
            print(f"  {t:<22} n={row['n']:<4} {rep['primary']}={row.get(rep['primary'], 0):.3f}")
        print(f"usage: {rep['usage']}  timing: {rep['timing']}  server LLM cost: ${rep.get('llm_usd', 0):.4f}")
    if rep.get("llm_refused"):
        print(f"LLM requests were refused: {rep['llm_refused']}")
    if rep.get("error_types"):
        print(f"exceptions raised by your pipeline: {rep['error_types']}")
    if s["log"]:
        print(s["log"][-3000:])
    elif s["status"] in ("failed", "timeout") and "overall" not in rep and not rep.get("failure"):
        print(f"exit code {rep.get('exit_code')}, last exception: {', '.join(rep.get('crash', {})) or 'none found'}. "
              "Logs of hidden-set runs are not shown; run the pipeline locally on the test set to see the error.")


if __name__ == "__main__":
    main()
