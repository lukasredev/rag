"""python -m studio [--port 8765] [--pipeline my_pipeline.py ...]

Shared, read-only deployment for a workshop (guide, starter code and corpus; nothing runs):

    RAG_STUDIO_PASSWORD=... python -m studio --public --host 0.0.0.0 --port 8765

Put HTTPS in front of it (e.g. Caddy), since the password is sent with the login form.
"""

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def main():
    p = argparse.ArgumentParser(description="RAG Studio: see what your pipeline does, question by question")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="anything but localhost needs --public")
    p.add_argument("--public", action="store_true",
                   help="shared read-only Studio behind a password (RAG_STUDIO_PASSWORD): guide, starter code, corpus")
    p.add_argument("--pipeline", type=Path, action="append", default=[],
                   help="extra pipeline file (or folder with pipeline.py) to list; repeatable")
    args = p.parse_args()
    if args.public:
        os.environ["RAG_STUDIO_PUBLIC"] = "1"
    from ragkit import config

    from . import app as studio

    if studio.PUBLIC and len(config.get("RAG_STUDIO_PASSWORD") or "") < 8:
        sys.exit("--public needs a password of at least 8 characters in RAG_STUDIO_PASSWORD (env or .env)")
    if not studio.PUBLIC and args.host not in ("127.0.0.1", "localhost", "::1"):
        # the normal Studio runs pipeline code and shows files: never on the network
        sys.exit(f"refusing to listen on {args.host}: use --public for a shared Studio")
    for path in args.pipeline:
        path = path.resolve()
        studio.EXTRA_PIPELINES.append(path / "pipeline.py" if path.is_dir() else path)
    mode = "shared read-only Studio" if studio.PUBLIC else "RAG Studio"
    print(f"{mode} on http://{args.host}:{args.port}")
    uvicorn.run(studio.app, host=args.host, port=args.port, log_level="warning", proxy_headers=True)


if __name__ == "__main__":
    main()
