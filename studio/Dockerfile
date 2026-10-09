# The shared, read-only RAG Studio (README: "A shared, read-only RAG Studio").
#
# Build it from the STUDENT BUNDLE, never from the instructor repo (which contains the solutions):
#
#   python scripts/make_student_repo.py ../rag-workshop      # in the instructor repo
#   cd ../rag-workshop
#   docker build -f studio/Dockerfile -t rag-studio .
#   docker run -d --name rag-studio --restart unless-stopped -p 127.0.0.1:8765:8765 \
#       -e RAG_STUDIO_PASSWORD='<workshop password>' rag-studio
#
# Then put HTTPS in front of 127.0.0.1:8765 (e.g. Caddy). The image has no API key and needs none: the shared
# Studio never runs pipelines or calls an LLM. No torch either, so it stays small.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ragkit + Studio's web server only (no sentence-transformers / torch: nothing is embedded here)
COPY pyproject.toml ./
COPY ragkit ragkit
RUN pip install -e ".[studio]"

COPY studio studio
COPY tracks tracks
COPY START_HERE.md README.md ./

RUN useradd --create-home --uid 10001 studio
USER studio
ENV RAG_STUDIO_PUBLIC=1 RAG_STUDIO_DIR=/tmp/rag-studio HF_HUB_OFFLINE=1

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/login', timeout=4)"
CMD ["python", "-m", "studio", "--public", "--host", "0.0.0.0", "--port", "8765"]
