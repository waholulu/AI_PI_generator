# Auto-PI Research Engine
# Python 3.10 (matches requires-python in pyproject.toml)

FROM python:3.10-slim

WORKDIR /app

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /usr/local/bin/uv

# Install dependencies first (layer cache)
COPY pyproject.toml uv.lock ./
RUN uv venv /app/.venv \
 && . /app/.venv/bin/activate \
 && uv pip install -r pyproject.toml \
 && ls -la /app/.venv/bin/uvicorn \
 && echo "uvicorn found OK"
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY agents/ agents/
COPY api/ api/
COPY prompts/ prompts/
COPY ui/ ui/
COPY main.py ./

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# $PORT is injected by Railway; fallback to 8000 for local Docker runs
CMD ["sh", "-c", "/app/.venv/bin/uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
