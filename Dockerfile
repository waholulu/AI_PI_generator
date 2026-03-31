# Auto-PI Research Engine
# Python 3.10 (matches requires-python in pyproject.toml)

FROM python:3.10-slim

WORKDIR /app

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /usr/local/bin/uv

# Install dependencies first (layer cache)
COPY pyproject.toml ./
# Generate lockfile and install — no committed uv.lock needed
RUN uv sync --no-dev

# Copy application source
COPY agents/ agents/
COPY api/ api/
COPY prompts/ prompts/
COPY main.py ./

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Explicitly start the API server — NOT main.py
# $PORT is injected by Railway; fallback to 8000 for local Docker runs
CMD ["sh", "-c", "uv run uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
