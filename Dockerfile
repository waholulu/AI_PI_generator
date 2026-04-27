# Auto-PI Research Engine
FROM python:3.11-slim

WORKDIR /app

# Install dependencies from pyproject (cloud default keeps image lean)
COPY . .
RUN pip install --no-cache-dir ".[geospatial]" && uvicorn --version

EXPOSE 8000

CMD ["sh", "-c", "echo PORT=$PORT && python -c 'from api.server import app; print(\"import OK\")' && exec uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
