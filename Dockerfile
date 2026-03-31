# Auto-PI Research Engine
FROM python:3.10-slim

WORKDIR /app

# Install dependencies (pip — simple, reliable)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "python-dotenv>=1.2.2" \
    "pyalex>=0.21" \
    "langgraph>=1.0.10" \
    "langchain-google-genai>=4.2.1" \
    "langchain-anthropic>=1.3.4" \
    "langgraph-checkpoint-sqlite>=3.0.3" \
    "pydantic>=2.0" \
    "tenacity>=9.1.4" \
    "requests>=2.32.5" \
 && uvicorn --version

# Copy application source
COPY agents/ agents/
COPY api/ api/
COPY prompts/ prompts/
COPY ui/ ui/
COPY main.py ./

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
