"""
Centralized path and configuration management for Auto-PI.

All file paths are rooted under AUTOPI_DATA_ROOT (defaults to "." for
backward compatibility with local development). Cloud deployments set
this to a persistent volume mount (e.g., /app/storage).
"""

import os
from contextvars import ContextVar, Token
from pathlib import Path


_RUN_SCOPE_ID: ContextVar[str | None] = ContextVar("autopi_run_scope_id", default=None)


def _base_root() -> Path:
    return Path(os.getenv("AUTOPI_DATA_ROOT", "."))


def _root() -> Path:
    """Backward-compatible alias to the global data root."""
    return _base_root()


def current_run_scope() -> str | None:
    return _RUN_SCOPE_ID.get()


def activate_run_scope(run_id: str) -> Token:
    """Activate run-scoped artifact paths for the current context."""
    return _RUN_SCOPE_ID.set(run_id)


def deactivate_run_scope(token: Token) -> None:
    _RUN_SCOPE_ID.reset(token)


def run_root(run_id: str, create: bool = True) -> Path:
    d = _base_root() / "runs" / run_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _scoped_root() -> Path:
    run_id = current_run_scope()
    if run_id:
        return run_root(run_id)
    return _base_root()


# ── Directory helpers ─────────────────────────────────────────────────

def global_output_dir() -> Path:
    d = _base_root() / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def global_config_dir() -> Path:
    d = _base_root() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def global_data_dir() -> Path:
    d = _base_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir() -> Path:
    d = _scoped_root() / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_dir() -> Path:
    d = _scoped_root() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    d = _scoped_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def memory_dir() -> Path:
    # Keep memory global so historical reuse spans runs.
    d = _base_root() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = global_data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prompts_dir() -> Path:
    """Prompts are baked into the repo, not the data root."""
    return Path("prompts")


# ── Specific file paths ──────────────────────────────────────────────

# Output files
def field_scan_path() -> str:
    return str(output_dir() / "field_scan.json")


def topic_screening_path() -> str:
    return str(output_dir() / "topic_screening.json")


def ideas_graveyard_path(domain: str | None = None) -> str:
    """Return the graveyard file path, namespaced by domain to prevent cross-run contamination.

    Each unique domain gets its own graveyard so that rejected ideas from one
    research area do not pollute ideation for an unrelated area. Concurrent runs
    on the same domain will still share a single file (by design: they should
    avoid the same dead ends). Pass domain=None to get the legacy global path.
    """
    if domain:
        import hashlib
        domain_key = hashlib.md5(domain.lower().strip().encode()).hexdigest()[:8]
        return str(global_output_dir() / f"ideas_graveyard_{domain_key}.json")
    # Legacy / fallback: global graveyard used when domain is unknown.
    return str(global_output_dir() / "ideas_graveyard.json")


def topic_ranking_path() -> str:
    return str(output_dir() / "topic_ranking.csv")


def research_context_path() -> str:
    return str(output_dir() / "research_context.json")


def draft_path() -> str:
    return str(output_dir() / "Draft_v1.md")


def references_bib_path() -> str:
    return str(output_dir() / "references.bib")


def run_index_path() -> str:
    return str(output_dir() / "run_index.json")


def checkpoints_db_path() -> str:
    # Keep checkpoints global for stable graph compilation in API process.
    return str(global_output_dir() / "checkpoints.sqlite")


# Config files
def research_plan_path() -> str:
    return str(config_dir() / "research_plan.json")


# Data files
def literature_dir() -> Path:
    d = data_dir() / "literature"
    d.mkdir(parents=True, exist_ok=True)
    return d


def literature_cards_dir() -> Path:
    d = literature_dir() / "cards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def literature_pdfs_dir() -> Path:
    d = literature_dir() / "pdfs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def literature_index_path() -> str:
    return str(literature_dir() / "index.json")


def raw_data_dir() -> Path:
    d = data_dir() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_manifest_path() -> str:
    return str(raw_data_dir() / "manifest.json")


# Memory files
def idea_memory_csv_path() -> str:
    return str(memory_dir() / "idea_memory.csv")


def enriched_top_candidates_path() -> str:
    return str(memory_dir() / "enriched_top_candidates.jsonl")


def idea_validation_path() -> str:
    return str(output_dir() / "idea_validation.json")


def data_source_registry_path() -> str:
    return str(prompts_dir() / "data_source_registry.json")


# Prompts
def academic_drafter_prompt_path() -> str:
    return str(prompts_dir() / "academic_drafter.txt")


# ── Database URL ─────────────────────────────────────────────────────

def get_db_url() -> str:
    """
    Returns DATABASE_URL for PostgreSQL (cloud) or a SQLite path (local).
    If DATABASE_URL is set, use it directly. Otherwise fall back to SQLite.
    """
    return os.getenv("DATABASE_URL", f"sqlite:///{checkpoints_db_path()}")


def is_postgres() -> bool:
    return get_db_url().startswith("postgres")
