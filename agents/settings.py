"""
Centralized path and configuration management for Auto-PI.

All file paths are rooted under AUTOPI_DATA_ROOT (defaults to "." for
backward compatibility with local development). Cloud deployments set
this to a persistent volume mount (e.g., /app/storage).
"""

import os
from pathlib import Path


def _root() -> Path:
    return Path(os.getenv("AUTOPI_DATA_ROOT", "."))


# ── Directory helpers ─────────────────────────────────────────────────

def output_dir() -> Path:
    d = _root() / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_dir() -> Path:
    d = _root() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    d = _root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def memory_dir() -> Path:
    d = _root() / "memory"
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


def ideas_graveyard_path() -> str:
    return str(output_dir() / "ideas_graveyard.json")


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
    return str(output_dir() / "checkpoints.sqlite")


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
