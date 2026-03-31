"""
Auto-PI FastAPI server.

Exposes the LangGraph pipeline as a REST API so it can be managed
from the cloud or via Claude Code without an interactive terminal.

Run locally:
    uvicorn api.server:app --reload

Endpoints:
  POST   /runs                          Start a new pipeline run
  GET    /runs                          List all runs
  GET    /runs/{run_id}/status          Get run status
  GET    /runs/{run_id}/logs            Get logs for a run
  GET    /runs/{run_id}/state           Get LangGraph state for a run
  POST   /runs/{run_id}/approve         Approve HITL checkpoint and continue
  POST   /runs/{run_id}/reject          Reject HITL checkpoint and abort
  GET    /runs/{run_id}/outputs         List output files
  GET    /runs/{run_id}/outputs/{name}  Download an output file
  GET    /health                        Health check
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from agents import settings
from agents.logging_config import setup_logging, get_logger

from api import log_store, run_manager
from api.models import (
    ApproveResponse,
    HealthResponse,
    LogEntry,
    OutputFile,
    OutputsResponse,
    RunListItem,
    RunStatus,
    StartRunRequest,
)

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), json_format=True)
log_store.install_handler()
logger = get_logger(__name__)

# ── In-memory task registry (run_id → asyncio.Task) ──────────────────
_tasks: dict[str, asyncio.Task] = {}

# ── Lazy graph cache ──────────────────────────────────────────────────
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from agents.orchestrator import build_orchestrator
        _graph = build_orchestrator()
    return _graph


# ── App ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Auto-PI API starting up.")
    yield
    logger.info("Auto-PI API shutting down.")


app = FastAPI(
    title="Auto-PI Research Engine",
    description="REST API for the Auto-PI multi-agent research pipeline.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Static UI ─────────────────────────────────────────────────────────────────
_UI_DIR = Path(__file__).parent.parent / "ui" / "static"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    index = _UI_DIR / "index.html"
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


# ── Background pipeline execution ────────────────────────────────────

async def _run_pipeline(run_id: str, thread_id: str, domain_input: str) -> None:
    """Run the pipeline phases in the background, handling HITL interrupt."""
    log_store.set_current_run(run_id)
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"domain_input": domain_input, "execution_status": "starting"}

    try:
        run_manager.update_status(run_id, "running", current_node="field_scanner")

        # Phase 1: run until HITL interrupt (after ideation, before literature)
        for output in graph.stream(initial_state, config):
            for node_key in output:
                run_manager.update_status(run_id, "running", current_node=node_key)
                logger.info("Node completed: %s", node_key)

        # Check if paused at HITL
        snapshot = graph.get_state(config)
        if snapshot.next:
            pending = list(snapshot.next)
            logger.info("Pipeline paused at HITL checkpoint. Pending: %s", pending)
            run_manager.update_status(run_id, "awaiting_approval", current_node=pending[0])
        else:
            run_manager.update_status(run_id, "completed")
            logger.info("Pipeline completed (no HITL interrupt).")

    except asyncio.CancelledError:
        run_manager.update_status(run_id, "aborted")
        logger.info("Run %s was cancelled.", run_id)
    except Exception as exc:
        run_manager.update_status(run_id, "failed", error=str(exc))
        logger.error("Run %s failed: %s", run_id, exc)
    finally:
        log_store.set_current_run(None)


async def _resume_pipeline(run_id: str, thread_id: str) -> None:
    """Resume a HITL-paused pipeline."""
    log_store.set_current_run(run_id)
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        run_manager.update_status(run_id, "running", current_node="literature")

        for output in graph.stream(None, config):
            for node_key in output:
                run_manager.update_status(run_id, "running", current_node=node_key)
                logger.info("Node completed: %s", node_key)

        run_manager.update_status(run_id, "completed")
        logger.info("Run %s completed successfully.", run_id)

    except asyncio.CancelledError:
        run_manager.update_status(run_id, "aborted")
        logger.info("Run %s was cancelled during resume.", run_id)
    except Exception as exc:
        run_manager.update_status(run_id, "failed", error=str(exc))
        logger.error("Run %s failed during resume: %s", run_id, exc)
    finally:
        log_store.set_current_run(None)


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/runs", response_model=RunStatus, status_code=202)
async def start_run(req: StartRunRequest, background_tasks: BackgroundTasks):
    """Start a new pipeline run. Returns immediately with run_id."""
    run = run_manager.create_run(req.domain_input, thread_id=req.thread_id)
    logger.info("Starting run %s for domain: %s", run.run_id, req.domain_input)
    task = asyncio.create_task(
        _run_pipeline(run.run_id, run.thread_id, req.domain_input)
    )
    _tasks[run.run_id] = task
    return run


@app.get("/runs", response_model=List[RunListItem])
async def list_runs():
    return run_manager.list_runs()


@app.get("/runs/{run_id}/status", response_model=RunStatus)
async def get_status(run_id: str):
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return run


@app.get("/runs/{run_id}/logs", response_model=List[LogEntry])
async def get_logs(
    run_id: str,
    level: Optional[str] = None,
    limit: int = 500,
):
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return log_store.get_logs(run_id, level=level, limit=limit)


@app.get("/runs/{run_id}/state")
async def get_state(run_id: str):
    """Return the current LangGraph checkpoint state for this run."""
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    graph = _get_graph()
    config = {"configurable": {"thread_id": run.thread_id}}
    try:
        snapshot = graph.get_state(config)
        return {
            "values": dict(snapshot.values),
            "next": list(snapshot.next),
            "metadata": snapshot.metadata,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/runs/{run_id}/approve", response_model=ApproveResponse)
async def approve_hitl(run_id: str):
    """Approve the HITL checkpoint. Resumes the pipeline from literature stage."""
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    if run.status != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run.status}', not 'awaiting_approval'.",
        )

    logger.info("HITL approved for run %s. Resuming...", run_id)
    task = asyncio.create_task(_resume_pipeline(run_id, run.thread_id))
    _tasks[run_id] = task

    return ApproveResponse(run_id=run_id, status="running", message="Pipeline resumed.")


@app.post("/runs/{run_id}/reject", response_model=ApproveResponse)
async def reject_hitl(run_id: str):
    """Reject the HITL checkpoint. Aborts the run."""
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    if run.status != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run.status}', not 'awaiting_approval'.",
        )

    run_manager.update_status(run_id, "aborted")
    logger.info("HITL rejected for run %s. Run aborted.", run_id)
    return ApproveResponse(run_id=run_id, status="aborted", message="Run aborted by user.")


_BINARY_EXTENSIONS = {".sqlite", ".parquet", ".pdf", ".db"}


@app.get("/runs/{run_id}/outputs", response_model=OutputsResponse)
async def list_outputs(run_id: str):
    """List all output files for a run (output/, config/, data/ directories)."""
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    root = settings._root()
    files: list[OutputFile] = []

    for scan_dir in [settings.output_dir(), settings.config_dir(), settings.data_dir()]:
        if not scan_dir.exists():
            continue
        for path in sorted(scan_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in _BINARY_EXTENSIONS:
                continue
            rel = path.relative_to(root)
            files.append(OutputFile(
                filename=str(rel),
                path=str(path),
                size_bytes=path.stat().st_size,
            ))

    return OutputsResponse(run_id=run_id, files=files)


@app.get("/runs/{run_id}/outputs/{filename:path}")
async def download_output(run_id: str, filename: str):
    """Download or preview a specific output file."""
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    root = settings._root()
    file_path = (root / filename).resolve()

    # Path traversal protection
    if not file_path.is_relative_to(root.resolve()):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename!r} not found.")

    return FileResponse(path=str(file_path), filename=file_path.name)
