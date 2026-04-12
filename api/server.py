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
    ApproveRequest,
    ApproveResponse,
    HealthResponse,
    LogEntry,
    Milestone,
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

def _apply_idea_selection(run_id: str, thread_id: str, idea_index: int) -> Optional[str]:
    """Promote candidate at *idea_index* to rank-1 and persist the updated files.

    Reads ``topic_screening.json`` from the run's scoped output directory,
    rotates the selected candidate to position 0, re-assigns ranks, and
    updates ``research_context.json``'s ``selected_topic`` accordingly.

    Returns the title of the newly selected idea, or None if the index is invalid.
    """
    import json as _json

    scope_token = settings.activate_run_scope(run_id)
    try:
        screening_path = settings.topic_screening_path()
        context_path = settings.research_context_path()
        plan_path = settings.research_plan_path()

        if not Path(screening_path).exists():
            return None

        with open(screening_path, "r", encoding="utf-8") as f:
            screening = _json.load(f)

        candidates: list = screening.get("candidates", [])
        if idea_index < 0 or idea_index >= len(candidates):
            return None

        # Rotate selected candidate to front
        selected = candidates.pop(idea_index)
        candidates.insert(0, selected)
        for i, c in enumerate(candidates):
            c["rank"] = i + 1
        screening["candidates"] = candidates

        with open(screening_path, "w", encoding="utf-8") as f:
            _json.dump(screening, f, indent=2, ensure_ascii=False)

        selected_title: str = selected.get("title", "")

        # Update research_context.json
        if Path(context_path).exists():
            try:
                with open(context_path, "r", encoding="utf-8") as f:
                    ctx = _json.load(f)
                if isinstance(ctx, dict):
                    ctx["selected_topic"] = {
                        "title": selected_title,
                        "score": selected.get("final_score", selected.get("initial_score")),
                        "quantitative_specs": selected.get("quantitative_specs", {}),
                        "data_sources": selected.get("data_sources", []),
                        "publishability": selected.get("publishability", ""),
                        "selection_overridden": True,
                    }
                    with open(context_path, "w", encoding="utf-8") as f:
                        _json.dump(ctx, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.warning("Could not update research_context.json after idea selection: %s", exc)

        # Update research_plan.json title to match the new top-1
        if Path(plan_path).exists():
            try:
                with open(plan_path, "r", encoding="utf-8") as f:
                    plan = _json.load(f)
                plan["project_title"] = selected_title
                plan["topic_screening"] = {"top_candidate_title": selected_title, "manually_selected": True}
                with open(plan_path, "w", encoding="utf-8") as f:
                    _json.dump(plan, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.warning("Could not update research_plan.json after idea selection: %s", exc)

        return selected_title
    finally:
        settings.deactivate_run_scope(scope_token)


async def _run_pipeline(run_id: str, thread_id: str, domain_input: str) -> None:
    """Run the pipeline phases in the background, handling HITL interrupt."""
    log_store.set_current_run(run_id)
    scope_token = settings.activate_run_scope(run_id)
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"domain_input": domain_input, "execution_status": "starting"}

    def _stream_phase1() -> Optional[str]:
        """Synchronous streaming phase; returns first pending node if HITL, else None."""
        for output in graph.stream(initial_state, config):
            for node_key in output:
                run_manager.update_status(run_id, "running", current_node=node_key)
                logger.info("Node completed: %s", node_key)
                run_manager.record_milestone(run_id, "node_completed", f"完成节点: {node_key}")
        snapshot = graph.get_state(config)
        if snapshot.next:
            return list(snapshot.next)[0]
        return None

    try:
        run_manager.update_status(run_id, "running", current_node="field_scanner")
        run_manager.record_milestone(run_id, "pipeline_started", f"研究方向: {domain_input}")

        # graph.stream() uses synchronous HTTP (requests/pyalex); run in a thread
        # so the event loop stays free to handle polling requests from the UI.
        pending_node = await asyncio.to_thread(_stream_phase1)

        if pending_node:
            logger.info("Pipeline paused at HITL checkpoint. Pending: %s", pending_node)
            run_manager.update_status(run_id, "awaiting_approval", current_node=pending_node)
            run_manager.record_milestone(run_id, "hitl_paused", "等待人工审批，请在状态页面批准或终止")
        else:
            run_manager.update_status(run_id, "completed")
            run_manager.record_milestone(run_id, "completed", "流水线执行完毕")
            logger.info("Pipeline completed (no HITL interrupt).")

    except asyncio.CancelledError:
        run_manager.update_status(run_id, "aborted")
        logger.info("Run %s was cancelled.", run_id)
    except Exception as exc:
        run_manager.update_status(run_id, "failed", error=str(exc))
        run_manager.record_milestone(run_id, "failed", str(exc)[:300])
        logger.error("Run %s failed: %s", run_id, exc)
    finally:
        settings.deactivate_run_scope(scope_token)
        log_store.set_current_run(None)


async def _regenerate_and_pause(run_id: str, thread_id: str) -> None:
    """Re-run ideation + idea_validator outside the graph, then pause again.

    The graph state at this point still has ``literature`` as the next node
    (the LangGraph checkpoint is untouched).  The ideation + validator agents
    overwrite the files on disk (topic_screening.json, idea_validation.json,
    research_plan.json, research_context.json), so when the client finally
    calls /approve with action='select' the resumed literature node picks up
    the new content via the existing file paths in ResearchState.
    """
    from agents.hitl_helpers import regenerate_topics

    log_store.set_current_run(run_id)
    scope_token = settings.activate_run_scope(run_id)
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    def _do_regenerate() -> None:
        snapshot = graph.get_state(config)
        state = dict(snapshot.values) if snapshot and snapshot.values else {}
        regenerate_topics(state)

    try:
        await asyncio.to_thread(_do_regenerate)
        run_manager.update_status(run_id, "awaiting_approval", current_node="literature")
        run_manager.record_milestone(
            run_id, "topics_regenerated", "已根据用户反馈重新生成候选选题"
        )
        logger.info("Run %s: topic regeneration complete.", run_id)
    except asyncio.CancelledError:
        run_manager.update_status(run_id, "aborted")
        logger.info("Run %s: regeneration was cancelled.", run_id)
    except Exception as exc:
        run_manager.update_status(run_id, "failed", error=str(exc))
        run_manager.record_milestone(run_id, "failed", f"重新生成选题失败: {str(exc)[:300]}")
        logger.exception("Run %s: regeneration failed: %s", run_id, exc)
    finally:
        settings.deactivate_run_scope(scope_token)
        log_store.set_current_run(None)


async def _resume_pipeline(run_id: str, thread_id: str) -> None:
    """Resume a HITL-paused pipeline."""
    log_store.set_current_run(run_id)
    scope_token = settings.activate_run_scope(run_id)
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    def _stream_phase2() -> None:
        for output in graph.stream(None, config):
            for node_key in output:
                run_manager.update_status(run_id, "running", current_node=node_key)
                logger.info("Node completed: %s", node_key)
                run_manager.record_milestone(run_id, "node_completed", f"完成节点: {node_key}")

    try:
        run_manager.update_status(run_id, "running", current_node="literature")
        run_manager.record_milestone(run_id, "approved", "已批准，继续执行文献收集 → 初稿撰写 → 数据获取")

        await asyncio.to_thread(_stream_phase2)

        run_manager.update_status(run_id, "completed")
        run_manager.record_milestone(run_id, "completed", "流水线执行完毕")
        logger.info("Run %s completed successfully.", run_id)

    except asyncio.CancelledError:
        run_manager.update_status(run_id, "aborted")
        logger.info("Run %s was cancelled during resume.", run_id)
    except Exception as exc:
        run_manager.update_status(run_id, "failed", error=str(exc))
        run_manager.record_milestone(run_id, "failed", str(exc)[:300])
        logger.error("Run %s failed during resume: %s", run_id, exc)
    finally:
        settings.deactivate_run_scope(scope_token)
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
    # Augment with degraded_nodes from the LangGraph checkpoint state, if available.
    try:
        graph = _get_graph()
        snapshot = graph.get_state({"configurable": {"thread_id": run.thread_id}})
        degraded = snapshot.values.get("degraded_nodes") or []
    except Exception:
        degraded = []
    run_dict = run.model_dump()
    run_dict["degraded_nodes"] = degraded
    return RunStatus(**run_dict)


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


@app.get("/runs/{run_id}/milestones", response_model=List[Milestone])
async def get_milestones(run_id: str):
    """Return the key-operation milestone log for this run."""
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return run_manager.get_milestones(run_id)


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


def _load_current_topics(run_id: str) -> list:
    """Load the current candidate list from this run's topic_screening.json."""
    import json as _json

    scope_token = settings.activate_run_scope(run_id)
    try:
        screening_path = settings.topic_screening_path()
        if not Path(screening_path).exists():
            return []
        try:
            with open(screening_path, "r", encoding="utf-8") as f:
                return _json.load(f).get("candidates", [])
        except (OSError, _json.JSONDecodeError):
            return []
    finally:
        settings.deactivate_run_scope(scope_token)


@app.post("/runs/{run_id}/approve", response_model=ApproveResponse)
async def approve_hitl(run_id: str, req: ApproveRequest = ApproveRequest()):
    """Handle the HITL checkpoint.

    Two actions are supported (via ``req.action``):
      * ``select`` (default) — promote ``selected_idea_index`` to rank-1 and
        resume the pipeline from the literature stage.
      * ``regenerate`` — record the current topics as rejected_by_user in
        memory/graveyard and re-run ideation + idea_validator outside the
        graph, then pause at the HITL checkpoint again with fresh topics.
    """
    from agents.hitl_helpers import record_rejected_topics, MAX_REGENERATION_ROUNDS

    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    if run.status != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run.status}', not 'awaiting_approval'.",
        )

    # ── Branch: regenerate ───────────────────────────────────────────
    if req.action == "regenerate":
        if run.regeneration_round >= MAX_REGENERATION_ROUNDS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Maximum regeneration rounds ({MAX_REGENERATION_ROUNDS}) reached. "
                    f"Please select one of the current topics with action='select'."
                ),
            )

        # Record current topics as rejected in memory + graveyard
        current_topics = await asyncio.to_thread(_load_current_topics, run_id)
        new_round = run_manager.increment_regeneration_round(run_id)

        def _record() -> None:
            scope_token = settings.activate_run_scope(run_id)
            try:
                record_rejected_topics(current_topics, run.domain_input, new_round)
            finally:
                settings.deactivate_run_scope(scope_token)

        await asyncio.to_thread(_record)

        run_manager.update_status(run_id, "regenerating", current_node="ideation")
        run_manager.record_milestone(
            run_id, "topics_rejected",
            f"用户拒绝所有候选，重新构思（round {new_round}）",
        )
        logger.info("HITL regenerate for run %s (round %d)", run_id, new_round)

        task = asyncio.create_task(_regenerate_and_pause(run_id, run.thread_id))
        _tasks[run_id] = task

        return ApproveResponse(
            run_id=run_id,
            status="regenerating",
            message=f"Regenerating topics (round {new_round})...",
            regeneration_round=new_round,
        )

    # ── Branch: select ───────────────────────────────────────────────
    if req.selected_idea_index is None:
        raise HTTPException(
            status_code=422,
            detail="selected_idea_index is required when action='select'.",
        )

    selected_title = await asyncio.to_thread(
        _apply_idea_selection, run_id, run.thread_id, req.selected_idea_index
    )
    if selected_title is None:
        raise HTTPException(
            status_code=422,
            detail=f"selected_idea_index {req.selected_idea_index} is out of range or screening file missing.",
        )
    logger.info(
        "HITL: idea %d selected for run %s: %s",
        req.selected_idea_index, run_id, selected_title,
    )

    logger.info("HITL approved for run %s. Resuming...", run_id)
    task = asyncio.create_task(_resume_pipeline(run_id, run.thread_id))
    _tasks[run_id] = task

    msg = f"Pipeline resumed with selected idea: {selected_title!r}"
    return ApproveResponse(
        run_id=run_id,
        status="running",
        message=msg,
        selected_idea=selected_title,
    )


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
    """List output/config/data files for a run."""
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    run_root = settings.run_root(run_id, create=False)
    if run_root.exists():
        root = run_root
        scan_dirs = [run_root / "output", run_root / "config", run_root / "data"]
    else:
        # Backward compatibility for historical non-scoped runs.
        root = settings._root()
        scan_dirs = [settings.output_dir(), settings.config_dir(), settings.data_dir()]
    files: list[OutputFile] = []

    for scan_dir in scan_dirs:
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

    run_root = settings.run_root(run_id, create=False)
    root = run_root if run_root.exists() else settings._root()
    file_path = (root / filename).resolve()

    # Path traversal protection
    if not file_path.is_relative_to(root.resolve()):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename!r} not found.")

    return FileResponse(path=str(file_path), filename=file_path.name)
