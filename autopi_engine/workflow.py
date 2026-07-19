from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agents import settings
from agents.arxiv_utils import multi_search_arxiv
from agents.candidate_factory_ideation import run_candidate_factory_ideation
from agents.development_pack_writer import write_development_pack
from agents.literature_agent import LiteratureHarvester
from agents.openalex_utils import configure_openalex, multi_search_openalex
from agents.research_plan_builder import build_research_plan_from_candidate

from .contracts import (
    AdvanceResult,
    CandidateSelection,
    InputKind,
    Manifest,
    NoveltyAssessment,
    NoveltyDecision,
    PendingAction,
    SearchPlan,
    StageEvent,
    SubmitResult,
    model_dump_jsonable,
)
from .storage import (
    input_path,
    load_manifest,
    manifest_path,
    new_manifest,
    read_json,
    run_lock,
    run_root,
    runs_root,
    save_manifest,
    validate_run_id,
    write_json_atomic,
)

MEMO_HEADINGS = [
    "Research Question",
    "Contribution",
    "Data",
    "Identification Strategy",
    "Related Literature",
    "Empirical Plan",
    "Risks and Limitations",
    "Execution Checklist",
]


def init_run(domain: str, run_id: str | None = None) -> Manifest:
    rid = validate_run_id(run_id or f"run_{uuid.uuid4().hex[:12]}")
    with run_lock(rid):
        if manifest_path(rid).exists():
            raise FileExistsError(f"run already exists: {rid}")
        return new_manifest(rid, domain)


def list_runs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    root = runs_root()
    for path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = Manifest.model_validate(read_json(path))
        except Exception:
            continue
        items.append(
            {
                "run_id": manifest.run_id,
                "domain": manifest.domain,
                "status": manifest.status,
                "current_stage": manifest.current_stage,
                "updated_at": manifest.updated_at,
            }
        )
    return items


def status(run_id: str) -> Manifest:
    return load_manifest(run_id)


def abort(run_id: str) -> Manifest:
    with run_lock(run_id):
        manifest = load_manifest(run_id)
        manifest.status = "aborted"
        manifest.pending_action = None
        manifest.stage_history.append(StageEvent(stage=manifest.current_stage, status="skipped", message="aborted by user"))
        return save_manifest(manifest)


def retry(run_id: str) -> Manifest:
    with run_lock(run_id):
        manifest = load_manifest(run_id)
        if manifest.status != "failed":
            return manifest
        manifest.status = "active"
        manifest.errors.clear()
        return save_manifest(manifest)


def submit(run_id: str, kind: InputKind, payload: dict[str, Any] | str) -> SubmitResult:
    with run_lock(run_id):
        manifest = load_manifest(run_id)
        if manifest.status in {"completed", "completed_with_warnings", "aborted"}:
            raise ValueError(f"run is terminal: {manifest.status}")
        if manifest.pending_action and manifest.pending_action.kind != kind:
            raise ValueError(f"expected {manifest.pending_action.kind}, got {kind}")

        parsed = _parse_input(kind, payload)
        if isinstance(parsed, str):
            stored: dict[str, Any] = {"markdown": parsed}
        else:
            stored = model_dump_jsonable(parsed)
        write_json_atomic(input_path(run_id, kind), stored)

        _apply_submission(manifest, kind, parsed)
        save_manifest(manifest)
        return SubmitResult(
            run_id=run_id,
            status=manifest.status,
            current_stage=manifest.current_stage,
            accepted_kind=kind,
            pending_action=manifest.pending_action,
        )


def advance(run_id: str) -> AdvanceResult:
    completed: list[str] = []
    with run_lock(run_id):
        manifest = load_manifest(run_id)
        while manifest.status == "active":
            stage = manifest.current_stage
            try:
                if stage == "field_scan":
                    _stage_field_scan(manifest)
                elif stage == "candidate_factory":
                    _stage_candidate_factory(manifest)
                elif stage == "novelty_search":
                    _stage_novelty_search(manifest)
                elif stage == "literature_harvest":
                    _stage_literature_harvest(manifest)
                elif stage == "development_pack":
                    _stage_development_pack(manifest)
                elif stage == "complete":
                    _stage_complete(manifest)
                else:
                    raise RuntimeError(f"unknown active stage: {stage}")
                completed.append(stage)
            except Exception as exc:
                manifest.status = "failed"
                manifest.errors.append(f"{stage}: {exc}")
                manifest.stage_history.append(StageEvent(stage=stage, status="failed", message=str(exc)))
                break
        save_manifest(manifest)
        return AdvanceResult(
            run_id=run_id,
            status=manifest.status,
            current_stage=manifest.current_stage,
            completed_stages=completed,
            pending_action=manifest.pending_action,
            artifacts=manifest.artifacts,
        )


def _parse_input(kind: str, payload: dict[str, Any] | str) -> BaseModel | str:
    if kind == "research_memo":
        return str(payload)
    if isinstance(payload, str):
        payload = json.loads(payload)
    mapping = {
        "search_plan": SearchPlan,
        "candidate_selection": CandidateSelection,
        "novelty_search_plan": SearchPlan,
        "novelty_assessment": NoveltyAssessment,
        "novelty_decision": NoveltyDecision,
        "literature_search_plan": SearchPlan,
    }
    model = mapping[kind]
    return model.model_validate(payload)


def _apply_submission(manifest: Manifest, kind: str, parsed: BaseModel | str) -> None:
    manifest.stage_history.append(StageEvent(stage=manifest.current_stage, status="completed", message=f"accepted {kind}", input_kind=kind))
    manifest.pending_action = None
    manifest.status = "active"
    if kind == "search_plan":
        manifest.current_stage = "field_scan"
    elif kind == "candidate_selection":
        selection = parsed
        assert isinstance(selection, CandidateSelection)
        _select_candidate(manifest, selection)
        manifest.current_stage = "novelty_search_plan"
        _await(manifest, "novelty_search_plan", "Draft a SearchPlan JSON for novelty checking the selected candidate.", ["config/research_plan.json"])
    elif kind == "novelty_search_plan":
        manifest.current_stage = "novelty_search"
    elif kind == "novelty_assessment":
        assessment = parsed
        assert isinstance(assessment, NoveltyAssessment)
        _record_novelty_assessment(manifest, assessment)
    elif kind == "novelty_decision":
        decision = parsed
        assert isinstance(decision, NoveltyDecision)
        if decision.action == "abort":
            manifest.status = "aborted"
            manifest.current_stage = "aborted"
        elif decision.action == "reselect":
            manifest.selected_candidate_id = ""
            manifest.current_stage = "candidate_selection"
            _await(manifest, "candidate_selection", "Select a different candidate_id from output/topic_screening.json.", ["output/topic_screening.json"])
        else:
            manifest.current_stage = "literature_search_plan"
            _await(manifest, "literature_search_plan", "Draft a SearchPlan JSON for final literature harvesting.", ["config/research_plan.json", "output/novelty_search_results.json"])
    elif kind == "literature_search_plan":
        _update_literature_queries(manifest)
        manifest.current_stage = "literature_harvest"
    elif kind == "research_memo":
        memo = parsed
        assert isinstance(memo, str)
        _stage_validate_memo(manifest, memo)


def _await(manifest: Manifest, kind: InputKind, prompt: str, refs: list[str] | None = None) -> None:
    manifest.status = "awaiting_input"
    manifest.pending_action = PendingAction(kind=kind, prompt=prompt, reference_files=refs or [])
    manifest.stage_history.append(StageEvent(stage=manifest.current_stage, status="awaiting_input", input_kind=kind, message=prompt))


def _artifact(manifest: Manifest, key: str, path: str | Path) -> None:
    root = run_root(manifest.run_id).resolve()
    p = Path(path).resolve()
    try:
        manifest.artifacts[key] = p.relative_to(root).as_posix()
    except ValueError:
        manifest.artifacts[key] = str(path)


def _run_scoped(manifest: Manifest):
    return settings.activate_run_scope(manifest.run_id)


def _stage_field_scan(manifest: Manifest) -> None:
    plan = SearchPlan.model_validate(read_json(input_path(manifest.run_id, "search_plan")))
    configure_openalex()
    papers, query_hits = multi_search_openalex(
        plan.queries,
        per_query_limit=plan.per_query_limit,
        final_limit=plan.final_limit,
        cache_prefix="field_scan_openalex_query",
    )
    top_results = [
        {
            "title": p.get("title"),
            "citations": p.get("citationCount", 0),
            "year": p.get("year"),
            "doi": p.get("doi"),
            "openalex_id": p.get("openalex_id"),
            "concepts": [c.get("name") for c in p.get("broad_concepts", []) if c.get("name")],
        }
        for p in papers
    ]
    payload = {
        "domain_scanned": manifest.domain,
        "search_plan": plan.model_dump(mode="json"),
        "query_hits": query_hits,
        "openalex_traction": {"top_results": top_results},
        "keywords": {"raw_query": manifest.domain, "high_traction": _top_concepts(top_results)},
    }
    token = _run_scoped(manifest)
    try:
        out = Path(settings.field_scan_path())
        write_json_atomic(out, payload)
    finally:
        settings.deactivate_run_scope(token)
    _artifact(manifest, "field_scan", out)
    manifest.stage_history.append(StageEvent(stage="field_scan", status="completed", artifacts=[manifest.artifacts["field_scan"]]))
    manifest.current_stage = "candidate_factory"


def _stage_candidate_factory(manifest: Manifest) -> None:
    token = _run_scoped(manifest)
    try:
        result = run_candidate_factory_ideation(
            {
                "run_id": manifest.run_id,
                "domain_input": manifest.domain,
                "template_id": manifest.template_id,
                "max_candidates": 40,
                "shortlist_size": 5,
                "speculative_size": 0,
                "enable_experimental": False,
                "technology_options": {"remote_sensing": True, "osmnx": True},
                "cloud_constraints": {"no_paid_api": True, "no_manual_download": True},
            }
        )
    finally:
        settings.deactivate_run_scope(token)
    for key, value in result.items():
        if key.endswith("_path") and value:
            _artifact(manifest, key.removesuffix("_path"), value)
    manifest.stage_history.append(StageEvent(stage="candidate_factory", status="completed", artifacts=list(manifest.artifacts.values())))
    manifest.current_stage = "candidate_selection"
    _await(manifest, "candidate_selection", "Choose one candidate_id from output/topic_screening.json.", ["output/topic_screening.json", "output/candidate_cards.json"])


def _select_candidate(manifest: Manifest, selection: CandidateSelection) -> None:
    screening = read_json(run_root(manifest.run_id) / "output" / "topic_screening.json")
    candidates = (screening or {}).get("candidates", [])
    selected = next((c for c in candidates if c.get("candidate_id") == selection.candidate_id or c.get("topic_id") == selection.candidate_id), None)
    if not selected:
        raise ValueError(f"candidate_id not found in shortlist: {selection.candidate_id}")
    evaluation = selected.get("evaluation") or {}
    plan = build_research_plan_from_candidate(selected, evaluation, manifest.run_id)
    plan_path = run_root(manifest.run_id) / "config" / "research_plan.json"
    write_json_atomic(plan_path, plan.model_dump(mode="json"))
    manifest.selected_candidate_id = selection.candidate_id
    _artifact(manifest, "research_plan", plan_path)


def _stage_novelty_search(manifest: Manifest) -> None:
    plan = SearchPlan.model_validate(read_json(input_path(manifest.run_id, "novelty_search_plan")))
    configure_openalex()
    openalex, query_hits = multi_search_openalex(
        plan.queries,
        per_query_limit=plan.per_query_limit,
        final_limit=plan.final_limit,
        cache_prefix="novelty_openalex_query",
    )
    arxiv = multi_search_arxiv(plan.queries, per_query_limit=min(plan.per_query_limit, 10), final_limit=min(plan.final_limit, 25))
    out = run_root(manifest.run_id) / "output" / "novelty_search_results.json"
    write_json_atomic(out, {"search_plan": plan.model_dump(mode="json"), "query_hits": query_hits, "openalex": openalex, "arxiv": arxiv})
    _artifact(manifest, "novelty_search_results", out)
    manifest.stage_history.append(StageEvent(stage="novelty_search", status="completed", artifacts=[manifest.artifacts["novelty_search_results"]]))
    manifest.current_stage = "novelty_assessment"
    _await(manifest, "novelty_assessment", "Assess novelty as JSON using NoveltyAssessment.", ["config/research_plan.json", "output/novelty_search_results.json"])


def _record_novelty_assessment(manifest: Manifest, assessment: NoveltyAssessment) -> None:
    out = run_root(manifest.run_id) / "output" / "novelty_assessment.json"
    write_json_atomic(out, assessment.model_dump(mode="json"))
    _artifact(manifest, "novelty_assessment", out)
    if assessment.verdict in {"already_published", "unavailable"} or assessment.recommended_action in {"reselect", "abort"}:
        manifest.current_stage = "novelty_decision"
        _await(manifest, "novelty_decision", "Submit {\"action\":\"proceed\"}, {\"action\":\"reselect\"}, or {\"action\":\"abort\"}.", ["output/novelty_assessment.json"])
    else:
        manifest.current_stage = "literature_search_plan"
        _await(manifest, "literature_search_plan", "Draft a SearchPlan JSON for final literature harvesting.", ["config/research_plan.json", "output/novelty_assessment.json"])


def _update_literature_queries(manifest: Manifest) -> None:
    plan = SearchPlan.model_validate(read_json(input_path(manifest.run_id, "literature_search_plan")))
    plan_path = run_root(manifest.run_id) / "config" / "research_plan.json"
    payload = read_json(plan_path)
    payload["literature_queries"] = plan.queries
    write_json_atomic(plan_path, payload)


def _stage_literature_harvest(manifest: Manifest) -> None:
    token = _run_scoped(manifest)
    try:
        result = LiteratureHarvester().run({"current_plan_path": settings.research_plan_path()})
    finally:
        settings.deactivate_run_scope(token)
    if result.get("literature_inventory_path"):
        _artifact(manifest, "literature_inventory", result["literature_inventory_path"])
    _artifact(manifest, "references_bib", run_root(manifest.run_id) / "output" / "references.bib")
    manifest.stage_history.append(StageEvent(stage="literature_harvest", status="completed", artifacts=[manifest.artifacts.get("literature_inventory", "")]))
    manifest.current_stage = "development_pack"


def _stage_development_pack(manifest: Manifest) -> None:
    selected = _load_selected_candidate(manifest)
    pack = write_development_pack(manifest.run_id, selected.get("_raw") or selected)
    _artifact(manifest, "development_pack", pack)
    manifest.stage_history.append(StageEvent(stage="development_pack", status="completed", artifacts=[manifest.artifacts["development_pack"]]))
    manifest.current_stage = "research_memo"
    _await(manifest, "research_memo", "Write the 8-section research memo in Markdown with citation keys from references.bib.", ["config/research_plan.json", "data/literature/index.json", "output/references.bib", manifest.artifacts["development_pack"]])


def _stage_validate_memo(manifest: Manifest, memo: str) -> None:
    errors = validate_memo(manifest.run_id, memo)
    out = run_root(manifest.run_id) / "output" / "research_memo.md"
    out.write_text(memo, encoding="utf-8")
    _artifact(manifest, "research_memo", out)
    if errors:
        manifest.status = "failed"
        manifest.errors.extend(errors)
        manifest.current_stage = "research_memo"
    else:
        manifest.current_stage = "complete"
        manifest.status = "active"


def _stage_complete(manifest: Manifest) -> None:
    index = {
        "run_id": manifest.run_id,
        "domain": manifest.domain,
        "selected_candidate_id": manifest.selected_candidate_id,
        "artifacts": manifest.artifacts,
    }
    out = run_root(manifest.run_id) / "output" / "run_index.json"
    write_json_atomic(out, index)
    _artifact(manifest, "run_index", out)
    manifest.stage_history.append(StageEvent(stage="complete", status="completed", artifacts=[manifest.artifacts["run_index"]]))
    manifest.status = "completed_with_warnings" if manifest.warnings else "completed"
    manifest.pending_action = None


def validate_memo(run_id: str, memo: str) -> list[str]:
    errors: list[str] = []
    for heading in MEMO_HEADINGS:
        if not re.search(rf"^##\s+{re.escape(heading)}\s*$", memo, flags=re.MULTILINE):
            errors.append(f"missing required heading: {heading}")
    bib = run_root(run_id) / "output" / "references.bib"
    citation_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib.read_text(encoding="utf-8") if bib.exists() else ""))
    used = set(re.findall(r"@([A-Za-z0-9_:-]+)", memo))
    missing = sorted(used - citation_keys)
    if missing:
        errors.append(f"memo cites keys not present in references.bib: {', '.join(missing)}")
    if citation_keys and not used:
        errors.append("memo must cite at least one key from references.bib")
    return errors


def _load_selected_candidate(manifest: Manifest) -> dict[str, Any]:
    cards = read_json(run_root(manifest.run_id) / "output" / "candidate_cards.json")
    selected = next((c for c in cards if c.get("candidate_id") == manifest.selected_candidate_id), None)
    if not selected:
        screening = read_json(run_root(manifest.run_id) / "output" / "topic_screening.json")
        selected = next((c for c in screening.get("candidates", []) if c.get("candidate_id") == manifest.selected_candidate_id), None)
    if not selected:
        raise ValueError(f"selected candidate not found: {manifest.selected_candidate_id}")
    return selected


def _top_concepts(results: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in results:
        for concept in row.get("concepts") or []:
            counts[str(concept)] = counts.get(str(concept), 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:15]]
