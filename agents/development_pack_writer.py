from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents import settings
from agents.implementation_spec_builder import build_implementation_spec
from models.candidate_composer_schema import ComposedCandidate


def _to_candidate(candidate: dict[str, Any]) -> ComposedCandidate:
    return ComposedCandidate(**candidate)


def _claude_task_prompt(candidate: ComposedCandidate, implementation_spec: dict[str, Any]) -> str:
    return (
        f"You are implementing an automated geospatial research pipeline for candidate {candidate.candidate_id}.\n\n"
        "Goal:\n"
        f"Build a cloud-safe Python pipeline that estimates the association between {candidate.exposure_family} "
        f"and {candidate.outcome_family} at {candidate.unit_of_analysis} level.\n\n"
        "Required outputs:\n"
        "1. data/processed/tract_features.csv\n"
        "2. output/tables/model_summary.csv\n"
        "3. output/report/technical_summary.md\n\n"
        "Constraints:\n"
        "1. Do not require manual downloads.\n"
        "2. Use a smoke test geography first.\n"
        "3. Keep runtime under 10 minutes for smoke test.\n"
        "4. If remote data calls fail, write clear fallback logs.\n"
        "5. Do not use paid APIs unless explicitly enabled.\n"
        "6. Do not store raw street view images.\n\n"
        f"Method template: {candidate.method_template}\n"
        f"Automation risk: {candidate.automation_risk}\n"
        f"Technology tags: {', '.join(candidate.technology_tags) or 'none'}\n"
    )


def write_development_pack(run_id: str, candidate_payload: dict[str, Any]) -> Path:
    token = settings.activate_run_scope(run_id)
    try:
        candidate = _to_candidate(candidate_payload)
        implementation_spec = build_implementation_spec(candidate).model_dump()

        pack_dir = settings.development_packs_dir() / candidate.candidate_id
        pack_dir.mkdir(parents=True, exist_ok=True)

        (pack_dir / "README.md").write_text(
            f"# Development Pack: {candidate.candidate_id}\n\n"
            f"Exposure: {candidate.exposure_family}\n"
            f"Outcome: {candidate.outcome_family}\n"
            f"Method: {candidate.method_template}\n",
            encoding="utf-8",
        )

        (pack_dir / "implementation_spec.json").write_text(
            json.dumps(implementation_spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pack_dir / "data_contract.yaml").write_text(
            "input_tables:\n"
            "  - exposure\n"
            "  - outcome\n"
            "  - controls\n"
            "join_key: GEOID\n",
            encoding="utf-8",
        )
        (pack_dir / "feature_plan.yaml").write_text(
            "features:\n" + "\n".join(f"  - {x}" for x in candidate.exposure_variables),
            encoding="utf-8",
        )
        (pack_dir / "analysis_plan.yaml").write_text(
            f"method: {candidate.method_template}\n"
            f"outcome: {candidate.outcome_family}\n",
            encoding="utf-8",
        )
        (pack_dir / "acceptance_tests.md").write_text(
            "- pytest passes\n"
            "- smoke test writes non-empty tract_features.csv\n"
            "- output report includes identification threats\n",
            encoding="utf-8",
        )
        (pack_dir / "claude_task_prompt.md").write_text(
            _claude_task_prompt(candidate, implementation_spec), encoding="utf-8"
        )

        # keep candidate-scoped implementation_spec copy as phase requirement
        candidate_dir = settings.candidates_dir() / candidate.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "implementation_spec.json").write_text(
            json.dumps(implementation_spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return pack_dir
    finally:
        settings.deactivate_run_scope(token)
