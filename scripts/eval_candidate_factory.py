from __future__ import annotations

import json

from agents.candidate_composer import compose_candidates
from agents.candidate_feasibility import precheck_candidate
from agents.candidate_repair import repair_candidate
from agents.final_ranker import score_candidate
from models.candidate_composer_schema import ComposeRequest


def main() -> None:
    req = ComposeRequest(
        template_id="built_environment_health",
        domain_input="Built environment and health",
        max_candidates=40,
        enable_experimental=False,
    )
    candidates = compose_candidates(req)

    pass_count = 0
    spec_ready = 0
    low_or_medium = 0
    score_completed = 0

    for c in candidates:
        gate = precheck_candidate(c)
        repaired, gate, history = repair_candidate(c, gate)
        scores = score_candidate(repaired.model_dump(), gate, history)

        if gate.get("overall") in {"pass", "warning"}:
            pass_count += 1
        if repaired.exposure_source and repaired.outcome_source and repaired.method_template:
            spec_ready += 1
        if repaired.automation_risk in {"low", "medium"}:
            low_or_medium += 1
        if scores.get("overall", 0) > 0:
            score_completed += 1

    total = max(len(candidates), 1)
    payload = {
        "candidate_count": len(candidates),
        "candidate_pass_rate": round(pass_count / total, 3),
        "development_pack_ready_rate": round(spec_ready / total, 3),
        "low_or_medium_automation_risk_rate": round(low_or_medium / total, 3),
        "score_completion_rate": round(score_completed / total, 3),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
