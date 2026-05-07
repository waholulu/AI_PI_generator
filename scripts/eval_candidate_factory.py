"""End-to-end evaluation of the candidate factory pipeline.

Usage
-----
python scripts/eval_candidate_factory.py \
  --template built_environment_health \
  --domain "Built environment and health outcomes" \
  --max-candidates 40 \
  --enable-experimental false \
  --output output/eval_candidate_factory.json

Thresholds (v1, enforced by test_eval_candidate_factory.py)
-----------------------------------------------------------
candidate_count                    >= 20
score_completion_rate              >= 0.95
implementation_spec_completion_rate >= 0.85
development_pack_ready_rate        >= 0.80
low_or_medium_automation_risk_rate >= 0.70
experimental_candidate_count       == 0  (when --enable-experimental false)

Data catalog thresholds (v2, enforced when --check-thresholds)
--------------------------------------------------------------
source_profile_completion_rate     >= 0.90
variable_mapping_completion_rate   >= 0.75
join_recipe_completion_rate        >= 0.85
source_aware_pack_rate             >= 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.candidate_composer import compose_candidates
from agents.candidate_export_validator import validate_candidate_export_contract
from agents.candidate_feasibility import precheck_candidate
from agents.candidate_flag_classifier import compute_candidate_readiness
from agents.candidate_repair import repair_candidate
from agents.final_ranker import rank_candidates, score_candidate
from agents.implementation_spec_builder import build_implementation_spec
from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposeRequest


def _str_to_bool(val: str) -> bool:
    return val.strip().lower() in {"true", "1", "yes"}


def evaluate(
    template_id: str,
    domain: str,
    max_candidates: int = 40,
    enable_experimental: bool = False,
    enable_tier2: bool = True,
    no_paid_api: bool = True,
) -> dict:
    """Run the full production pipeline (same steps as candidate_factory_ideation.py)."""
    req = ComposeRequest(
        template_id=template_id,
        domain_input=domain,
        max_candidates=max_candidates,
        enable_experimental=enable_experimental,
        enable_tier2=enable_tier2,
        no_paid_api=no_paid_api,
    )
    candidates = compose_candidates(req)
    registry = SourceRegistry.load()

    pass_count = 0
    spec_ready = 0
    spec_complete = 0
    pack_ready = 0
    low_or_medium = 0
    score_completed = 0
    experimental_count = 0
    ready_count = 0
    blocked_count = 0
    repair_success_count = 0
    repair_attempted_count = 0
    claude_code_ready_count = 0

    # Data catalog metrics
    source_profile_count = 0
    variable_mapping_count = 0
    join_recipe_count = 0
    source_aware_spec_count = 0

    top_candidate_ids: list[str] = []
    cards: list[dict] = []

    for c in candidates:
        # Step 1: precheck (role-based G3)
        gate = precheck_candidate(c)
        # Step 2: deterministic repair
        repaired, gate, history = repair_candidate(c, gate)
        # Step 3: strict export validation (matches production)
        gate = validate_candidate_export_contract(repaired, gate, no_paid_api=no_paid_api)
        # Step 4: user-facing readiness classification
        readiness_summary = compute_candidate_readiness(repaired.model_dump(), gate, history)
        # Step 5: weighted score
        scores = score_candidate(repaired.model_dump(), gate, history)
        readiness = readiness_summary.get("readiness", "blocked")
        shortlist_status = gate.get("shortlist_status", "blocked")

        if gate.get("overall") in {"pass", "warning"}:
            pass_count += 1

        if history:
            repair_attempted_count += 1
            if gate.get("overall") in {"pass", "warning"}:
                repair_success_count += 1

        if scores.get("overall", 0) > 0:
            score_completed += 1

        # Implementation spec completeness
        if (
            repaired.exposure_source
            and repaired.outcome_source
            and repaired.method_template
            and repaired.exposure_variables
        ):
            spec_complete += 1

        # Development pack pre-qualification: would this candidate be claude_code_ready
        # if a pack were generated?  (eval runs without writing files, so we check conditions.)
        is_pack_prequalified = (
            repaired.exposure_source
            and repaired.outcome_source
            and repaired.method_template
            and gate.get("overall") in {"pass", "warning"}
            and shortlist_status != "blocked"
            and repaired.automation_risk in {"low", "medium"}
            and not repaired.required_secrets
            and not any(t in {"experimental", "streetview_cv", "deep_learning", "satellite_cv"}
                        for t in (repaired.technology_tags or []))
        )
        if is_pack_prequalified:
            pack_ready += 1
            claude_code_ready_count += 1

        # Implementation spec: at minimum, source + method available
        if repaired.exposure_source and repaired.outcome_source and repaired.method_template:
            spec_ready += 1

        if repaired.automation_risk in {"low", "medium"}:
            low_or_medium += 1

        if "experimental" in repaired.technology_tags:
            experimental_count += 1

        if readiness in {"ready", "ready_after_auto_fix"}:
            ready_count += 1
        if readiness == "blocked" or shortlist_status == "blocked":
            blocked_count += 1

        # Data catalog metrics
        exp_profile = registry.get_profile(repaired.exposure_source)
        if exp_profile is not None:
            source_profile_count += 1

        if registry.has_variable_mapping(repaired.exposure_source, repaired.exposure_family):
            variable_mapping_count += 1

        has_missing_recipe = any(
            r == "missing_join_recipe" for r in gate.get("reasons", [])
        )
        if not has_missing_recipe:
            join_recipe_count += 1

        try:
            spec_obj = build_implementation_spec(repaired)
            if spec_obj.source_use_specs:
                source_aware_spec_count += 1
        except Exception:
            pass

        cards.append(
            {
                "candidate_id": repaired.candidate_id,
                "overall": scores.get("overall", 0),
                "readiness": readiness,
                "shortlist_status": shortlist_status,
                "automation_risk": repaired.automation_risk,
                "technology_tags": repaired.technology_tags,
                "scores": scores,
                "gate_status": gate,
                "repair_history": history,
                "readiness_summary": readiness_summary,
            }
        )

    total = max(len(candidates), 1)

    # Rank full pool (same as production)
    ranked_cards = rank_candidates(cards)

    # Shortlist: top non-blocked candidates (production contract)
    shortlist = [
        c for c in ranked_cards
        if c.get("readiness") in {"ready", "ready_after_auto_fix", "needs_review"}
    ][:5]
    shortlist_blocked_count = sum(1 for c in shortlist if c.get("readiness") == "blocked")

    top_candidate_ids = [c["candidate_id"] for c in ranked_cards[:5]]

    repair_success_rate = (
        round(repair_success_count / repair_attempted_count, 3)
        if repair_attempted_count > 0
        else None
    )

    return {
        "template_id": template_id,
        "domain": domain,
        "enable_experimental": enable_experimental,
        "candidate_count": len(candidates),
        "candidate_pass_rate": round(pass_count / total, 3),
        "repair_success_rate": repair_success_rate,
        "score_completion_rate": round(score_completed / total, 3),
        "implementation_spec_completion_rate": round(spec_complete / total, 3),
        "development_pack_ready_rate": round(pack_ready / total, 3),
        "low_or_medium_automation_risk_rate": round(low_or_medium / total, 3),
        "experimental_candidate_count": experimental_count,
        "ready_candidate_count": ready_count,
        "blocked_candidate_count": blocked_count,
        "claude_code_ready_count": claude_code_ready_count,
        "shortlist_count": len(shortlist),
        "shortlist_blocked_count": shortlist_blocked_count,
        "top_5_candidate_ids": top_candidate_ids,
        # Data catalog metrics
        "source_profile_completion_rate": round(source_profile_count / total, 3),
        "variable_mapping_completion_rate": round(variable_mapping_count / total, 3),
        "join_recipe_completion_rate": round(join_recipe_count / total, 3),
        "source_aware_pack_rate": round(source_aware_spec_count / total, 3),
    }


def _check_thresholds(result: dict) -> list[str]:
    """Return list of threshold violations (empty = all pass)."""
    failures: list[str] = []

    if result["candidate_count"] < 20:
        failures.append(f"candidate_count {result['candidate_count']} < 20")

    if result["score_completion_rate"] < 0.95:
        failures.append(f"score_completion_rate {result['score_completion_rate']:.2%} < 95%")

    if result["implementation_spec_completion_rate"] < 0.85:
        failures.append(
            f"implementation_spec_completion_rate "
            f"{result['implementation_spec_completion_rate']:.2%} < 85%"
        )

    if result["development_pack_ready_rate"] < 0.80:
        failures.append(
            f"development_pack_ready_rate {result['development_pack_ready_rate']:.2%} < 80%"
        )

    if result["low_or_medium_automation_risk_rate"] < 0.70:
        failures.append(
            f"low_or_medium_automation_risk_rate "
            f"{result['low_or_medium_automation_risk_rate']:.2%} < 70%"
        )

    if not result["enable_experimental"] and result["experimental_candidate_count"] != 0:
        failures.append(
            f"experimental_candidate_count {result['experimental_candidate_count']} != 0 "
            f"(experimental disabled)"
        )

    # E2E contract: shortlist must have ≥5 candidates and zero blocked
    if result.get("shortlist_count", 0) < 5:
        failures.append(
            f"shortlist_count {result.get('shortlist_count', 0)} < 5"
        )

    if result.get("shortlist_blocked_count", 0) != 0:
        failures.append(
            f"shortlist_blocked_count {result.get('shortlist_blocked_count', 0)} != 0 "
            f"(blocked candidates must not appear in shortlist)"
        )

    if result.get("claude_code_ready_count", 0) < 8:
        failures.append(
            f"claude_code_ready_count {result.get('claude_code_ready_count', 0)} < 8"
        )

    # Data catalog thresholds
    if result.get("source_profile_completion_rate", 1.0) < 0.90:
        failures.append(
            f"source_profile_completion_rate "
            f"{result['source_profile_completion_rate']:.2%} < 90%"
        )

    if result.get("variable_mapping_completion_rate", 1.0) < 0.75:
        failures.append(
            f"variable_mapping_completion_rate "
            f"{result['variable_mapping_completion_rate']:.2%} < 75%"
        )

    if result.get("join_recipe_completion_rate", 1.0) < 0.85:
        failures.append(
            f"join_recipe_completion_rate "
            f"{result['join_recipe_completion_rate']:.2%} < 85%"
        )

    if result.get("source_aware_pack_rate", 1.0) < 0.85:
        failures.append(
            f"source_aware_pack_rate "
            f"{result['source_aware_pack_rate']:.2%} < 85%"
        )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate factory pipeline metrics.")
    parser.add_argument("--template", default="built_environment_health")
    parser.add_argument("--domain", default="Built environment and health outcomes")
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--enable-experimental", default="false")
    parser.add_argument("--enable-tier2", default="true")
    parser.add_argument("--no-paid-api", default="true")
    parser.add_argument("--output", default=None, help="Write JSON result to this path")
    parser.add_argument("--check-thresholds", action="store_true", help="Exit 1 if any threshold fails")
    args = parser.parse_args()

    result = evaluate(
        template_id=args.template,
        domain=args.domain,
        max_candidates=args.max_candidates,
        enable_experimental=_str_to_bool(args.enable_experimental),
        enable_tier2=_str_to_bool(args.enable_tier2),
        no_paid_api=_str_to_bool(args.no_paid_api),
    )

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"\nResult written to {out_path}", file=sys.stderr)

    if args.check_thresholds:
        failures = _check_thresholds(result)
        if failures:
            print("\nThreshold failures:", file=sys.stderr)
            for f in failures:
                print(f"  ✗ {f}", file=sys.stderr)
            sys.exit(1)
        else:
            print("\nAll thresholds passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
