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
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.candidate_composer import compose_candidates
from agents.candidate_feasibility import precheck_candidate
from agents.candidate_repair import repair_candidate
from agents.final_ranker import score_candidate
from agents.implementation_spec_builder import build_implementation_spec
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
    req = ComposeRequest(
        template_id=template_id,
        domain_input=domain,
        max_candidates=max_candidates,
        enable_experimental=enable_experimental,
        enable_tier2=enable_tier2,
        no_paid_api=no_paid_api,
    )
    candidates = compose_candidates(req)

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
    top_candidate_ids: list[str] = []
    scored_cards: list[dict] = []

    for c in candidates:
        gate = precheck_candidate(c)
        repaired, gate, history = repair_candidate(c, gate)

        if gate.get("overall") in {"pass", "warning"}:
            pass_count += 1

        if history:
            repair_attempted_count += 1
            if gate.get("overall") in {"pass", "warning"}:
                repair_success_count += 1

        scores = score_candidate(repaired.model_dump(), gate, history)

        if scores.get("overall", 0) > 0:
            score_completed += 1

        # Implementation spec completeness: can we build a non-trivial spec?
        if (
            repaired.exposure_source
            and repaired.outcome_source
            and repaired.method_template
            and repaired.exposure_variables
        ):
            spec_complete += 1

        # Development pack readiness: spec + gate pass + low/medium risk
        is_pack_ready = (
            repaired.exposure_source
            and repaired.outcome_source
            and repaired.method_template
            and gate.get("overall") in {"pass", "warning"}
            and repaired.automation_risk in {"low", "medium"}
        )
        if is_pack_ready:
            pack_ready += 1

        # Implementation spec: at minimum, source + method available
        if repaired.exposure_source and repaired.outcome_source and repaired.method_template:
            spec_ready += 1

        if repaired.automation_risk in {"low", "medium"}:
            low_or_medium += 1

        if "experimental" in repaired.technology_tags:
            experimental_count += 1

        shortlist = gate.get("shortlist_status", "blocked")
        if shortlist == "ready":
            ready_count += 1
        elif shortlist == "blocked":
            blocked_count += 1

        scored_cards.append(
            {
                "candidate_id": repaired.candidate_id,
                "overall": scores.get("overall", 0),
                "shortlist_status": shortlist,
                "automation_risk": repaired.automation_risk,
                "technology_tags": repaired.technology_tags,
                "scores": scores,
            }
        )

    total = max(len(candidates), 1)
    scored_cards.sort(key=lambda c: -c["overall"])
    top_candidate_ids = [c["candidate_id"] for c in scored_cards[:5]]

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
        "top_5_candidate_ids": top_candidate_ids,
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
