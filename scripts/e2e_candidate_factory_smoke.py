"""End-to-end smoke test for the candidate factory pipeline (no running server required).

Runs the full factory loop — compose → precheck → repair → score → pack — for a
fixed template and validates that the resulting artefacts satisfy minimum contracts.

Usage
-----
python scripts/e2e_candidate_factory_smoke.py
python scripts/e2e_candidate_factory_smoke.py --output output/e2e_candidate_factory_report.json

Exit codes
----------
0  All checks pass
1  One or more checks failed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEMPLATE_ID = "built_environment_health"
DOMAIN = "Built environment and health outcomes"
TOP_N_PACKS = 3

REQUIRED_PACK_FILES = [
    "README.md",
    "implementation_spec.json",
    "data_contract.yaml",
    "feature_plan.yaml",
    "analysis_plan.yaml",
    "acceptance_tests.md",
    "claude_task_prompt.md",
]

MIN_CLAUDE_TASK_PROMPT_CHARS = 500
MIN_ACCEPTANCE_TESTS_LINES = 3


def _check(name: str, passed: bool, detail: str = "") -> dict:
    status = "pass" if passed else "fail"
    entry: dict = {"check": name, "status": status}
    if detail:
        entry["detail"] = detail
    return entry


def run_smoke(tmp_dir: Path) -> dict:
    # Import after setting env so settings picks up the tmp root
    from agents.candidate_composer import compose_candidates
    from agents.candidate_feasibility import precheck_candidate
    from agents.candidate_repair import repair_candidate
    from agents.candidate_output_writer import (
        write_feasibility_report,
        write_development_pack_index,
        write_gate_trace,
    )
    from agents.development_pack_writer import write_development_pack
    from agents.final_ranker import rank_candidates, score_candidate
    from agents.implementation_spec_builder import build_implementation_spec
    from models.candidate_composer_schema import ComposeRequest
    import agents.settings as _settings

    checks: list[dict] = []
    t0 = time.time()

    # ── 1. Compose ────────────────────────────────────────────────────────────
    req = ComposeRequest(
        template_id=TEMPLATE_ID,
        domain_input=DOMAIN,
        max_candidates=40,
        enable_experimental=False,
        enable_tier2=True,
        no_paid_api=True,
    )
    candidates = compose_candidates(req)
    checks.append(_check(
        "compose_candidates",
        len(candidates) >= 20,
        f"got {len(candidates)} candidates",
    ))
    checks.append(_check(
        "no_experimental_candidates",
        all("experimental" not in c.technology_tags for c in candidates),
        "experimental tags present" if any(
            "experimental" in c.technology_tags for c in candidates
        ) else "",
    ))

    # ── 2. Precheck + Repair + Score ─────────────────────────────────────────
    cards: list[dict] = []
    for c in candidates:
        gate = precheck_candidate(c)
        repaired, gate, history = repair_candidate(c, gate)
        scores = score_candidate(repaired.model_dump(), gate, history)
        cards.append({
            "candidate_id": repaired.candidate_id,
            "candidate": repaired.model_dump(),
            "gate_status": gate,
            "repair_history": history,
            "scores": scores,
            "shortlist_status": gate.get("shortlist_status", "blocked"),
            "automation_risk": repaired.automation_risk,
            "technology_tags": list(repaired.technology_tags),
        })

    pass_count = sum(1 for c in cards if c["gate_status"].get("overall") in {"pass", "warning"})
    pass_rate = pass_count / max(len(cards), 1)
    checks.append(_check(
        "candidate_pass_rate_gte_60pct",
        pass_rate >= 0.60,
        f"{pass_rate:.1%} ({pass_count}/{len(cards)})",
    ))

    ready_cards = [c for c in cards if c["shortlist_status"] == "ready"]
    checks.append(_check(
        "at_least_10_ready_candidates",
        len(ready_cards) >= 10,
        f"{len(ready_cards)} ready candidates",
    ))

    low_med = sum(1 for c in cards if c["automation_risk"] in {"low", "medium"})
    checks.append(_check(
        "low_or_medium_automation_risk_gte_70pct",
        low_med / max(len(cards), 1) >= 0.70,
        f"{low_med / max(len(cards), 1):.1%}",
    ))

    # ── 3. Score + Rank ───────────────────────────────────────────────────────
    ranked = rank_candidates(cards)
    checks.append(_check(
        "all_candidates_scored",
        all(c["scores"].get("overall", 0) > 0 for c in ranked),
        "",
    ))
    unique_novelty = {round(c["scores"].get("novelty", 0), 2) for c in ranked}
    checks.append(_check(
        "novelty_not_all_identical",
        len(unique_novelty) > 1,
        f"unique novelty values: {sorted(unique_novelty)}",
    ))

    top_ready = [c for c in ranked if c["shortlist_status"] == "ready"][:TOP_N_PACKS]
    checks.append(_check(
        f"at_least_{TOP_N_PACKS}_ready_for_packs",
        len(top_ready) >= TOP_N_PACKS,
        f"only {len(top_ready)} ready candidates",
    ))

    # ── 4. Output writer ──────────────────────────────────────────────────────
    run_id = "smoke_e2e_run"
    token = _settings.activate_run_scope(run_id)
    try:
        write_feasibility_report(run_id, cards)
        write_development_pack_index(run_id, cards)
        write_gate_trace(run_id, cards)
        out_dir = _settings.output_dir()
        for fname in ("feasibility_report.json", "development_pack_index.json", "gate_trace.json"):
            fpath = out_dir / fname
            checks.append(_check(
                f"output_writer_{fname}",
                fpath.exists() and fpath.stat().st_size > 10,
                str(fpath),
            ))
    except Exception as exc:
        checks.append(_check("output_writer_smoke", False, str(exc)))
    finally:
        _settings.deactivate_run_scope(token)

    # ── 5. Development Pack generation ───────────────────────────────────────
    for card in top_ready:
        cid = card["candidate_id"]
        try:
            pack_dir = write_development_pack(run_id, card["candidate"])
        except Exception as exc:
            checks.append(_check(f"pack_write_{cid}", False, str(exc)))
            continue

        for fname in REQUIRED_PACK_FILES:
            fpath = pack_dir / fname
            checks.append(_check(
                f"pack_file_exists:{cid}/{fname}",
                fpath.exists() and fpath.stat().st_size > 0,
                f"missing or empty" if not (fpath.exists() and fpath.stat().st_size > 0) else "",
            ))

        # Claude task prompt quality
        prompt_path = pack_dir / "claude_task_prompt.md"
        if prompt_path.exists():
            content = prompt_path.read_text(encoding="utf-8")
            checks.append(_check(
                f"claude_task_prompt_length:{cid}",
                len(content) >= MIN_CLAUDE_TASK_PROMPT_CHARS,
                f"{len(content)} chars (min {MIN_CLAUDE_TASK_PROMPT_CHARS})",
            ))

        # Acceptance tests quality
        tests_path = pack_dir / "acceptance_tests.md"
        if tests_path.exists():
            lines = [ln for ln in tests_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            checks.append(_check(
                f"acceptance_tests_lines:{cid}",
                len(lines) >= MIN_ACCEPTANCE_TESTS_LINES,
                f"{len(lines)} non-empty lines (min {MIN_ACCEPTANCE_TESTS_LINES})",
            ))

        # implementation_spec.json valid JSON with required sections
        spec_path = pack_dir / "implementation_spec.json"
        if spec_path.exists():
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                has_acq = bool(spec.get("data_acquisition_steps"))
                has_feat = bool(spec.get("feature_engineering_steps"))
                has_analysis = bool(spec.get("analysis_steps"))
                checks.append(_check(
                    f"implementation_spec_complete:{cid}",
                    has_acq and has_feat and has_analysis,
                    f"acq={has_acq} feat={has_feat} analysis={has_analysis}",
                ))
                roles = {s.get("source_role") for s in spec.get("data_acquisition_steps", [])}
                checks.append(_check(
                    f"spec_has_control_or_boundary:{cid}",
                    bool(roles & {"control", "boundary"}),
                    f"roles present: {sorted(roles)}",
                ))
            except json.JSONDecodeError as exc:
                checks.append(_check(f"implementation_spec_valid_json:{cid}", False, str(exc)))

        # data_contract.yaml parseable with inputs key
        contract_path = pack_dir / "data_contract.yaml"
        if contract_path.exists():
            try:
                import yaml
                contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
                checks.append(_check(
                    f"data_contract_parseable:{cid}",
                    isinstance(contract, dict) and "inputs" in contract,
                    f"keys: {list((contract or {}).keys())}",
                ))
            except Exception as exc:
                checks.append(_check(f"data_contract_parseable:{cid}", False, str(exc)))

    elapsed = round(time.time() - t0, 1)
    total = len(checks)
    passed = sum(1 for c in checks if c["status"] == "pass")
    failed = total - passed

    return {
        "template_id": TEMPLATE_ID,
        "elapsed_seconds": elapsed,
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "all_passed": failed == 0,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E smoke test for the candidate factory.")
    parser.add_argument("--output", default="output/e2e_candidate_factory_report.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Point AUTOPI_DATA_ROOT at a throwaway directory so smoke test
        # never pollutes the real output/ tree.
        os.environ["AUTOPI_DATA_ROOT"] = tmp_dir

        result = run_smoke(Path(tmp_dir))

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"\nReport written to {out_path}", file=sys.stderr)

    if result["failed"] > 0:
        print(f"\n{result['failed']} check(s) FAILED:", file=sys.stderr)
        for c in result["checks"]:
            if c["status"] == "fail":
                detail = f" — {c['detail']}" if c.get("detail") else ""
                print(f"  ✗ {c['check']}{detail}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {result['passed']} checks passed in {result['elapsed_seconds']}s.", file=sys.stderr)


if __name__ == "__main__":
    main()
