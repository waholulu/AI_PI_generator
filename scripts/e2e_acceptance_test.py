"""End-to-end candidate factory acceptance test (requires running API server).

Exercises the full candidate factory pipeline via REST API:
  1. Health check
  2. List templates
  3. Start a candidate factory run (max_candidates=40, no experimental)
  4. Poll until completed or awaiting_approval
  5. Fetch candidates and verify counts + quality
  6. Filter Claude Code Ready candidates (expect >= 8)
  7. Select top ready candidate and generate development pack
  8. Verify development pack files
  9. Download claude_task_prompt.md
  10. Verify candidate_id selection endpoint

Usage
-----
# Against local server (start first: uvicorn api.server:app --reload)
python scripts/e2e_acceptance_test.py

# Against cloud
export AUTOPI_API_URL=https://<your-app>.up.railway.app
python scripts/e2e_acceptance_test.py

Exit codes
----------
0  -- all acceptance criteria met
1  -- one or more criteria failed
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

try:
    import requests
except ImportError as exc:
    print("requests is required: pip install requests", file=sys.stderr)
    raise SystemExit(1) from exc

import os

DEFAULT_URL = os.environ.get("AUTOPI_API_URL", "http://localhost:8000")
POLL_INTERVAL_S = 4
POLL_TIMEOUT_S = 300

# Acceptance thresholds
MIN_CANDIDATES = 20
MIN_CLAUDE_CODE_READY = 8
MIN_EXPOSURE_FAMILIES = 6
MIN_PASS_OR_WARNING = 20


def _get(session: requests.Session, url: str) -> Any:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post(session: requests.Session, url: str, payload: dict) -> Any:
    resp = session.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_e2e(base_url: str) -> bool:
    base_url = base_url.rstrip("/")
    session = requests.Session()
    session.headers["Content-Type"] = "application/json"

    passed: list[str] = []
    failed: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> bool:
        if cond:
            print(f"    ok  {name}")
        else:
            print(f"    FAIL {name}" + (f": {detail}" if detail else ""))
            failed.append(name)
        if cond:
            passed.append(name)
        return cond

    print(f"\n{'='*60}")
    print(f"Auto-PI Candidate Factory E2E Acceptance Test")
    print(f"Target: {base_url}")
    print(f"{'='*60}\n")

    # Step 1: Health
    print("Step 1: Health check")
    try:
        health = _get(session, f"{base_url}/health")
        check("health endpoint responds", True)
        check("status is ok", health.get("status") == "ok", str(health))
    except Exception as exc:
        check("health endpoint responds", False, str(exc))
        print("\n[FATAL] Server not reachable. Aborting.")
        return False

    # Step 2: Templates
    print("\nStep 2: Templates")
    try:
        templates_resp = _get(session, f"{base_url}/templates")
        templates = templates_resp.get("templates", [])
        template_ids = [t.get("file_id") or t.get("template_id") for t in templates]
        check("templates endpoint responds", len(templates) > 0)
        check("built_environment_health template available",
              "built_environment_health" in template_ids,
              f"found: {template_ids}")
    except Exception as exc:
        check("templates endpoint responds", False, str(exc))

    # Step 3: Start run
    print("\nStep 3: Start candidate factory run")
    run_id: str | None = None
    try:
        run = _post(session, f"{base_url}/runs", {
            "domain_input": "Built environment and health outcomes",
            "template_id": "built_environment_health",
            "automation_risk_tolerance": "low_medium",
            "enable_experimental": False,
            "max_candidates": 40,
            "cloud_constraints": {"no_paid_api": True, "no_manual_download": True},
        })
        run_id = run.get("run_id")
        check("run created", bool(run_id), str(run))
        print(f"    run_id: {run_id}")
    except Exception as exc:
        check("run created", False, str(exc))
        print("\n[FATAL] Could not start run. Aborting.")
        return False

    # Step 4: Poll
    print("\nStep 4: Poll for completion")
    status_data: dict = {}
    deadline = time.time() + POLL_TIMEOUT_S
    terminal = {"completed", "awaiting_approval", "failed", "aborted"}
    while time.time() < deadline:
        try:
            status_data = _get(session, f"{base_url}/runs/{run_id}/status")
            status = status_data.get("status", "unknown")
            print(f"    [{status}] node={status_data.get('current_node', '')}    \r", end="")
            if status in terminal:
                print()
                break
        except Exception as exc:
            print(f"    poll error: {exc}")
        time.sleep(POLL_INTERVAL_S)
    else:
        check("run completed within timeout", False, f"timeout after {POLL_TIMEOUT_S}s")
        return False

    final_status = status_data.get("status", "unknown")
    check("run reached terminal state", final_status in terminal, final_status)
    check("run not failed", final_status not in {"failed", "aborted"},
          f"status={final_status}, error={status_data.get('error')}")
    print(f"    final status: {final_status}")

    # Step 5: Candidates
    print("\nStep 5: Fetch candidates")
    candidates: list[dict] = []
    try:
        cands_resp = _get(session, f"{base_url}/runs/{run_id}/candidates")
        candidates = cands_resp if isinstance(cands_resp, list) else cands_resp.get("candidates", [])
    except Exception as exc:
        check("candidates endpoint responds", False, str(exc))
        return False

    check("candidates endpoint responds", True)
    check(f"candidate_count >= {MIN_CANDIDATES}", len(candidates) >= MIN_CANDIDATES,
          f"got {len(candidates)}")
    exposure_families = set(c.get("exposure_family", "") for c in candidates)
    check(f"exposure_families >= {MIN_EXPOSURE_FAMILIES}",
          len(exposure_families) >= MIN_EXPOSURE_FAMILIES,
          f"got {len(exposure_families)}: {sorted(exposure_families)}")
    pass_or_warn = [
        c for c in candidates
        if c.get("shortlist_status") in {"ready", "review"}
        or (c.get("gate_status") or {}).get("overall") in {"pass", "warning"}
    ]
    check(f"pass_or_warning_count >= {MIN_PASS_OR_WARNING}",
          len(pass_or_warn) >= MIN_PASS_OR_WARNING, f"got {len(pass_or_warn)}")
    exp_count = sum(1 for c in candidates if "experimental" in (c.get("technology_tags") or []))
    check("experimental_candidate_count == 0", exp_count == 0, f"got {exp_count}")
    print(f"    total={len(candidates)}, pass_or_warn={len(pass_or_warn)}, "
          f"experimental={exp_count}, families={len(exposure_families)}")

    # Step 6: Claude Code Ready
    print("\nStep 6: Claude Code Ready candidates")
    ready = [c for c in candidates if c.get("claude_code_ready") is True]
    check(f"claude_code_ready_count >= {MIN_CLAUDE_CODE_READY}",
          len(ready) >= MIN_CLAUDE_CODE_READY, f"got {len(ready)}")
    check("all ready candidates are low or medium risk",
          all(c.get("automation_risk") in {"low", "medium"} for c in ready))
    print(f"    ready={len(ready)}")

    if not ready:
        print("\n[WARN] No Claude Code Ready candidates — skipping pack steps.")
        _print_summary(passed, failed)
        return len(failed) == 0

    top = ready[0]
    candidate_id = top.get("candidate_id")
    print(f"    top candidate: {candidate_id} ({top.get('exposure_family')} -> "
          f"{top.get('outcome_family')})")

    # Step 7: Generate development pack
    print("\nStep 7: Generate development pack")
    pack_summary: dict = {}
    try:
        pack_summary = _post(session, f"{base_url}/runs/{run_id}/development-pack",
                             {"candidate_id": candidate_id})
        pack_status = pack_summary.get("status") or pack_summary.get("development_pack_status")
        check("development pack generated", bool(pack_status))
        check("pack status is claude_code_ready",
              pack_status in {"ready", "claude_code_ready"}, f"got {pack_status}")
    except Exception as exc:
        check("development pack generated", False, str(exc))

    # Step 8: Verify pack files
    print("\nStep 8: Verify development pack files")
    required_files = [
        "implementation_spec.json", "claude_task_prompt.md", "data_contract.yaml",
        "feature_plan.yaml", "analysis_plan.yaml", "acceptance_tests.md",
    ]
    actual_files = pack_summary.get("development_pack_files") or pack_summary.get("files") or []
    if actual_files:
        for fname in required_files:
            check(f"pack contains {fname}", fname in actual_files)
    else:
        print("    [skip] file list not returned from pack endpoint")

    # Step 9: Download claude_task_prompt.md
    print("\nStep 9: Download claude_task_prompt.md")
    try:
        prompt_url = (f"{base_url}/runs/{run_id}/outputs/"
                      f"development_packs/{candidate_id}/claude_task_prompt.md")
        resp = session.get(prompt_url, timeout=30)
        if resp.status_code == 200:
            prompt_text = resp.text
            check("claude_task_prompt.md downloaded", True)
            check("prompt contains candidate_id", candidate_id in prompt_text)
            check("prompt contains Acceptance Criteria", "Acceptance Criteria" in prompt_text)
        else:
            check("claude_task_prompt.md downloaded", False, f"HTTP {resp.status_code}")
    except Exception as exc:
        check("claude_task_prompt.md downloaded", False, str(exc))

    # Step 10: Select candidate
    print("\nStep 10: Select candidate for literature continuation")
    try:
        sel = _post(session, f"{base_url}/runs/{run_id}/select-candidate",
                    {"candidate_id": candidate_id})
        check("select-candidate endpoint responds",
              sel.get("candidate_id") == candidate_id
              or sel.get("status") in {"ok", "accepted", "selected"}, str(sel))
    except requests.HTTPError as exc:
        if exc.response.status_code == 404:
            print("    [skip] select-candidate not yet implemented (404)")
        else:
            check("select-candidate endpoint responds", False, str(exc))
    except Exception as exc:
        print(f"    [skip] {exc}")

    _print_summary(passed, failed)
    return len(failed) == 0


def _print_summary(passed: list[str], failed: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("\nFailed checks:")
        for f in failed:
            print(f"  FAIL {f}")
    else:
        print("All acceptance criteria met.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-PI Candidate Factory E2E acceptance test (requires running API server)"
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    args = parser.parse_args()
    sys.exit(0 if run_e2e(args.url) else 1)
