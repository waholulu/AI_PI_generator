"""Cloud smoke test for the AutoPI REST API.

Validates the candidate factory → development pack pipeline end-to-end.
Does NOT test literature, drafter, or data_fetcher (those require LLM keys).

Usage
-----
export AUTOPI_API_URL=https://<your-app>.up.railway.app
python scripts/cloud_smoke_test.py

Or against a local server:
python scripts/cloud_smoke_test.py --url http://localhost:8000

Exit codes
----------
0  — all checks passed
1  — one or more checks failed
"""
from __future__ import annotations

import argparse
import json
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
POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 120


def _get(session: requests.Session, url: str) -> dict[str, Any]:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post(session: requests.Session, url: str, payload: dict) -> dict[str, Any]:
    resp = session.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}" + (f": {detail}" if detail else ""))
    return condition


def run_smoke_test(base_url: str) -> bool:
    base_url = base_url.rstrip("/")
    session = requests.Session()
    session.headers["Content-Type"] = "application/json"

    passed = 0
    failed = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if _check(name, cond, detail):
            passed += 1
        else:
            failed += 1

    print(f"\nSmoke testing: {base_url}\n")

    # ── 1. Health check ──────────────────────────────────────────────────────
    print("Step 1: Health check")
    try:
        health = _get(session, f"{base_url}/health")
        ok("GET /health returns 200", True)
        ok("status is ok", health.get("status") in {"ok", "healthy"}, str(health))
    except Exception as exc:
        ok("GET /health", False, str(exc))
        print("\nServer not reachable — aborting smoke test.", file=sys.stderr)
        return False

    # ── 2. Templates ─────────────────────────────────────────────────────────
    print("\nStep 2: Templates")
    try:
        templates_resp = _get(session, f"{base_url}/templates")
        templates = templates_resp if isinstance(templates_resp, list) else templates_resp.get("templates", [])
        ok("GET /templates returns list", isinstance(templates, list))
        ok("built_environment_health template present",
           any("built_environment" in str(t) for t in templates),
           str(templates[:3]))
    except Exception as exc:
        ok("GET /templates", False, str(exc))

    # ── 3. Start a run ───────────────────────────────────────────────────────
    print("\nStep 3: Start candidate factory run")
    run_id: str | None = None
    try:
        run_resp = _post(session, f"{base_url}/runs", {
            "template_id": "built_environment_health",
            "domain_input": "Built environment and health outcomes",
            "max_candidates": 30,
            "enable_experimental": False,
        })
        run_id = run_resp.get("run_id")
        ok("POST /runs creates run", run_id is not None, str(run_resp))
    except Exception as exc:
        ok("POST /runs", False, str(exc))

    if not run_id:
        print("Cannot continue without run_id.", file=sys.stderr)
        return failed == 0

    # ── 4. Poll for awaiting_approval ────────────────────────────────────────
    print(f"\nStep 4: Polling run {run_id!r} (up to {POLL_TIMEOUT_S}s)")
    deadline = time.time() + POLL_TIMEOUT_S
    final_status = None
    while time.time() < deadline:
        try:
            state = _get(session, f"{base_url}/runs/{run_id}/state")
            final_status = state.get("status", "unknown")
            if final_status in {"awaiting_approval", "completed", "failed"}:
                break
            time.sleep(POLL_INTERVAL_S)
        except Exception as exc:
            print(f"  poll error: {exc}", file=sys.stderr)
            time.sleep(POLL_INTERVAL_S)

    ok("run reached awaiting_approval or completed",
       final_status in {"awaiting_approval", "completed"},
       f"status={final_status}")

    # ── 5. Fetch candidates ──────────────────────────────────────────────────
    print("\nStep 5: Fetch candidates")
    candidates: list[dict] = []
    try:
        cand_resp = _get(session, f"{base_url}/runs/{run_id}/candidates")
        candidates = cand_resp if isinstance(cand_resp, list) else cand_resp.get("candidates", [])
        ok("GET /candidates returns list", isinstance(candidates, list))
        ok("candidate_count >= 20", len(candidates) >= 20, f"got {len(candidates)}")
    except Exception as exc:
        ok("GET /candidates", False, str(exc))

    # ── 6. Find top ready candidate ──────────────────────────────────────────
    print("\nStep 6: Find top ready candidate")
    top_candidate: dict | None = None
    ready = [c for c in candidates if c.get("shortlist_status") == "ready"]
    ok("at least one ready candidate", len(ready) >= 1, f"ready={len(ready)}")
    if ready:
        top_candidate = sorted(ready, key=lambda c: -c.get("scores", {}).get("overall", 0))[0]
        print(f"  top candidate: {top_candidate.get('candidate_id')}")

    # ── 7. Development pack ──────────────────────────────────────────────────
    print("\nStep 7: Generate development pack")
    if top_candidate:
        cand_id = top_candidate["candidate_id"]
        try:
            pack_resp = _post(
                session,
                f"{base_url}/runs/{run_id}/candidates/{cand_id}/development-pack",
                {},
            )
            pack_path = pack_resp.get("pack_dir") or pack_resp.get("path")
            ok("POST development-pack returns path", pack_path is not None, str(pack_resp))
        except Exception as exc:
            ok("POST development-pack", False, str(exc))
            pack_path = None

        # ── 8. Download claude_task_prompt.md ────────────────────────────────
        print("\nStep 8: Download claude_task_prompt.md")
        try:
            prompt_url = f"{base_url}/runs/{run_id}/outputs/{cand_id}/claude_task_prompt.md"
            resp = session.get(prompt_url, timeout=30)
            ok("GET claude_task_prompt.md returns 200", resp.status_code == 200,
               f"status={resp.status_code}")
            ok("prompt is non-empty", len(resp.content) > 50, f"bytes={len(resp.content)}")
        except Exception as exc:
            ok("GET claude_task_prompt.md", False, str(exc))
    else:
        print("  (skipping pack + prompt download — no ready candidate)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoPI cloud smoke test")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base API URL")
    args = parser.parse_args()

    ok = run_smoke_test(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
