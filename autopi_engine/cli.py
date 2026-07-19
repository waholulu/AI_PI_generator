from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import workflow


def _print(payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_payload(path: str, kind: str) -> dict[str, Any] | str:
    text = Path(path).read_text(encoding="utf-8")
    if kind == "research_memo":
        return text
    return json.loads(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autopi", description="Manifest-backed Auto-PI Skill runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--domain", required=True)
    p_init.add_argument("--run-id")

    sub.add_parser("list")

    p_status = sub.add_parser("status")
    p_status.add_argument("--run-id", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--run-id", required=True)
    p_submit.add_argument("--kind", required=True, choices=[
        "search_plan",
        "candidate_selection",
        "novelty_search_plan",
        "novelty_assessment",
        "novelty_decision",
        "literature_search_plan",
        "research_memo",
    ])
    p_submit.add_argument("--file", required=True)

    p_advance = sub.add_parser("advance")
    p_advance.add_argument("--run-id", required=True)

    p_retry = sub.add_parser("retry")
    p_retry.add_argument("--run-id", required=True)

    p_abort = sub.add_parser("abort")
    p_abort.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            _print(workflow.init_run(args.domain, args.run_id))
        elif args.command == "list":
            _print(workflow.list_runs())
        elif args.command == "status":
            _print(workflow.status(args.run_id))
        elif args.command == "submit":
            _print(workflow.submit(args.run_id, args.kind, _load_payload(args.file, args.kind)))
        elif args.command == "advance":
            _print(workflow.advance(args.run_id))
        elif args.command == "retry":
            _print(workflow.retry(args.run_id))
        elif args.command == "abort":
            _print(workflow.abort(args.run_id))
        else:
            parser.error(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"autopi: error: {exc}", file=sys.stderr)
        return 2
    return 0
