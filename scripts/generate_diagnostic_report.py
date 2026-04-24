#!/usr/bin/env python3
"""
Generate a Markdown diagnostic report from Module 1 reflection traces.

Reads all {topic_id}_trace.json files in output/ideation_traces/ and
produces output/diagnostic_report.md with:
  - Gate pass rates across all topics
  - Refine operation frequency
  - Oscillation / early-stop statistics
  - Free-form field completeness
  - Budget distribution

Usage:
    python scripts/generate_diagnostic_report.py [--output PATH]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.settings import ideation_traces_dir, global_output_dir


def load_traces(traces_dir: Path) -> list[dict]:
    traces = []
    for f in sorted(traces_dir.glob("*_trace.json")):
        try:
            with open(f) as fh:
                traces.append(json.load(fh))
        except Exception as e:
            print(f"  [WARN] Could not read {f.name}: {e}", file=sys.stderr)
    return traces


def compute_gate_pass_rates(traces: list[dict]) -> dict[str, dict]:
    gate_counts: dict[str, Counter] = defaultdict(Counter)
    for trace in traces:
        for rnd in trace.get("rounds", []):
            for gr in rnd.get("gate_results", []):
                gid = gr.get("gate_id", "?")
                gate_counts[gid]["total"] += 1
                if gr.get("passed"):
                    gate_counts[gid]["passed"] += 1
    result = {}
    for gid, counts in sorted(gate_counts.items()):
        total = counts["total"]
        passed = counts["passed"]
        result[gid] = {
            "pass_rate": passed / total if total else 0.0,
            "total": total,
            "passed": passed,
        }
    return result


def compute_refine_op_frequency(traces: list[dict]) -> Counter:
    ops_counter: Counter = Counter()
    for trace in traces:
        for rnd in trace.get("rounds", []):
            for op in rnd.get("applied_operations", []):
                ops_counter[op.get("op", "unknown")] += 1
    return ops_counter


def compute_status_breakdown(traces: list[dict]) -> Counter:
    statuses: Counter = Counter()
    for trace in traces:
        statuses[trace.get("final_status", "UNKNOWN")] += 1
    return statuses


def compute_budget_stats(traces: list[dict]) -> dict:
    costs = [trace.get("total_cost_usd", 0.0) for trace in traces]
    if not costs:
        return {}
    total = sum(costs)
    avg = total / len(costs)
    max_cost = max(costs)
    return {"total_usd": total, "avg_usd": avg, "max_usd": max_cost, "n": len(costs)}


def compute_cost_histogram(traces: list[dict]) -> list[tuple[str, int]]:
    buckets = [
        (0.0, 0.05, "[0,0.05)"),
        (0.05, 0.10, "[0.05,0.1)"),
        (0.10, 0.20, "[0.1,0.2)"),
        (0.20, 0.50, "[0.2,0.5)"),
        (0.50, 1.00, "[0.5,1.0)"),
        (1.00, float("inf"), "[1.0,+inf)"),
    ]
    counts = {label: 0 for _, _, label in buckets}
    for trace in traces:
        c = float(trace.get("total_cost_usd", 0.0) or 0.0)
        for low, high, label in buckets:
            if low <= c < high:
                counts[label] += 1
                break
    return [(label, counts[label]) for _, _, label in buckets]


def compute_free_form_rationale_top(traces: list[dict], limit: int = 30) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for trace in traces:
        for rnd in trace.get("rounds", []):
            for op in rnd.get("applied_operations", []):
                if op.get("op") != "free_form":
                    continue
                rationale = (
                    op.get("params", {}).get("rationale")
                    or op.get("rationale")
                    or ""
                ).strip()
                if rationale:
                    counter[rationale] += 1
    return counter.most_common(limit)


def compute_op_to_accept_rate(traces: list[dict]) -> list[tuple[str, int, int, float]]:
    op_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # total, accepted
    for trace in traces:
        final = trace.get("final_status")
        ops = set()
        for rnd in trace.get("rounds", []):
            for op in rnd.get("applied_operations", []):
                op_name = op.get("op")
                if op_name:
                    ops.add(op_name)
        for op_name in ops:
            op_stats[op_name][0] += 1
            if final == "ACCEPTED":
                op_stats[op_name][1] += 1
    rows = []
    for op, (total, accepted) in op_stats.items():
        rate = accepted / total if total else 0.0
        rows.append((op, total, accepted, rate))
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


def compute_free_form_completeness(traces: list[dict]) -> dict:
    has_title = 0
    has_abstract = 0
    n = len(traces)
    for trace in traces:
        ft = trace.get("final_topic") or {}
        if ft.get("free_form_title"):
            has_title += 1
        if ft.get("free_form_abstract"):
            has_abstract += 1
    return {
        "pct_has_title": has_title / n if n else 0.0,
        "pct_has_abstract": has_abstract / n if n else 0.0,
    }


def compute_oscillation_stats(traces: list[dict]) -> dict:
    tentative_from_oscillation = 0
    early_stop = 0
    for trace in traces:
        rounds = trace.get("rounds", [])
        if len(rounds) < 2:
            continue
        sigs = [r.get("four_tuple_sig") for r in rounds]
        # Oscillation: last N signatures identical
        if len(sigs) >= 3 and len(set(sigs[-3:])) == 1:
            tentative_from_oscillation += 1
        # Early stop heuristic: TENTATIVE with < max_rounds (3)
        if trace.get("final_status") == "TENTATIVE" and len(rounds) < 3:
            early_stop += 1
    return {
        "oscillation_count": tentative_from_oscillation,
        "early_stop_count": early_stop,
    }


def render_report(
    traces: list[dict],
    gate_rates: dict,
    refine_ops: Counter,
    status_breakdown: Counter,
    budget: dict,
    free_form: dict,
    oscillation: dict,
    cost_hist: list[tuple[str, int]],
    free_form_top: list[tuple[str, int]],
    op_accept_rates: list[tuple[str, int, int, float]],
) -> str:
    n = len(traces)
    lines = [
        "# Module 1 Reflection Loop — Diagnostic Report",
        "",
        f"**Traces analysed:** {n}",
        "",
    ]

    if n < 10:
        lines += [
            "> **Note:** Fewer than 10 traces available. Statistics may not be representative.",
            "",
        ]

    # Status breakdown
    lines += ["## Status Breakdown", ""]
    for status, count in sorted(status_breakdown.items()):
        pct = count / n * 100 if n else 0
        lines.append(f"- **{status}**: {count} ({pct:.1f}%)")
    lines.append("")

    # Gate pass rates
    lines += ["## Gate Pass Rates", ""]
    lines.append("| Gate | Pass Rate | Passed | Total |")
    lines.append("|------|-----------|--------|-------|")
    for gid, info in gate_rates.items():
        lines.append(
            f"| {gid} | {info['pass_rate']:.0%} | {info['passed']} | {info['total']} |"
        )
    lines.append("")

    # Refine operation frequency
    lines += ["## Refine Operation Frequency", ""]
    if refine_ops:
        lines.append("| Operation | Count |")
        lines.append("|-----------|-------|")
        for op, count in refine_ops.most_common(15):
            lines.append(f"| `{op}` | {count} |")
    else:
        lines.append("_No refine operations recorded._")
    lines.append("")

    # Oscillation / early-stop
    lines += ["## Oscillation & Early Stop", ""]
    lines.append(f"- Topics forced TENTATIVE by oscillation: **{oscillation['oscillation_count']}**")
    lines.append(f"- Topics early-stopped: **{oscillation['early_stop_count']}**")
    lines.append("")

    # Free-form field completeness
    lines += ["## Free-form Field Completeness", ""]
    lines.append(f"- Topics with `free_form_title`: **{free_form['pct_has_title']:.0%}**")
    lines.append(f"- Topics with `free_form_abstract`: **{free_form['pct_has_abstract']:.0%}**")
    lines.append("")

    # Budget
    if budget:
        lines += ["## Budget Distribution", ""]
        lines.append(f"- **Total spent:** ${budget['total_usd']:.4f}")
        lines.append(f"- **Average per topic:** ${budget['avg_usd']:.4f}")
        lines.append(f"- **Max single topic:** ${budget['max_usd']:.4f}")
        lines.append(f"- **Topics tracked:** {budget['n']}")
        lines.append("")
        lines.append("### Per-topic cost histogram")
        max_count = max([c for _, c in cost_hist], default=1)
        for label, count in cost_hist:
            bar = "#" * (0 if max_count == 0 else int((count / max_count) * 30))
            lines.append(f"- {label:>10} | {bar} ({count})")
        lines.append("")

    lines += ["## Free-form rationale top list", ""]
    if free_form_top:
        for rationale, count in free_form_top:
            lines.append(f"- ({count}) {rationale}")
    else:
        lines.append("_No free_form rationale found._")
    lines.append("")

    lines += ["## Refine op -> final ACCEPT rate", ""]
    if op_accept_rates:
        lines.append("| Operation | Traces | ACCEPTED | Accept Rate |")
        lines.append("|-----------|--------|----------|-------------|")
        for op, total, accepted, rate in op_accept_rates:
            lines.append(f"| `{op}` | {total} | {accepted} | {rate:.1%} |")
    else:
        lines.append("_No refine operations recorded._")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Module 1 diagnostic report")
    parser.add_argument("--output", default=None, help="Output Markdown file path")
    args = parser.parse_args()

    traces_dir = ideation_traces_dir()
    traces = load_traces(traces_dir)

    if len(traces) < 10:
        print(f"Only {len(traces)} traces found (< 10). Skip diagnostic report generation.", file=sys.stderr)
        sys.exit(0)

    gate_rates = compute_gate_pass_rates(traces)
    refine_ops = compute_refine_op_frequency(traces)
    status_breakdown = compute_status_breakdown(traces)
    budget = compute_budget_stats(traces)
    free_form = compute_free_form_completeness(traces)
    oscillation = compute_oscillation_stats(traces)
    cost_hist = compute_cost_histogram(traces)
    free_form_top = compute_free_form_rationale_top(traces)
    op_accept_rates = compute_op_to_accept_rate(traces)

    report = render_report(
        traces, gate_rates, refine_ops, status_breakdown,
        budget, free_form, oscillation, cost_hist, free_form_top, op_accept_rates,
    )

    out_path = Path(args.output) if args.output else (global_output_dir() / "diagnostic_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Diagnostic report written to {out_path}")


if __name__ == "__main__":
    main()
