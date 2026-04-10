import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from agents import orchestrator, settings


NODE_DISPLAY_NAMES = {
    "field_scanner": "领域扫描与版图分析",
    "ideation": "研究方向与选题生成",
    "idea_validator": "选题验证（原创性 + 数据可用性）",
    "literature": "文献检索与知识盘点",
    "drafter": "研究计划与内容撰写",
    "data_fetcher": "数据资源与材料整理",
}

NODE_ORDER = [
    "field_scanner",
    "ideation",
    "idea_validator",
    "literature",
    "drafter",
    "data_fetcher",
]

HITL_INTERRUPT_NODES = {"literature"}


def _log(message: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {message}")


def _print_banner() -> None:
    print("=" * 60)
    print("   Auto-PI: Automated Research Engine v2.0")
    print("=" * 60)


def _print_progress(node_key: str, state: dict, index: int, total: int) -> None:
    name = NODE_DISPLAY_NAMES.get(node_key, node_key)
    ratio = min(max(index / total, 0.0), 1.0) if total else 0.0
    bar_width = 24
    filled = int(bar_width * ratio)
    bar = "█" * filled + "░" * (bar_width - filled)
    percent = int(ratio * 100)

    print()
    _log(f"流程节点 {index}/{total}：{name} [{node_key}]")
    print(f"[{bar}] {percent:3d}%")

    interesting_keys = [
        k for k in state.keys() if k.endswith("_path") or k == "execution_status"
    ]
    for k in interesting_keys:
        v = state.get(k)
        if not v:
            continue
        if isinstance(v, str) and len(v) > 100:
            v = v[:97] + "..."
        print(f"  - {k}: {v}")
    print("-" * 60)


def _stream_phase(graph, input_state, config, seen_nodes: set, total_nodes: int) -> None:
    """Run one phase of graph.stream and display progress for each completed node."""
    for output in graph.stream(input_state, config):
        for key, value in output.items():
            if not isinstance(key, str):
                continue
            seen_nodes.add(key)
            idx = (NODE_ORDER.index(key) + 1) if key in NODE_ORDER else len(seen_nodes)
            state_dict = value if isinstance(value, dict) else {}
            _print_progress(key, state_dict, idx, total_nodes)


def _check_hitl_interrupt(graph, config) -> bool:
    """Return True if the graph is currently paused at an interrupt point."""
    snapshot = graph.get_state(config)
    return bool(snapshot.next)


def _display_validation_report() -> list:
    """Show a summary of idea validation results at the HITL checkpoint.

    Returns the list of validated ideas so the caller can prompt for selection.
    """
    validation_path = settings.idea_validation_path()
    if not os.path.exists(validation_path):
        return []
    try:
        with open(validation_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    print()
    _log("--- 选题验证报告 ---", level="HITL")
    subs = report.get("substitutions_made", 0)
    if subs > 0:
        _log(f"共进行 {subs} 次替补", level="HITL")

    ideas = report.get("validated_ideas", [])
    for idx, idea in enumerate(ideas):
        verdict = idea.get("overall_verdict", "?")
        novelty = idea.get("novelty", {}).get("verdict", "?")
        title = idea.get("title", "?")[:60]
        rank = idea.get("rank", "?")
        tag = {"passed": "✓", "warning": "⚠", "failed": "✗"}.get(verdict, "?")
        _log(f"  [{idx}] [{tag}] #{rank} {title}", level="HITL")
        _log(f"      原创性: {novelty}", level="HITL")

        # Show data source status
        data_checks = idea.get("data_availability", [])
        verified = sum(1 for d in data_checks if d.get("status") == "verified")
        total = len(data_checks)
        if total > 0:
            _log(f"      数据源: {verified}/{total} 已验证", level="HITL")

        # Show similar papers if any
        similar = idea.get("novelty", {}).get("similar_papers", [])
        for sp in similar[:2]:
            sp_title = sp.get("title", "?")[:50]
            sp_verdict = sp.get("similarity_verdict", "?")
            _log(f"      → [{sp_verdict}] {sp_title}", level="HITL")

        if idea.get("failure_reasons"):
            for reason in idea["failure_reasons"]:
                _log(f"      ⚠ {reason}", level="HITL")
    print()
    return ideas


def _prompt_idea_selection(ideas: list) -> int:
    """Ask the user to pick which idea to proceed with. Returns 0-based index."""
    if len(ideas) <= 1:
        return 0
    valid_indices = list(range(len(ideas)))
    while True:
        raw = input(
            f"\n请选择要深入研究的选题编号（0-{len(ideas)-1}，直接回车默认选 0 号）： "
        ).strip()
        if raw == "":
            _log("默认选择编号 0 的选题。", level="HITL")
            return 0
        try:
            idx = int(raw)
        except ValueError:
            _log(f"输入无效，请输入 0 到 {len(ideas)-1} 之间的数字。", level="WARN")
            continue
        if idx not in valid_indices:
            _log(f"编号超出范围，请输入 0 到 {len(ideas)-1} 之间的数字。", level="WARN")
            continue
        return idx


def _apply_selection_to_files(idea_idx: int, ideas: list) -> None:
    """Rotate the chosen idea to rank-1 in topic_screening.json and research_context.json."""
    if idea_idx == 0:
        return  # Top-1 already selected; nothing to change.

    screening_path = settings.topic_screening_path()
    context_path = settings.research_context_path()
    plan_path = settings.research_plan_path()

    if not os.path.exists(screening_path):
        _log("找不到 topic_screening.json，无法更改选题顺序。", level="WARN")
        return

    try:
        with open(screening_path, "r", encoding="utf-8") as f:
            screening = json.load(f)
        candidates: list = screening.get("candidates", [])
        if idea_idx >= len(candidates):
            _log(f"选题编号 {idea_idx} 超出候选列表范围，保持默认选题。", level="WARN")
            return
        selected = candidates.pop(idea_idx)
        candidates.insert(0, selected)
        for i, c in enumerate(candidates):
            c["rank"] = i + 1
        screening["candidates"] = candidates
        with open(screening_path, "w", encoding="utf-8") as f:
            json.dump(screening, f, indent=2, ensure_ascii=False)

        selected_title = selected.get("title", "")
        _log(f"已将 「{selected_title}」 设为第一候选选题。", level="HITL")

        # Update research_context.json
        if os.path.exists(context_path):
            with open(context_path, "r", encoding="utf-8") as f:
                ctx = json.load(f)
            if isinstance(ctx, dict):
                ctx["selected_topic"] = {
                    "title": selected_title,
                    "score": selected.get("final_score", selected.get("initial_score")),
                    "quantitative_specs": selected.get("quantitative_specs", {}),
                    "data_sources": selected.get("data_sources", []),
                    "publishability": selected.get("publishability", ""),
                    "selection_overridden": True,
                }
                with open(context_path, "w", encoding="utf-8") as f:
                    json.dump(ctx, f, indent=2, ensure_ascii=False)

        # Update research_plan.json title
        if os.path.exists(plan_path):
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            plan["project_title"] = selected_title
            plan["topic_screening"] = {"top_candidate_title": selected_title, "manually_selected": True}
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)

    except Exception as exc:
        _log(f"更新选题文件时出错：{exc}", level="WARN")


def _display_degraded_warnings(graph, config) -> None:
    """Print a warning block if any agents ran in degraded (LLM-fallback) mode."""
    try:
        snapshot = graph.get_state(config)
        degraded = snapshot.values.get("degraded_nodes") or []
    except Exception:
        return
    if not degraded:
        return
    print()
    print("=" * 60)
    _log("⚠  质量警告：以下节点在 LLM 不可用时启用了降级模式", level="WARN")
    for entry in degraded:
        _log(f"  • {entry}", level="WARN")
    _log("  输出内容可能不完整，建议检查并在 LLM 正常后重新运行。", level="WARN")
    print("=" * 60)


def main():
    _print_banner()

    load_dotenv()

    domain_input = input(
        "\n请输入您的宏观研究领域描述（例如：'GeoAI and Urban Planning'）： "
    ).strip()

    if not domain_input:
        _log("未输入有效领域，程序退出。", level="ERROR")
        sys.exit(1)

    _log(f"初始化工作流，目标领域：{domain_input!r}")

    graph = orchestrator.build_orchestrator()

    initial_state = {
        "domain_input": domain_input,
        "execution_status": "starting",
    }

    config = {"configurable": {"thread_id": "1"}}

    total_nodes = len(NODE_ORDER)
    seen_nodes: set[str] = set()

    try:
        # Phase 1: run until HITL interrupt (after ideation, before literature)
        _stream_phase(graph, initial_state, config, seen_nodes, total_nodes)

        # Handle HITL interrupt: prompt user to review ideation output and continue
        while _check_hitl_interrupt(graph, config):
            snapshot = graph.get_state(config)
            pending = list(snapshot.next)
            pending_names = ", ".join(
                NODE_DISPLAY_NAMES.get(n, n) for n in pending
            )

            print()
            print("=" * 60)
            _log(f"工作流在以下节点前暂停（HITL 检查点）：{pending_names}", level="HITL")
            _log("请检查 output/ 目录下的中间产物（选题、研究计划等）。", level="HITL")
            ideas = _display_validation_report()
            print("=" * 60)

            choice = input("\n是否继续执行后续节点？(y/n): ").strip().lower()
            if choice not in ("y", "yes", "是"):
                _log("用户选择中止，工作流停止。", level="ABORT")
                return

            # Let user pick which idea to proceed with
            selected_idx = _prompt_idea_selection(ideas)
            _apply_selection_to_files(selected_idx, ideas)

            _log("用户确认，恢复工作流...", level="HITL")
            _stream_phase(graph, None, config, seen_nodes, total_nodes)

        _log("工作流全部节点执行完毕。", level="SUCCESS")
        _display_degraded_warnings(graph, config)
    except KeyboardInterrupt:
        print()
        _log("用户中断（Ctrl+C），工作流停止。", level="ABORT")
    except Exception as e:
        _log(f"工作流执行发生异常：{e}", level="ERROR")


if __name__ == "__main__":
    main()
