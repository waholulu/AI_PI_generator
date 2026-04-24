import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from agents import orchestrator, settings
from agents.hitl_helpers import (
    load_validated_topics,
    record_rejected_topics,
    regenerate_topics,
    MAX_REGENERATION_ROUNDS,
)


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
    values = getattr(snapshot, "values", {}) or {}
    if values.get("hitl_interruption"):
        return True
    return bool(snapshot.next)


def _render_level1_hitl_card(current_state: dict, initial_state: dict) -> str:
    inter = current_state.get("hitl_interruption", {}) or {}
    topic_path = current_state.get("user_topic_path", initial_state.get("user_topic_path", "your_topic.yaml"))
    domain = current_state.get("domain_input", initial_state.get("domain_input", "your domain"))

    failed_gates = inter.get("failed_gates", []) or []
    failed_gates_section = (
        "Failed Gates:\n  - " + "\n  - ".join(str(g) for g in failed_gates)
        if failed_gates
        else "Failed Gates:\n  - (none provided)"
    )
    suggested_ops = inter.get("suggested_operations", []) or []
    if suggested_ops:
        lines = []
        for op in suggested_ops:
            if isinstance(op, dict):
                lines.append(f"  - {op.get('op', 'unknown')}: {op.get('description', '')}".rstrip(": "))
            else:
                lines.append(f"  - {op}")
        suggested_ops_section = "Suggested Operations:\n" + "\n".join(lines)
    else:
        suggested_ops_section = "Suggested Operations:\n  - (none)"

    diff_from_original = inter.get("diff_from_original", {}) or {}
    if diff_from_original:
        diff_section = "Diff from original:\n" + "\n".join(
            f"  - {k}: {v}" for k, v in diff_from_original.items()
        )
    else:
        diff_section = "Diff from original:\n  - (none)"

    template_path = settings.prompts_dir() / "level1_hitl_card.txt"
    try:
        template = template_path.read_text(encoding="utf-8")
    except Exception:
        template = (
            "Level1 HITL Required\n"
            "Topic: {topic_title}\nTopic ID: {topic_id}\nStatus: {status}\n\n"
            "{failed_gates_section}\n{suggested_ops_section}\n{diff_section}\n"
            "Re-run: python main.py --mode level_1 --user-topic {topic_path}\n"
            "Switch: python main.py --mode level_2 --domain \"{domain}\"\n"
        )

    return template.format(
        topic_title=inter.get("topic_title", "(unknown)"),
        topic_id=inter.get("topic_id", "(unknown)"),
        status=inter.get("kind", "hitl_required"),
        failed_gates_section=failed_gates_section,
        suggested_ops_section=suggested_ops_section,
        diff_section=diff_section,
        topic_path=topic_path,
        domain=domain,
    )


def _display_validation_report() -> list:
    """Show a summary of idea validation results at the HITL checkpoint.

    Returns the list of validated ideas so the caller can prompt for selection.
    """
    ideas = load_validated_topics()
    if not ideas:
        return []

    print()
    _log("--- 当前候选选题 ---", level="HITL")

    for idx, idea in enumerate(ideas):
        verdict = idea.get("overall_verdict", "?")
        novelty = idea.get("novelty", {}).get("verdict", "?")
        title = idea.get("title", "?")
        rationale = idea.get("brief_rationale", "")
        tag = {"passed": "\u2713", "warning": "\u26a0", "failed": "\u2717"}.get(verdict, "?")
        print()
        _log(f"  [{idx + 1}] [{tag}] {title}", level="HITL")
        if rationale:
            _log(f"      \u63a8\u8350\u7406\u7531: {rationale}", level="HITL")
        _log(f"      \u539f\u521b\u6027: {novelty}", level="HITL")

        # Show data source status
        data_checks = idea.get("data_availability", [])
        verified = sum(1 for d in data_checks if d.get("status") == "verified")
        total = len(data_checks)
        if total > 0:
            _log(f"      \u6570\u636e\u6e90: {verified}/{total} \u5df2\u9a8c\u8bc1", level="HITL")

        # Show similar papers if any
        similar = idea.get("novelty", {}).get("similar_papers", [])
        for sp in similar[:2]:
            sp_title = sp.get("title", "?")[:50]
            sp_verdict = sp.get("similarity_verdict", "?")
            _log(f"      \u2192 [{sp_verdict}] {sp_title}", level="HITL")

        if idea.get("failure_reasons"):
            for reason in idea["failure_reasons"]:
                _log(f"      \u26a0 {reason}", level="HITL")
    print()
    return ideas


def _render_level1_hitl_card(current_state: dict) -> None:
    hitl = current_state.get("hitl_interruption") or {}
    topic_path = current_state.get("user_topic_path", "<topic.yaml>")
    domain = current_state.get("domain_input", "")
    failed = hitl.get("failed_gates") or []
    suggested = hitl.get("suggested_operations") or []
    diff = hitl.get("diff_from_original") or {}

    failed_section = "Failed gates: " + (", ".join(failed) if failed else "(none)")
    suggested_section = "Suggested ops:\n" + (
        "\n".join(f"  - {op}" for op in suggested) if suggested else "  (none)"
    )
    diff_section = "Diff from original:\n" + (
        json.dumps(diff, ensure_ascii=False, indent=2) if diff else "  (none)"
    )

    try:
        tpl_path = settings.prompts_dir() / "level1_hitl_card.txt"
        tpl = tpl_path.read_text(encoding="utf-8")
        rendered = tpl.format(
            topic_title=hitl.get("topic_title", "N/A"),
            topic_id=hitl.get("topic_id", "N/A"),
            status=hitl.get("kind", "hitl_required"),
            failed_gates_section=failed_section,
            suggested_ops_section=suggested_section,
            diff_section=diff_section,
            topic_path=topic_path,
            domain=domain,
        )
        print(rendered)
    except Exception:
        _log("Level 1 HITL required.", level="HITL")
        _log(failed_section, level="HITL")
        _log(suggested_section, level="HITL")


def _prompt_topic_choice(ideas: list, allow_regenerate: bool = True) -> int:
    """Present numbered topic choices + optional regenerate option.

    Returns:
        0-based idea index (0, 1, 2) if a topic was selected, or
        -1 if the user chose to regenerate topics.
    """
    n = len(ideas)
    if n == 0:
        return -1

    regen_idx = n + 1  # e.g. 4 when there are 3 ideas
    options = ", ".join(str(i + 1) for i in range(n))
    if allow_regenerate:
        _log(f"  [{regen_idx}] 都不满意，重新构思主题筛选", level="HITL")
        options += f", {regen_idx}"

    while True:
        raw = input(f"\n请选择（{options}，直接回车默认选 1）： ").strip()
        if raw == "":
            _log("默认选择选题 1。", level="HITL")
            return 0
        try:
            choice = int(raw)
        except ValueError:
            _log(f"输入无效，请输入 {options} 中的数字。", level="WARN")
            continue
        if allow_regenerate and choice == regen_idx:
            return -1
        if 1 <= choice <= n:
            return choice - 1  # convert to 0-based
        _log(f"编号超出范围，请输入 {options} 中的数字。", level="WARN")


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


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-PI Research Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["level_1", "level_2"],
        default="level_2",
        help="Ideation mode: level_1 (user-provided topic YAML) or level_2 (auto-generate from domain)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Research domain description for Level 2 (e.g. 'GeoAI and Urban Planning')",
    )
    parser.add_argument(
        "--user-topic",
        dest="user_topic",
        default=None,
        metavar="PATH",
        help="Path to user-supplied structured topic YAML file (Level 1 mode)",
    )
    parser.add_argument(
        "--legacy-ideation",
        dest="legacy_ideation",
        action="store_true",
        default=False,
        help="Use legacy IdeationAgentV0 instead of V2 (backward compatibility)",
    )
    parser.add_argument(
        "--budget-override-usd",
        dest="budget_override_usd",
        type=float,
        default=None,
        metavar="USD",
        help="Override per-run LLM budget in USD (default: from reflection_config.yaml)",
    )
    parser.add_argument(
        "--skip-reflection",
        dest="skip_reflection",
        action="store_true",
        default=False,
        help="Skip reflection loop in Level 2 (one-shot topic generation, no iterative refinement)",
    )
    return parser.parse_args()


def main():
    _print_banner()

    load_dotenv()

    args = _parse_args()

    # Determine domain input
    if args.mode == "level_1" and args.user_topic:
        # Level 1: domain derived from topic YAML or defaults
        domain_input = args.domain or ""
        if not domain_input:
            _log("Level 1 模式：从用户提供的 topic YAML 推导领域。")
    elif args.domain:
        domain_input = args.domain.strip()
    else:
        domain_input = input(
            "\n请输入您的宏观研究领域描述（例如：'GeoAI and Urban Planning'）： "
        ).strip()

    if not domain_input and args.mode == "level_2":
        _log("未输入有效领域，程序退出。", level="ERROR")
        sys.exit(1)

    # Apply --legacy-ideation as env var so IdeationAgent router picks it up
    if args.legacy_ideation:
        import os as _os
        _os.environ["LEGACY_IDEATION"] = "1"

    _log(f"初始化工作流，目标领域：{domain_input!r}" if domain_input else "初始化工作流（Level 1 模式）")

    graph = orchestrator.build_orchestrator()

    initial_state = {
        "domain_input": domain_input,
        "execution_status": "starting",
        "legacy_ideation": args.legacy_ideation,
        "ideation_mode": args.mode,
        "budget_override_usd": args.budget_override_usd,
        "skip_reflection": args.skip_reflection,
    }
    if args.user_topic:
        import os as _os
        user_topic_path = _os.path.abspath(args.user_topic)
        if not _os.path.exists(user_topic_path):
            _log(f"--user-topic 文件不存在：{user_topic_path}", level="ERROR")
            sys.exit(1)
        initial_state["user_topic_path"] = user_topic_path
        _log(f"Level 1 模式：从 {user_topic_path} 加载用户主题。")

    config = {"configurable": {"thread_id": "1"}}

    total_nodes = len(NODE_ORDER)
    seen_nodes: set[str] = set()

    try:
        # Phase 1: run until HITL interrupt (after ideation, before literature)
        _stream_phase(graph, initial_state, config, seen_nodes, total_nodes)

        # Handle HITL interrupt: let user choose a topic or regenerate
        regeneration_round = 0
        while _check_hitl_interrupt(graph, config):
            snapshot = graph.get_state(config)
            current_state = dict(snapshot.values)
            pending = list(snapshot.next)
            pending_names = ", ".join(
                NODE_DISPLAY_NAMES.get(n, n) for n in pending
            )

            print()
            print("=" * 60)
            _log(f"工作流在以下节点前暂停（HITL 检查点）：{pending_names}", level="HITL")
            if regeneration_round > 0:
                _log(f"（第 {regeneration_round} 次重新构思）", level="HITL")
            ideas = _display_validation_report()
            print("=" * 60)

            if not ideas:
                _log("未找到候选选题，工作流中止。", level="ERROR")
                return

            allow_regen = regeneration_round < MAX_REGENERATION_ROUNDS
            if not allow_regen:
                _log(
                    f"已达到最大重新构思次数（{MAX_REGENERATION_ROUNDS}），请从当前选题中选择。",
                    level="WARN",
                )

            selected_idx = _prompt_topic_choice(ideas, allow_regenerate=allow_regen)

            if selected_idx == -1:
                # User chose to regenerate
                regeneration_round += 1
                domain = current_state.get("domain_input", "")
                _log("正在记录当前选题并重新构思...", level="HITL")

                # Record current topics as rejected
                screening_path = settings.topic_screening_path()
                screening_topics = []
                if os.path.exists(screening_path):
                    try:
                        with open(screening_path, "r", encoding="utf-8") as f:
                            screening_topics = json.load(f).get("candidates", [])
                    except (json.JSONDecodeError, OSError):
                        screening_topics = []
                record_rejected_topics(
                    screening_topics or ideas, domain, regeneration_round,
                )

                # Re-run ideation + validator outside the graph
                _log("重新生成候选选题中，请稍候...", level="HITL")
                regenerate_topics(current_state)
                _log("新一轮选题已生成。", level="HITL")
                continue  # loop back to display new topics

            # User selected a topic — apply and resume
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
