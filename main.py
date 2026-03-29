import sys
from datetime import datetime

from dotenv import load_dotenv

from agents import orchestrator


NODE_DISPLAY_NAMES = {
    "field_scanner": "领域扫描与版图分析",
    "ideation": "研究方向与选题生成",
    "literature": "文献检索与知识盘点",
    "drafter": "研究计划与内容撰写",
    "data_fetcher": "数据资源与材料整理",
}

NODE_ORDER = [
    "field_scanner",
    "ideation",
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
            print("=" * 60)

            choice = input("\n是否继续执行后续节点？(y/n): ").strip().lower()
            if choice not in ("y", "yes", "是"):
                _log("用户选择中止，工作流停止。", level="ABORT")
                return

            _log("用户确认，恢复工作流...", level="HITL")
            _stream_phase(graph, None, config, seen_nodes, total_nodes)

        _log("工作流全部节点执行完毕。", level="SUCCESS")
    except KeyboardInterrupt:
        print()
        _log("用户中断（Ctrl+C），工作流停止。", level="ABORT")
    except Exception as e:
        _log(f"工作流执行发生异常：{e}", level="ERROR")


if __name__ == "__main__":
    main()
