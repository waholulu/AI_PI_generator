import streamlit as st
import json
import os
import sys

# Add parent dir to path so we can import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import build_orchestrator, ResearchState
from agents.hitl_helpers import (
    apply_idea_selection,
    load_validated_topics,
    load_tentative_topics,
    promote_tentative,
    kill_tentative,
    rerun_tentative_reflection,
    record_rejected_topics,
    regenerate_topics,
    MAX_REGENERATION_ROUNDS,
)

st.set_page_config(page_title="Auto-PI Monitoring UI", layout="wide")

st.title("Auto-PI: Multi-Agent Research Scaffold UI  v0.3.0")
st.markdown("Automating ideation, literature gathering, drafting, and data collection.")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo_thread_1"
if "app_graph" not in st.session_state:
    st.session_state.app_graph = build_orchestrator()
if "regeneration_round" not in st.session_state:
    st.session_state.regeneration_round = 0

graph = st.session_state.app_graph
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Three-tab layout ──────────────────────────────────────────────────────────
tab_control, tab_monitor, tab_tentative = st.tabs([
    "Control Panel",
    "Pipeline State Monitor",
    "TENTATIVE Review",
])

# ── Tab 1: Control Panel ──────────────────────────────────────────────────────
with tab_control:
    st.header("Control Panel")
    domain = st.text_input("Research Domain:", "GeoAI and Urban Planning")

    if st.button("Start / Resume Pipeline"):
        with st.spinner("Running Agent Workflow..."):
            initial_state = ResearchState(domain_input=domain, execution_status="starting")

            for event in graph.stream(initial_state, config, stream_mode="values"):
                st.session_state.current_event = event

            st.success("Pipeline Step Complete or Needs Intervention!")

# ── Tab 2: Pipeline State Monitor ────────────────────────────────────────────
with tab_monitor:
    st.header("Pipeline State Monitor")

    try:
        current_state = graph.get_state(config)
        state_values = current_state.values
        next_nodes = current_state.next

        st.subheader("Current Execution Status")
        st.info(state_values.get("execution_status", "Not Started"))

        if next_nodes:
            st.warning(f"Workflow paused. Next executing node: **{next_nodes[0]}**")
            if "literature" in next_nodes:
                ideas = load_validated_topics()
                regen_round = st.session_state.regeneration_round

                if regen_round > 0:
                    st.info(f"已重新构思 {regen_round} 次")

                st.subheader("候选选题（请选择一个推进）")
                for i, idea in enumerate(ideas):
                    verdict = idea.get("overall_verdict", "?")
                    icon = {"passed": "✅", "warning": "⚠️", "failed": "❌"}.get(verdict, "❓")
                    title = idea.get("title", "?")
                    rationale = idea.get("brief_rationale", "")
                    novelty = idea.get("novelty", {}).get("verdict", "?")
                    data_checks = idea.get("data_availability", [])
                    verified = sum(1 for d in data_checks if d.get("status") == "verified")
                    total = len(data_checks)
                    with st.expander(f"{icon} 选题 {i + 1}: {title}", expanded=True):
                        if rationale:
                            st.markdown(f"**推荐理由**: {rationale}")
                        st.write(f"**原创性**: {novelty}")
                        if total > 0:
                            st.write(f"**数据源**: {verified}/{total} 已验证")
                        for sp in idea.get("novelty", {}).get("similar_papers", [])[:3]:
                            st.write(f"- [{sp.get('similarity_verdict', '?')}] {sp.get('title', '?')}")
                        if idea.get("failure_reasons"):
                            for reason in idea["failure_reasons"]:
                                st.error(reason)

                if ideas:
                    allow_regen = regen_round < MAX_REGENERATION_ROUNDS
                    option_labels = [
                        f"选题 {i + 1}: {idea.get('title', '?')[:60]}"
                        for i, idea in enumerate(ideas)
                    ]
                    if allow_regen:
                        option_labels.append("都不满意，重新构思主题筛选")
                    else:
                        st.warning(
                            f"已达到最大重新构思次数（{MAX_REGENERATION_ROUNDS}），请从当前选题中选择。"
                        )

                    choice = st.radio(
                        "请选择：", option_labels, key=f"hitl_choice_{regen_round}",
                    )

                    if st.button("确认 / Confirm"):
                        selected_index = option_labels.index(choice)
                        is_regenerate = allow_regen and selected_index == len(ideas)

                        if is_regenerate:
                            with st.spinner("正在记录并重新构思选题..."):
                                from agents import settings as _settings
                                screening_path = _settings.topic_screening_path()
                                screening_topics = []
                                if os.path.exists(screening_path):
                                    try:
                                        with open(screening_path, "r", encoding="utf-8") as f:
                                            screening_topics = json.load(f).get("candidates", [])
                                    except (OSError, json.JSONDecodeError):
                                        screening_topics = []

                                domain_val = state_values.get("domain_input", "")
                                new_round = regen_round + 1
                                record_rejected_topics(
                                    screening_topics or ideas, domain_val, new_round,
                                )
                                regenerate_topics(dict(state_values))
                                st.session_state.regeneration_round = new_round
                            st.success("选题已重新生成，请再次审阅。")
                            st.rerun()
                        else:
                            with st.spinner("应用选题并恢复工作流..."):
                                selected_title = apply_idea_selection(selected_index)
                                if not selected_title:
                                    st.error("应用选题失败")
                                    st.stop()

                                for event in graph.stream(None, config, stream_mode="values"):
                                    pass
                                st.session_state.regeneration_round = 0
                            st.success("工作流已恢复，后续节点执行完毕。")
                            st.rerun()

        st.subheader("Files Generated (Pointers)")
        st.json(state_values)

    except Exception as e:
        st.write("No active pipeline data available or checkpointer error.")
        st.write(str(e))

# ── Tab 3: TENTATIVE Review ───────────────────────────────────────────────────
with tab_tentative:
    st.header("TENTATIVE Review")
    st.markdown(
        "Topics that failed ≥ 1 refinable gate but did not hit a hard-blocker are "
        "held here for human review. You can **Promote** (push to rank-1 candidate), "
        "**Kill** (send to graveyard), or **Re-run** (trigger one more reflection round)."
    )

    pool = load_tentative_topics()

    if st.button("Refresh", key="refresh_tentative"):
        st.rerun()

    if not pool:
        st.info("No TENTATIVE topics pending review.")
    else:
        st.write(f"**{len(pool)} topic(s) pending review:**")

        tentative_domain = st.text_input(
            "Domain (required for Kill → graveyard):", "Urban Planning", key="tent_domain"
        )

        for i, entry in enumerate(pool):
            title = entry.get("title") or entry.get("topic_id", f"Topic {i+1}")
            failed = entry.get("failed_gates", [])
            score = entry.get("legacy_six_gates", {})
            rerun_status = entry.get("last_rerun_status", "")

            with st.expander(f"[{i+1}] {title}", expanded=True):
                col_info, col_actions = st.columns([2, 1])

                with col_info:
                    st.markdown(f"**Topic ID:** `{entry.get('topic_id', '?')}`")
                    if failed:
                        st.markdown(f"**Failed gates:** {', '.join(failed)}")
                    if rerun_status:
                        st.markdown(f"**Last rerun status:** `{rerun_status}`")

                    seven_gates = score.get("full_seven_gates", {})
                    if seven_gates:
                        gate_rows = []
                        for gid, gdata in seven_gates.items():
                            gate_name = gdata.get("gate", gid)
                            gate_pass = "✅" if gdata.get("passed") else "❌"
                            gate_score = gdata.get("score")
                            score_str = f" (score={gate_score}/5)" if gate_score is not None else ""
                            gate_rows.append(f"- {gate_pass} **{gate_name}**{score_str}")
                        st.markdown("\n".join(gate_rows))

                with col_actions:
                    if st.button("Promote", key=f"promote_{i}"):
                        ok = promote_tentative(i)
                        if ok:
                            st.success(f"'{title}' promoted to rank-1 candidate.")
                        else:
                            st.error("Promote failed — index may have changed.")
                        st.rerun()

                    if st.button("Kill", key=f"kill_{i}"):
                        ok = kill_tentative(i, domain=tentative_domain or "unknown")
                        if ok:
                            st.success(f"'{title}' sent to graveyard.")
                        else:
                            st.error("Kill failed — index may have changed.")
                        st.rerun()

                    if st.button("Re-run Reflection", key=f"rerun_{i}"):
                        with st.spinner("Running one reflection round..."):
                            updated = rerun_tentative_reflection(i)
                        if updated:
                            new_status = updated.get("last_rerun_status", "unknown")
                            st.success(f"Rerun complete. New status: **{new_status}**")
                        else:
                            st.error("Rerun failed.")
                        st.rerun()
