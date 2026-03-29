import streamlit as st
import json
import os
import sys

# Add parent dir to path so we can import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import build_orchestrator, ResearchState

st.set_page_config(page_title="Auto-PI Monitoring UI", layout="wide")

st.title("Auto-PI: Multi-Agent Research Scaffold UI")
st.markdown("Automating ideation, literature gathering, drafting, and data collection.")

# Initialize session state for the graph thread
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo_thread_1"
if "app_graph" not in st.session_state:
    st.session_state.app_graph = build_orchestrator()

graph = st.session_state.app_graph
config = {"configurable": {"thread_id": st.session_state.thread_id}}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Control Panel")
    domain = st.text_input("Research Domain:", "GeoAI and Urban Planning")
    
    if st.button("Start / Resume Pipeline"):
        with st.spinner("Running Agent Workflow..."):
            initial_state = ResearchState(domain_input=domain, execution_status="starting")
            
            # Stream the events
            for event in graph.stream(initial_state, config, stream_mode="values"):
                st.session_state.current_event = event
            
            st.success("Pipeline Step Complete or Needs Intervention!")

with col2:
    st.header("Pipeline State Monitor")
    
    # Retrieve current state from checkpointer
    try:
        current_state = graph.get_state(config)
        state_values = current_state.values
        next_nodes = current_state.next
        
        st.subheader("Current Execution Status")
        st.info(state_values.get("execution_status", "Not Started"))
        
        if next_nodes:
            st.warning(f"Workflow paused. Next executing node: **{next_nodes[0]}**")
            # If paused before literature, show the topic screening results
            if "literature" in next_nodes:
                plan_path = state_values.get("current_plan_path", "")
                if os.path.exists(plan_path):
                    st.subheader("Action Required: Review Research Plan")
                    with open(plan_path, "r", encoding="utf-8") as f:
                        plan_data = json.load(f)
                    st.json(plan_data)
                    st.markdown("**Approve to continue harvesting literature and data.**")
                    if st.button("Approve & Continue"):
                        st.spinner("Resuming...")
                        for event in graph.stream(None, config, stream_mode="values"):
                            pass
                        st.rerun()

        st.subheader("Files Generated (Pointers)")
        st.json(state_values)
        
    except Exception as e:
        st.write("No active pipeline data available or checkpointer error.")
        st.write(str(e))
