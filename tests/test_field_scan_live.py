import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from agents.field_scanner_agent import FieldScannerAgent
from agents.orchestrator import ResearchState


@pytest.mark.live_openalex
def test_field_scan_live_uses_openalex(tmp_path) -> None:
    """
    端到端测试：使用真实 OpenAlex API 跑一遍领域牵引扫描。

    - 需要 .env 中配置 OPENALEX_API_KEY / OPENALEX_EMAIL
    - 需要安装 pyalex 依赖
    """
    # 从项目根目录加载 .env（与 test_api_keys 的行为一致）
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("OPENALEX_API_KEY")
    email = os.getenv("OPENALEX_EMAIL")
    if not api_key or not email:
        raise AssertionError("OPENALEX_API_KEY / OPENALEX_EMAIL must be configured for live OpenAlex tests.")

    try:
        import pyalex  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing
        raise AssertionError("pyalex must be installed for live OpenAlex tests.") from exc

    # 构造最小可行的 ResearchState；只关心 domain_input 与 execution_status
    state: ResearchState = {
        "domain_input": "GeoAI and Urban Planning",
        "field_scan_path": "",
        "candidate_topics_path": "",
        "current_plan_path": "",
        "literature_inventory_path": "",
        "draft_content_path": "",
        "raw_data_manifest_path": "",
        "execution_status": "starting",
    }

    agent = FieldScannerAgent()
    new_state = agent.run(state)

    # 基本断言：调用成功且产生了 field_scan.json，且包含非空 OpenAlex 结果
    assert new_state["execution_status"] == "ideation"
    assert "field_scan_path" in new_state

    field_scan_path = new_state["field_scan_path"]
    assert os.path.exists(field_scan_path)

    with open(field_scan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("openalex_traction", {}).get("top_results") or []
    assert isinstance(results, list) and results, "Expected non-empty OpenAlex results in live scan"
