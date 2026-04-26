import asyncio
import json

from agents import settings
from api import server


class _Run:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "awaiting_approval"
        self.thread_id = f"thread-{run_id}"


def test_candidate_card_contract_has_no_raw_blob(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-card-contract"
    run_root = settings.run_root(run_id, create=True)
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_001",
                "title": "Title",
                "research_question": "RQ",
                "exposure_family": "x",
                "outcome_family": "y",
                "exposure_source": "src_x",
                "outcome_source": "src_y",
                "method_template": "cross_sectional_spatial_association",
                "unit_of_analysis": "tract",
            }
        ]
    }
    out = run_root / "output" / "candidate_cards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))
    response = asyncio.run(server.list_candidates(run_id))
    card = response["candidates"][0]
    assert "_raw" not in card
    assert {"candidate_id", "title", "scores", "gate_summary"}.issubset(card.keys())
