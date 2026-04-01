import os

import pytest
from fastapi import HTTPException

from agents import settings
from api import server


@pytest.mark.asyncio
async def test_list_outputs_prefers_run_scoped_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-outputs-1"
    run_root = settings.run_root(run_id, create=True)
    scoped_output = run_root / "output" / "topic_screening.json"
    scoped_output.parent.mkdir(parents=True, exist_ok=True)
    scoped_output.write_text("{}", encoding="utf-8")

    class _Run:
        pass

    fake_run = _Run()
    fake_run.run_id = run_id
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: fake_run)

    resp = await server.list_outputs(run_id)
    names = [f.filename for f in resp.files]
    assert "output/topic_screening.json" in names


@pytest.mark.asyncio
async def test_download_output_uses_run_root_and_blocks_traversal(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-outputs-2"
    run_root = settings.run_root(run_id, create=True)
    p = run_root / "output" / "research_context.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")

    class _Run:
        pass

    fake_run = _Run()
    fake_run.run_id = run_id
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: fake_run)

    ok = await server.download_output(run_id, "output/research_context.json")
    assert os.path.basename(ok.path) == "research_context.json"

    with pytest.raises(HTTPException) as exc:
        await server.download_output(run_id, "../secret.txt")
    assert exc.value.status_code == 403
