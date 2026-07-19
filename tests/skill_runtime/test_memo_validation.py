from __future__ import annotations

from autopi_engine.storage import run_root
from autopi_engine.workflow import MEMO_HEADINGS, validate_memo


def test_memo_requires_exact_headings_and_known_citations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    root = run_root("memo_run")
    refs = root / "output" / "references.bib"
    refs.parent.mkdir(parents=True, exist_ok=True)
    refs.write_text("@article{paper_2024_test,\n  title={Test}\n}\n", encoding="utf-8")

    memo = "\n\n".join(f"## {heading}\nText." for heading in MEMO_HEADINGS)
    assert validate_memo("memo_run", memo + "\n\nCites @paper_2024_test.") == []
    assert validate_memo("memo_run", memo + "\n\nCites @missing_key.")


def test_memo_requires_all_eight_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_root("memo_run")

    errors = validate_memo("memo_run", "## Research Question\nText.")

    assert any("Contribution" in err for err in errors)
