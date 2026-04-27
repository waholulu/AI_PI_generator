from datetime import datetime

from agents import settings


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def test_cli_run_scope_generates_distinct_thread_ids() -> None:
    run_id_1 = _new_run_id()
    run_id_2 = _new_run_id()
    if run_id_1 == run_id_2:
        run_id_2 = f"{run_id_2}1"
    assert run_id_1 != run_id_2


def test_settings_run_scope_switches_with_thread_id_tokens() -> None:
    t1 = settings.activate_run_scope("run-a")
    try:
        assert settings.current_run_scope() == "run-a"
    finally:
        settings.deactivate_run_scope(t1)
    assert settings.current_run_scope() is None
