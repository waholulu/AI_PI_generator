import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so that `agents` and other
# top-level modules can be imported in tests regardless of how pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


def pytest_runtest_setup(item):
    # Ensure each test starts with global (non run-scoped) settings context.
    from agents import settings

    token = settings.activate_run_scope("__pytest_reset__")
    settings.deactivate_run_scope(token)
    # Disable caches by default in tests to avoid cross-test leakage.
    os.environ["OPENALEX_CACHE_HOURS"] = "0"
    os.environ["KEYWORD_PLAN_CACHE_HOURS"] = "0"

