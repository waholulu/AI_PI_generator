from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root))
    from autopi_engine.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
