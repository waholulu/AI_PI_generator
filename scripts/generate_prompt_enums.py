#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.topic_schema import (
    ContributionPrimary,
    ExposureFamily,
    Frequency,
    IdentificationPrimary,
    OutcomeFamily,
    SamplingMode,
)


def _enum_section(enum_cls) -> list[str]:
    lines = [f"## {enum_cls.__name__}"]
    lines.extend(f"- {member.value}" for member in enum_cls)
    lines.append("")
    return lines


def main() -> None:
    lines: list[str] = []
    for enum_cls in (
        ExposureFamily,
        OutcomeFamily,
        SamplingMode,
        Frequency,
        IdentificationPrimary,
        ContributionPrimary,
    ):
        lines.extend(_enum_section(enum_cls))

    output_path = Path("prompts") / "_enums.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
