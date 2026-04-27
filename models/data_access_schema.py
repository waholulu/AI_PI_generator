from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DataAccessCheck(BaseModel):
    source_name: str
    access_url: str = ""
    documentation_url: str = ""
    reachable: bool = False
    machine_readable: bool = False
    expected_format: str = ""
    license_found: bool = False
    covers_exposure: bool = False
    covers_outcome: bool = False
    geography_compatible: bool = False
    time_compatible: bool = False
    verdict: Literal["pass", "warning", "fail"]
    reasons: list[str] = Field(default_factory=list)
