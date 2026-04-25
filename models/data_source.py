"""DataSourceEntry — canonical representation of a local data source catalog entry."""

from typing import Optional

from pydantic import BaseModel


class DataSourceEntry(BaseModel):
    name: str
    alias: list[str] = []
    description: str = ""
    spatial_units: list[str] = []
    coverage_years: Optional[list[int]] = None
    coverage_year_min: Optional[int] = None
    coverage_year_max: Optional[int] = None
    skills_required: list[str] = []
    auth_required: bool = False
    url: str = ""
    format: str = ""
    notes: str = ""
