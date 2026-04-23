"""
Rule engine for deterministic (zero-LLM-cost) gate checks in Module 1.

Implements G2 (scale_alignment), G3 (data_availability), G6 (automation_feasibility),
and the G4 threat-coverage helper.  Hard-blocker gates return refinable=False; a
single hard-blocker failure triggers immediate REJECTED in the reflection loop.

YAML configs are loaded lazily and cached; if any config file is missing the
corresponding gate returns passed=True with reason="config_unavailable_skip"
so the pipeline degrades gracefully rather than crashing.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Optional

import yaml

from agents.logging_config import get_logger
from agents.settings import (
    data_sources_yaml_path,
    skill_registry_path,
    spatial_units_path,
)
from models.topic_schema import IdentificationPrimary, Topic

logger = get_logger(__name__)


# ── GateResult ────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    gate_id: str
    name: str
    passed: bool
    refinable: bool
    reason: str
    details: dict = field(default_factory=dict)
    score: Optional[int] = None       # for LLM-judged gates (1-5)
    max_score: Optional[int] = None


# ── RuleEngine ────────────────────────────────────────────────────────────────

class RuleEngine:
    """Loads YAML configs once and exposes per-gate check methods."""

    def __init__(self) -> None:
        self._spatial_units: Optional[dict] = None   # name → granularity_rank
        self._data_sources: Optional[list[dict]] = None
        self._skill_registry: Optional[dict] = None  # id → status
        self._identification_skill_map: Optional[dict] = None

    # ── YAML loaders ─────────────────────────────────────────────────────────

    def _load_spatial_units(self) -> dict[str, int]:
        if self._spatial_units is not None:
            return self._spatial_units
        try:
            with open(spatial_units_path()) as f:
                data = yaml.safe_load(f) or {}
            result: dict[str, int] = {}
            for entry in data.get("spatial_units", []):
                rank = entry["granularity_rank"]
                result[entry["name"].lower()] = rank
                for alias in entry.get("aliases", []):
                    result[alias.lower()] = rank
            self._spatial_units = result
            return result
        except Exception as e:
            logger.warning("spatial_units.yaml unavailable: %s", e)
            self._spatial_units = {}
            return {}

    def _load_data_sources(self) -> list[dict]:
        if self._data_sources is not None:
            return self._data_sources
        try:
            with open(data_sources_yaml_path()) as f:
                data = yaml.safe_load(f) or {}
            self._data_sources = data.get("data_sources", [])
            return self._data_sources
        except Exception as e:
            logger.warning("data_sources.yaml unavailable: %s", e)
            self._data_sources = []
            return []

    def _load_skill_registry(self) -> tuple[dict[str, str], dict[str, str]]:
        """Returns (skill_id → status, identification_method → skill_id)."""
        if self._skill_registry is not None:
            return self._skill_registry, self._identification_skill_map  # type: ignore[return-value]
        try:
            with open(skill_registry_path()) as f:
                data = yaml.safe_load(f) or {}
            skill_status = {s["id"]: s["status"] for s in data.get("skills", [])}
            id_map = data.get("identification_skill_map", {})
            self._skill_registry = skill_status
            self._identification_skill_map = id_map
            return skill_status, id_map
        except Exception as e:
            logger.warning("skill_registry.yaml unavailable: %s", e)
            self._skill_registry = {}
            self._identification_skill_map = {}
            return {}, {}

    # ── Fuzzy name matcher ────────────────────────────────────────────────────

    @staticmethod
    def _fuzzy_match(name: str, candidates: list[str], threshold: float = 0.6) -> bool:
        name_lower = name.lower().strip()
        for cand in candidates:
            ratio = difflib.SequenceMatcher(None, name_lower, cand.lower()).ratio()
            if ratio >= threshold:
                return True
        return False

    # ── G2: scale_alignment ───────────────────────────────────────────────────

    def check_G2_scale_alignment(self, topic: Topic) -> GateResult:
        """Hard-blocker: X spatial_unit and Y spatial_unit must have rank diff ≤ 4."""
        units = self._load_spatial_units()
        if not units:
            return GateResult(
                gate_id="G2", name="scale_alignment",
                passed=True, refinable=False,
                reason="config_unavailable_skip",
            )

        x_unit = topic.exposure_X.spatial_unit.lower().strip()
        y_unit = topic.outcome_Y.spatial_unit.lower().strip()
        x_rank = units.get(x_unit)
        y_rank = units.get(y_unit)

        if x_rank is None or y_rank is None:
            unknown = []
            if x_rank is None:
                unknown.append(x_unit)
            if y_rank is None:
                unknown.append(y_unit)
            return GateResult(
                gate_id="G2", name="scale_alignment",
                passed=True, refinable=False,
                reason=f"unknown_spatial_unit_skip: {unknown}",
                details={"x_unit": x_unit, "y_unit": y_unit},
            )

        diff = abs(x_rank - y_rank)
        passed = diff <= 4
        return GateResult(
            gate_id="G2", name="scale_alignment",
            passed=passed, refinable=False,
            reason="ok" if passed else f"rank_diff={diff} exceeds max=4",
            details={"x_unit": x_unit, "x_rank": x_rank, "y_unit": y_unit,
                     "y_rank": y_rank, "rank_diff": diff},
        )

    # ── G3: data_availability ─────────────────────────────────────────────────

    def check_G3_data_availability(
        self, topic: Topic, declared_sources: list[str]
    ) -> GateResult:
        """Hard-blocker: at least one declared source must be in catalog and cover the topic.

        Only sources that ARE in the catalog are checked for year/spatial coverage.
        Sources not found in the catalog are noted but do not cause a hard block as
        long as at least one declared source is recognized.
        """
        sources = self._load_data_sources()
        if not sources:
            return GateResult(
                gate_id="G3", name="data_availability",
                passed=True, refinable=False,
                reason="config_unavailable_skip",
            )

        if not declared_sources:
            return GateResult(
                gate_id="G3", name="data_availability",
                passed=False, refinable=False,
                reason="no_declared_sources",
            )

        # Build lookup: all names + aliases → entry dict
        catalog: dict[str, dict] = {}
        for entry in sources:
            catalog[entry["name"].lower()] = entry
            for alias in entry.get("alias", []):
                catalog[alias.lower()] = entry

        t_start = topic.temporal_scope.start_year
        t_end = topic.temporal_scope.end_year
        t_spatial = topic.spatial_scope.spatial_unit.lower().strip()

        coverage_issues: list[str] = []   # year/spatial mismatches on recognized sources
        not_in_catalog: list[str] = []    # unknown source names (soft warning only)
        auth_warnings: list[str] = []
        recognized_count = 0

        for src in declared_sources:
            # Fuzzy catalog lookup
            entry = catalog.get(src.lower())
            if entry is None:
                all_names = list(catalog.keys())
                if self._fuzzy_match(src, all_names):
                    entry = catalog[
                        max(all_names,
                            key=lambda n: difflib.SequenceMatcher(
                                None, src.lower(), n).ratio())
                    ]
                else:
                    not_in_catalog.append(src)
                    continue

            recognized_count += 1

            if entry.get("auth_required", False):
                auth_warnings.append(src)

            # Year coverage
            y_min = entry.get("coverage_year_min")
            y_max = entry.get("coverage_year_max")
            if y_min is not None and t_start < y_min:
                coverage_issues.append(f"year_gap:{src} starts {y_min} > requested {t_start}")
            if y_max is not None and t_end > y_max:
                coverage_issues.append(f"year_gap:{src} ends {y_max} < requested {t_end}")

            # Spatial unit coverage (warn only if list is non-empty)
            entry_units = [u.lower() for u in entry.get("spatial_units", [])]
            if entry_units and not self._fuzzy_match(t_spatial, entry_units, threshold=0.7):
                coverage_issues.append(f"unit_gap:{src} covers {entry_units} not {t_spatial}")

        if auth_warnings:
            logger.warning("G3: auth_required sources (warn only): %s", auth_warnings)
        if not_in_catalog:
            logger.warning("G3: sources not in catalog (soft warning): %s", not_in_catalog)

        # Hard-block only if: no recognized source at all, or a recognized source has coverage issues
        if recognized_count == 0:
            reason = "no_sources_in_catalog: " + "; ".join(not_in_catalog)
            passed = False
        elif coverage_issues:
            reason = "; ".join(coverage_issues)
            passed = False
        else:
            reason = "ok" if not not_in_catalog else f"ok (unknown_sources_ignored: {not_in_catalog})"
            passed = True

        return GateResult(
            gate_id="G3", name="data_availability",
            passed=passed, refinable=False,
            reason=reason,
            details={"coverage_issues": coverage_issues, "not_in_catalog": not_in_catalog,
                     "auth_warnings": auth_warnings, "declared_sources": declared_sources},
        )

    # ── G6: automation_feasibility ────────────────────────────────────────────

    def check_G6_automation_feasibility(
        self, topic: Topic, declared_sources: list[str]
    ) -> GateResult:
        """Hard-blocker: all skills inferred from sources + method must be available."""
        sources = self._load_data_sources()
        skill_status, id_map = self._load_skill_registry()

        if not skill_status:
            return GateResult(
                gate_id="G6", name="automation_feasibility",
                passed=True, refinable=False,
                reason="config_unavailable_skip",
            )

        # Build source name → entry lookup
        src_lookup: dict[str, dict] = {}
        for entry in sources:
            src_lookup[entry["name"].lower()] = entry
            for alias in entry.get("alias", []):
                src_lookup[alias.lower()] = entry

        required_skills: set[str] = set()

        # Skills from data sources
        for src in declared_sources:
            entry = src_lookup.get(src.lower())
            if entry:
                required_skills.update(entry.get("skills_required", []))

        # Skill from identification method
        method_val = topic.identification.primary.value
        method_skill = id_map.get(method_val)
        if method_skill:
            required_skills.add(method_skill)

        missing = [
            s for s in required_skills
            if skill_status.get(s, "unavailable") != "available"
        ]

        passed = len(missing) == 0
        return GateResult(
            gate_id="G6", name="automation_feasibility",
            passed=passed, refinable=False,
            reason="ok" if passed else f"missing_skills: {missing}",
            details={"required_skills": sorted(required_skills),
                     "missing_skills": missing},
        )

    # ── G4 coverage helper ────────────────────────────────────────────────────

    def check_G4_threat_coverage(self, topic: Topic) -> GateResult:
        """Partial rule check for G4: mitigations must cover ≥ 80% of key_threats.

        The full G4 verdict also requires an LLM quality score; this function
        checks only the deterministic coverage ratio.
        """
        threats = topic.identification.key_threats
        mitigations = topic.identification.mitigations

        if not threats:
            return GateResult(
                gate_id="G4", name="identification_validity",
                passed=True, refinable=True,
                reason="no_threats_declared_skip",
            )

        # Fuzzy matching: a mitigation "covers" a threat if similarity >= 0.5
        covered = 0
        for threat in threats:
            for mitigation in mitigations:
                ratio = difflib.SequenceMatcher(None, threat.lower(), mitigation.lower()).ratio()
                if ratio >= 0.5:
                    covered += 1
                    break

        ratio = covered / len(threats)
        threshold = 0.80
        passed = ratio >= threshold
        return GateResult(
            gate_id="G4", name="identification_validity",
            passed=passed, refinable=True,
            reason="ok" if passed else f"coverage={ratio:.2%} < threshold={threshold:.0%}",
            details={"threats": threats, "mitigations": mitigations,
                     "covered": covered, "coverage_ratio": ratio},
        )

    # ── Convenience: run all hard-blockers ───────────────────────────────────

    def run_hard_blockers(
        self, topic: Topic, declared_sources: list[str]
    ) -> list[GateResult]:
        """Run G2, G3, G6 and return all three results."""
        return [
            self.check_G2_scale_alignment(topic),
            self.check_G3_data_availability(topic, declared_sources),
            self.check_G6_automation_feasibility(topic, declared_sources),
        ]
