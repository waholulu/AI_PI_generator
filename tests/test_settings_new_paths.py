"""Tests for new path helpers added to agents/settings.py — Day 1 TDD."""

import os
from pathlib import Path

import pytest

import agents.settings as settings


# ── repo_config_dir ───────────────────────────────────────────────────────────

def test_repo_config_dir_is_absolute():
    p = settings.repo_config_dir()
    assert p.is_absolute()


def test_repo_config_dir_ends_with_config():
    p = settings.repo_config_dir()
    assert p.name == "config"


def test_repo_config_dir_not_under_autopi_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    p = settings.repo_config_dir()
    # Should be repo-relative, NOT under AUTOPI_DATA_ROOT
    assert not str(p).startswith(str(tmp_path))


# ── YAML path helpers ─────────────────────────────────────────────────────────

def test_reflection_config_path_ends_correctly():
    assert settings.reflection_config_path().endswith("reflection_config.yaml")


def test_gate_config_path_ends_correctly():
    assert settings.gate_config_path().endswith("gate_config.yaml")


def test_data_sources_yaml_path_ends_correctly():
    assert settings.data_sources_yaml_path().endswith("data_sources.yaml")


def test_skill_registry_path_ends_correctly():
    assert settings.skill_registry_path().endswith("skill_registry.yaml")


def test_spatial_units_path_ends_correctly():
    assert settings.spatial_units_path().endswith("spatial_units.yaml")


def test_refine_operations_path_ends_correctly():
    assert settings.refine_operations_path().endswith("refine_operations.yaml")


def test_all_yaml_config_files_exist():
    for fn in [
        settings.reflection_config_path,
        settings.gate_config_path,
        settings.data_sources_yaml_path,
        settings.skill_registry_path,
        settings.spatial_units_path,
        settings.refine_operations_path,
    ]:
        p = Path(fn())
        assert p.exists(), f"Expected YAML to exist: {p}"


def test_all_yaml_configs_are_loadable():
    import yaml

    for fn in [
        settings.reflection_config_path,
        settings.gate_config_path,
        settings.data_sources_yaml_path,
        settings.skill_registry_path,
        settings.spatial_units_path,
        settings.refine_operations_path,
    ]:
        path = fn()
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data is not None, f"YAML loaded as None: {path}"


# ── tentative_pool_path ───────────────────────────────────────────────────────

def test_tentative_pool_path_ends_correctly(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    p = settings.tentative_pool_path()
    assert p.endswith("tentative_pool.json")


def test_tentative_pool_path_is_under_output(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    p = settings.tentative_pool_path()
    assert "output" in p


# ── ideation_traces_dir ───────────────────────────────────────────────────────

def test_ideation_traces_dir_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    d = settings.ideation_traces_dir()
    assert d.exists()
    assert d.is_dir()


def test_ideation_traces_dir_under_output(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    d = settings.ideation_traces_dir()
    assert "output" in str(d)
    assert "ideation_traces" in str(d)


# ── ideation_run_summary_path ─────────────────────────────────────────────────

def test_ideation_run_summary_path_ends_correctly(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    p = settings.ideation_run_summary_path()
    assert p.endswith("ideation_run_summary.json")


# ── run_scope isolation ───────────────────────────────────────────────────────

def test_tentative_pool_path_changes_with_run_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    p_global = settings.tentative_pool_path()

    token = settings.activate_run_scope("run_abc")
    try:
        p_scoped = settings.tentative_pool_path()
    finally:
        settings.deactivate_run_scope(token)

    assert p_global != p_scoped
    assert "run_abc" in p_scoped
