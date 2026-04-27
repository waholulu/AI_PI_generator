"""Tests for StreetView provider policy enforcement.

These tests verify policy guardrails without network calls.
"""
from __future__ import annotations

import os

import pytest

from agents.feature_modules.streetview_provider import (
    GoogleStreetViewProvider,
    LocalFixtureProvider,
    MapillaryProvider,
    get_provider,
)


# ---------------------------------------------------------------------------
# LocalFixtureProvider
# ---------------------------------------------------------------------------

def test_local_fixture_provider_requires_no_secret():
    provider = LocalFixtureProvider()
    assert provider.required_secrets() == []


def test_local_fixture_provider_always_available():
    provider = LocalFixtureProvider()
    assert provider.is_available() is True
    assert provider.status() == "available"


def test_local_fixture_metadata():
    provider = LocalFixtureProvider()
    meta = provider.get_metadata(42.37, -71.11)
    assert meta["provider"] == "local_fixture"
    assert meta["available"] is True


def test_local_fixture_image_returns_bytes():
    provider = LocalFixtureProvider()
    data = provider.get_image(42.37, -71.11, 0)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_local_fixture_cache_policy():
    provider = LocalFixtureProvider()
    assert provider.cache_policy() == "fixture_only"


# ---------------------------------------------------------------------------
# GoogleStreetViewProvider
# ---------------------------------------------------------------------------

def test_google_provider_refuses_raw_image_caching():
    provider = GoogleStreetViewProvider(api_key="dummy")
    assert provider.cache_policy() == "do_not_cache_raw_images"


def test_google_provider_requires_secret():
    provider = GoogleStreetViewProvider(api_key=None)
    assert "GOOGLE_STREET_VIEW_API_KEY" in provider.required_secrets()


def test_google_provider_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_STREET_VIEW_API_KEY", raising=False)
    provider = GoogleStreetViewProvider(api_key=None)
    assert provider.is_available() is False
    assert provider.status() == "unavailable"


def test_google_provider_available_with_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_STREET_VIEW_API_KEY", "test_key_123")
    provider = GoogleStreetViewProvider()
    assert provider.is_available() is True


def test_google_provider_get_image_raises_without_key():
    provider = GoogleStreetViewProvider(api_key=None)
    with pytest.raises(PermissionError):
        provider.get_image(42.37, -71.11, 0)


# ---------------------------------------------------------------------------
# MapillaryProvider
# ---------------------------------------------------------------------------

def test_mapillary_requires_secret():
    provider = MapillaryProvider(api_key=None)
    assert "MAPILLARY_TOKEN" in provider.required_secrets()


def test_mapillary_cache_policy():
    provider = MapillaryProvider(api_key="dummy")
    assert provider.cache_policy() == "cache_metadata_and_extracted_features_only"


def test_mapillary_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    provider = MapillaryProvider(api_key=None)
    assert provider.is_available() is False


def test_mapillary_metadata_unavailable_without_key():
    provider = MapillaryProvider(api_key=None)
    meta = provider.get_metadata(42.37, -71.11)
    assert meta["available"] is False


def test_mapillary_get_image_raises_without_key():
    provider = MapillaryProvider(api_key=None)
    with pytest.raises(PermissionError):
        provider.get_image(42.37, -71.11, 0)


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------

def test_get_provider_local_fixture_no_experimental():
    provider = get_provider("local_fixture", enable_experimental=False)
    assert isinstance(provider, LocalFixtureProvider)


def test_get_provider_google_requires_experimental():
    with pytest.raises(PermissionError, match="experimental mode"):
        get_provider("google_street_view", enable_experimental=False)


def test_get_provider_mapillary_requires_experimental():
    with pytest.raises(PermissionError, match="experimental mode"):
        get_provider("mapillary", enable_experimental=False)


def test_get_provider_google_with_experimental():
    provider = get_provider("google_street_view", enable_experimental=True)
    assert isinstance(provider, GoogleStreetViewProvider)


def test_get_provider_unknown_name():
    with pytest.raises(ValueError, match="Unknown"):
        get_provider("nonexistent_provider", enable_experimental=True)


# ---------------------------------------------------------------------------
# Policy: unavailable provider must not make candidate "ready"
# ---------------------------------------------------------------------------

def test_unavailable_provider_blocks_candidate_readiness():
    """Simulate the policy check: if provider unavailable, candidate cannot be ready."""
    provider = GoogleStreetViewProvider(api_key=None)
    assert not provider.is_available()

    # Simulate the check that the candidate factory must perform
    missing = provider.missing_secrets()
    assert len(missing) > 0

    # A candidate relying on an unavailable provider is NOT ready
    shortlist_status = "blocked" if not provider.is_available() else "ready"
    assert shortlist_status == "blocked"
