"""Tests for vision feature extractor (light mode + deep mode guardrail).

Light mode tests use the synthetic fixture image (no API keys needed).
Deep mode tests only verify the guardrail raises correctly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.feature_modules.vision_features import (
    ExperimentalFeatureUnavailable,
    extract_deep_vision_features,
    extract_light_vision_features,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "images" / "street_fixture.jpg"


def _fixture_bytes() -> bytes:
    assert _FIXTURE_PATH.exists(), f"Fixture image missing: {_FIXTURE_PATH}"
    return _FIXTURE_PATH.read_bytes()


# ---------------------------------------------------------------------------
# Light mode
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="Street fixture image not found — run fixture generation first",
)
def test_light_vision_features_from_fixture():
    features = extract_light_vision_features(_fixture_bytes())
    assert 0.0 <= features["green_pixel_share"] <= 1.0
    assert "brightness_mean" in features
    assert features["feature_mode"] == "light"


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="Street fixture image not found",
)
def test_light_features_brightness_in_range():
    features = extract_light_vision_features(_fixture_bytes())
    assert 0.0 <= features["brightness_mean"] <= 255.0


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="Street fixture image not found",
)
def test_light_features_edge_density_in_range():
    features = extract_light_vision_features(_fixture_bytes())
    assert 0.0 <= features["edge_density"] <= 1.0


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="Street fixture image not found",
)
def test_light_features_sky_proxy_in_range():
    features = extract_light_vision_features(_fixture_bytes())
    assert 0.0 <= features["sky_proxy_share"] <= 1.0


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="Street fixture image not found",
)
def test_light_features_gray_surface_in_range():
    features = extract_light_vision_features(_fixture_bytes())
    assert 0.0 <= features["gray_surface_proxy"] <= 1.0


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="Street fixture image not found",
)
def test_light_features_all_keys_present():
    features = extract_light_vision_features(_fixture_bytes())
    expected_keys = {
        "green_pixel_share",
        "brightness_mean",
        "sky_proxy_share",
        "edge_density",
        "gray_surface_proxy",
        "feature_mode",
    }
    assert expected_keys.issubset(features.keys())


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="Street fixture image not found",
)
def test_synthetic_fixture_has_sky_signal():
    """Fixture upper-third is blue sky — sky_proxy_share should be detectable."""
    features = extract_light_vision_features(_fixture_bytes())
    # Synthetic image has a blue sky band in the upper third
    assert features["sky_proxy_share"] > 0.0


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="Street fixture image not found",
)
def test_synthetic_fixture_has_green_signal():
    """Fixture has a green strip — green_pixel_share should be detectable."""
    features = extract_light_vision_features(_fixture_bytes())
    assert features["green_pixel_share"] > 0.0


# ---------------------------------------------------------------------------
# Deep mode guardrail
# ---------------------------------------------------------------------------

def test_deep_vision_not_default():
    """Deep vision raises ExperimentalFeatureUnavailable without opt-in."""
    with pytest.raises(ExperimentalFeatureUnavailable):
        extract_deep_vision_features(b"fake_bytes", enable_experimental=False)


def test_deep_vision_raises_without_experimental_flag():
    with pytest.raises(ExperimentalFeatureUnavailable, match="experimental mode"):
        extract_deep_vision_features(b"", enable_experimental=False)


def test_deep_vision_experimental_flag_still_needs_extras():
    """Even with enable_experimental=True, torch must be installed."""
    try:
        import torch  # noqa: F401
        # torch is installed — the function will raise NotImplementedError
        with pytest.raises(NotImplementedError):
            extract_deep_vision_features(b"", enable_experimental=True)
    except ImportError:
        # torch not installed — should raise ExperimentalFeatureUnavailable
        with pytest.raises(ExperimentalFeatureUnavailable, match="deepvision"):
            extract_deep_vision_features(b"", enable_experimental=True)
