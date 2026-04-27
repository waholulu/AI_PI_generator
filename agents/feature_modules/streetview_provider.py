"""Street View provider abstraction for experimental vision-based candidates.

Policy constraints (hard-coded, not configurable):
  - Google Street View: raw image caching is FORBIDDEN
  - Mapillary: only metadata and extracted features may be cached
  - Non-fixture providers require explicit secrets and experimental mode
  - Providers without valid credentials report status="unavailable"
  - Unavailable providers MUST NOT allow candidates to reach "ready" status

Only LocalFixtureProvider is active without experimental mode.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StreetViewProvider(ABC):
    """Abstract base for all street view providers."""

    @abstractmethod
    def get_metadata(self, lat: float, lon: float) -> dict[str, Any]:
        """Return availability metadata for a coordinate."""

    @abstractmethod
    def get_image(self, lat: float, lon: float, heading: int) -> bytes:
        """Return raw image bytes for a coordinate + heading.

        Callers must respect cache_policy() before storing anything.
        """

    @abstractmethod
    def cache_policy(self) -> str:
        """Describe what may and may not be cached.

        Known values:
          "fixture_only"
          "cache_metadata_and_extracted_features_only"
          "do_not_cache_raw_images"
        """

    @abstractmethod
    def required_secrets(self) -> list[str]:
        """List env-var names that must be set for this provider to be active."""

    def status(self) -> str:
        """Return 'available' or 'unavailable' based on secret resolution."""
        missing = [s for s in self.required_secrets() if not os.environ.get(s)]
        return "unavailable" if missing else "available"

    def is_available(self) -> bool:
        return self.status() == "available"

    def missing_secrets(self) -> list[str]:
        return [s for s in self.required_secrets() if not os.environ.get(s)]


# ---------------------------------------------------------------------------
# LocalFixtureProvider — no secrets required; safe in CI and default mode
# ---------------------------------------------------------------------------

_FIXTURE_IMAGE_PATH = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "images" / "street_fixture.jpg"


def _load_fixture_image_bytes() -> bytes:
    if _FIXTURE_IMAGE_PATH.exists():
        return _FIXTURE_IMAGE_PATH.read_bytes()
    # Minimal valid 1×1 white JPEG (no PIL dependency)
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
        b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05"
        b"\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06"
        b"\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br"
        b"\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZ"
        b"cdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94"
        b"\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa"
        b"\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7"
        b"\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3"
        b"\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8"
        b"\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03"
        b"\xff\xd9"
    )


class LocalFixtureProvider(StreetViewProvider):
    """Always-available provider backed by a local fixture image.

    Safe in CI, default mode, and without any API keys.
    """

    def get_metadata(self, lat: float, lon: float) -> dict[str, Any]:
        return {"provider": "local_fixture", "available": True, "lat": lat, "lon": lon}

    def get_image(self, lat: float, lon: float, heading: int) -> bytes:
        return _load_fixture_image_bytes()

    def cache_policy(self) -> str:
        return "fixture_only"

    def required_secrets(self) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# MapillaryProvider — experimental; metadata + feature cache only
# ---------------------------------------------------------------------------

class MapillaryProvider(StreetViewProvider):
    """Mapillary street-level imagery.

    Cache policy: metadata and extracted features only (no raw images).
    Requires MAPILLARY_TOKEN env var.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("MAPILLARY_TOKEN")

    def get_metadata(self, lat: float, lon: float) -> dict[str, Any]:
        if not self._api_key:
            return {"provider": "mapillary", "available": False, "reason": "missing_token"}
        return {"provider": "mapillary", "available": True, "lat": lat, "lon": lon}

    def get_image(self, lat: float, lon: float, heading: int) -> bytes:
        if not self._api_key:
            raise PermissionError("MAPILLARY_TOKEN not set; provider unavailable")
        raise NotImplementedError(
            "Mapillary image download not implemented.  "
            "Cache policy: cache_metadata_and_extracted_features_only. "
            "Raw images must not be stored."
        )

    def cache_policy(self) -> str:
        return "cache_metadata_and_extracted_features_only"

    def required_secrets(self) -> list[str]:
        return ["MAPILLARY_TOKEN"]


# ---------------------------------------------------------------------------
# GoogleStreetViewProvider — experimental; NO raw image caching (policy)
# ---------------------------------------------------------------------------

class GoogleStreetViewProvider(StreetViewProvider):
    """Google Street View Static API.

    Cache policy: raw images MUST NOT be cached (Google ToS + privacy policy).
    Requires GOOGLE_STREET_VIEW_API_KEY env var.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_STREET_VIEW_API_KEY")

    def get_metadata(self, lat: float, lon: float) -> dict[str, Any]:
        if not self._api_key:
            return {"provider": "google_street_view", "available": False, "reason": "missing_api_key"}
        return {"provider": "google_street_view", "available": True, "lat": lat, "lon": lon}

    def get_image(self, lat: float, lon: float, heading: int) -> bytes:
        if not self._api_key:
            raise PermissionError("GOOGLE_STREET_VIEW_API_KEY not set; provider unavailable")
        raise NotImplementedError(
            "Google Street View image fetch not implemented.  "
            "IMPORTANT: cache_policy is 'do_not_cache_raw_images'. "
            "Raw images must NEVER be written to disk or object storage."
        )

    def cache_policy(self) -> str:
        # Hard-coded per Google ToS and project policy — do not change.
        return "do_not_cache_raw_images"

    def required_secrets(self) -> list[str]:
        return ["GOOGLE_STREET_VIEW_API_KEY"]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, type[StreetViewProvider]] = {
    "local_fixture": LocalFixtureProvider,
    "mapillary": MapillaryProvider,
    "google_street_view": GoogleStreetViewProvider,
}


def get_provider(name: str, enable_experimental: bool = False) -> StreetViewProvider:
    """Return a StreetViewProvider by name.

    Non-fixture providers are only allowed when enable_experimental=True.
    Raises ValueError for unknown names or policy violations.
    """
    if name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown street view provider: {name!r}. Choose from {list(PROVIDER_REGISTRY)}")

    if name != "local_fixture" and not enable_experimental:
        raise PermissionError(
            f"Provider {name!r} requires experimental mode. "
            "Set enable_experimental=True and supply the required API key."
        )

    return PROVIDER_REGISTRY[name]()
