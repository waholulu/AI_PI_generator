# Experimental Mode

Experimental mode unlocks street view imagery, deep learning, and paid-API data sources for research candidates.  It is **off by default** and protected by multiple guardrails.

## What is disabled by default

| Feature | Default | Experimental only |
|---------|---------|-------------------|
| Street View (Google) | ✗ blocked | ✓ with key |
| Street View (Mapillary) | ✗ blocked | ✓ with key |
| SAM / CLIP segmentation | ✗ blocked | ✓ with extras |
| Deep learning features | ✗ blocked | ✓ with extras |
| Paid API sources | ✗ blocked | ✓ explicit |
| `LocalFixtureProvider` | ✓ always | ✓ |
| OSMnx (OpenStreetMap) | ✓ default | ✓ |
| NLCD / EPA / CDC | ✓ default | ✓ |

---

## How to enable

### CLI / API

Pass `enable_experimental=true` in the compose request:

```bash
# API
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"template_id": "built_environment_health",
       "domain_input": "...",
       "enable_experimental": true}'

# Python
from models.candidate_composer_schema import ComposeRequest
req = ComposeRequest(..., enable_experimental=True)
```

### Required secrets

Each provider declares its required env-vars:

| Provider | Required env-var |
|----------|-----------------|
| Google Street View | `GOOGLE_STREET_VIEW_API_KEY` |
| Mapillary | `MAPILLARY_TOKEN` |

Set secrets via Railway:

```bash
railway variables set GOOGLE_STREET_VIEW_API_KEY=...
railway variables set MAPILLARY_TOKEN=...
```

---

## Provider policy (hard-coded, non-configurable)

### Google Street View

- `cache_policy = "do_not_cache_raw_images"` — **no exceptions**
- Raw images must never be written to disk or object storage
- Only extracted numeric features may be persisted

### Mapillary

- `cache_policy = "cache_metadata_and_extracted_features_only"`
- Metadata and computed features may be cached
- Raw image bytes must not be stored

### LocalFixtureProvider

- `cache_policy = "fixture_only"`
- Safe in CI, default mode, no keys required
- Used for tests and smoke runs without real imagery

---

## Guardrails

### Composition time

- Experimental-tier sources are filtered out when `enable_experimental=False`
- No exception: the filter is in `_pick_source()` and cannot be bypassed via config

### Scoring time

- `automation_risk=high` → `overall_score` capped at **0.65**
- `required_secrets` present → `automation_feasibility` capped at **0.45**
- Experimental candidates cannot be ranked #1 without explicit user promotion

### Provider factory

```python
from agents.feature_modules.streetview_provider import get_provider

# This raises PermissionError if enable_experimental=False:
provider = get_provider("google_street_view", enable_experimental=False)

# This works:
provider = get_provider("local_fixture", enable_experimental=False)
```

### Status check

```python
provider = GoogleStreetViewProvider()
if not provider.is_available():
    # API key not set → candidate cannot be "ready"
    shortlist_status = "blocked"
    missing = provider.missing_secrets()  # ["GOOGLE_STREET_VIEW_API_KEY"]
```

---

## Street View provider interface

```python
from agents.feature_modules.streetview_provider import (
    LocalFixtureProvider,
    MapillaryProvider,
    GoogleStreetViewProvider,
    get_provider,
)

# Default: fixture only
provider = get_provider("local_fixture")
meta = provider.get_metadata(42.37, -71.11)
img_bytes = provider.get_image(42.37, -71.11, heading=0)

# Experimental: Mapillary
provider = get_provider("mapillary", enable_experimental=True)
# Requires MAPILLARY_TOKEN in env
```

---

## Vision features

### Light mode (default)

Light mode is always available with `pip install pillow numpy`.  It computes simple numeric proxies from image bytes:

```python
from agents.feature_modules.vision_features import extract_light_vision_features

features = extract_light_vision_features(img_bytes)
# {
#   "green_pixel_share": 0.23,
#   "brightness_mean": 128.4,
#   "sky_proxy_share": 0.31,
#   "edge_density": 0.17,
#   "gray_surface_proxy": 0.44,
#   "feature_mode": "light"
# }
```

### Deep mode (experimental)

Deep mode (SAM, CLIP, segmentation) requires:

1. `pip install 'ai-pi-generator[deepvision]'`
2. `enable_experimental=True`

Without both, calling `extract_deep_vision_features()` raises `ExperimentalFeatureUnavailable`.

```python
from agents.feature_modules.vision_features import (
    extract_deep_vision_features,
    ExperimentalFeatureUnavailable,
)

try:
    features = extract_deep_vision_features(img_bytes, enable_experimental=True)
except ExperimentalFeatureUnavailable as e:
    print(f"Not available: {e}")
```

Deep vision model weights and torch are **not in the default Docker image**.

---

## Testing experimental mode

Tests for experimental guardrails do not require real API keys:

```bash
uv run python -m pytest tests/test_streetview_provider_policy.py -v
uv run python -m pytest tests/test_vision_features.py -v
```

Vision light mode tests require the `vision` extras:

```bash
uv run --extra vision python -m pytest tests/test_vision_features.py -v
```
