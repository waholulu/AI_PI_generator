"""Street scene vision feature extractors.

light mode  — Pillow + numpy only; no deep learning; safe as default dependency.
deep mode   — raises ExperimentalFeatureUnavailable; requires opt-in extras.

Policy:
  - Deep vision (SAM, CLIP, segmentation) is never active in default mode.
  - torch / segment-anything are NOT in the default dependency set.
  - To use deep vision, install the 'deepvision' extras AND set
    enable_experimental=True in the compose request.
"""
from __future__ import annotations


class ExperimentalFeatureUnavailable(RuntimeError):
    """Raised when a deep/experimental feature is invoked without opt-in."""


# ---------------------------------------------------------------------------
# Light mode — green share, brightness, edge density (Pillow + numpy only)
# ---------------------------------------------------------------------------

def extract_light_vision_features(image_bytes: bytes) -> dict[str, float]:
    """Extract simple numeric street scene proxies without deep learning.

    Parameters
    ----------
    image_bytes:
        Raw bytes of a JPEG/PNG street scene image.

    Returns
    -------
    dict with float values for each light feature plus a 'feature_mode' key.

    Features
    --------
    green_pixel_share    : fraction of pixels with greenish hue (0–1)
    brightness_mean      : mean luminance across all pixels (0–255)
    sky_proxy_share      : fraction of pixels in the upper third that are
                           bright-blue (rough sky proxy, 0–1)
    edge_density         : Sobel-approximated edge density (0–1)
    gray_surface_proxy   : fraction of pixels with low saturation (0–1)
    feature_mode         : always "light"
    """
    try:
        from PIL import Image  # type: ignore[import]
        import io
        import numpy as np  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "Pillow and numpy are required for light vision features. "
            "Install them or use the vision extras."
        ) from exc

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Green share: green channel dominant and above threshold
    green_mask = (g > 80) & (g > r * 1.1) & (g > b * 1.1)
    green_pixel_share = float(green_mask.mean())

    # Brightness: simple luminance
    brightness_mean = float((0.299 * r + 0.587 * g + 0.114 * b).mean())

    # Sky proxy: upper third, high blue + bright
    upper = arr[: arr.shape[0] // 3, :, :]
    ur, ug, ub = upper[:, :, 0], upper[:, :, 1], upper[:, :, 2]
    sky_mask = (ub > 120) & (ub > ur * 1.1) & (ub > ug * 0.95)
    sky_proxy_share = float(sky_mask.mean())

    # Edge density via finite differences (Sobel approximation)
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(dx**2 + dy**2)
    edge_density = float(np.clip(edges / 255.0, 0, 1).mean())

    # Gray surface proxy: low saturation pixels
    cmax = arr.max(axis=2)
    cmin = arr.min(axis=2)
    saturation = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0)
    gray_surface_proxy = float((saturation < 0.15).mean())

    return {
        "green_pixel_share": round(green_pixel_share, 4),
        "brightness_mean": round(brightness_mean, 2),
        "sky_proxy_share": round(sky_proxy_share, 4),
        "edge_density": round(edge_density, 4),
        "gray_surface_proxy": round(gray_surface_proxy, 4),
        "feature_mode": "light",
    }


# ---------------------------------------------------------------------------
# Deep mode — guardrail only; real implementation lives behind the deepvision extra
# ---------------------------------------------------------------------------

def extract_deep_vision_features(
    image_bytes: bytes,
    model: str = "sam",
    enable_experimental: bool = False,
) -> dict[str, float]:
    """Segment-based street scene features (SAM, CLIP, etc.).

    NOT available in default mode.  Raises ExperimentalFeatureUnavailable
    unless you:
      1. Install the 'deepvision' extras (torch, segment-anything, …)
      2. Pass enable_experimental=True

    This function exists so callers can reference a stable interface
    without accidentally importing heavy dependencies.
    """
    if not enable_experimental:
        raise ExperimentalFeatureUnavailable(
            "Deep vision requires experimental mode. "
            "Set enable_experimental=True and install the 'deepvision' extras."
        )

    # Attempt dynamic import only when explicitly enabled
    try:
        import torch  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        raise ExperimentalFeatureUnavailable(
            "Deep vision extras not installed. "
            "Run: pip install 'ai-pi-generator[deepvision]'"
        ) from exc

    raise NotImplementedError(
        "Deep vision feature extraction is not yet implemented. "
        "Contribute to agents/feature_modules/vision_features.py."
    )
