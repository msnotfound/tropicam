"""Dependency-free visualisation.

Writes NetPBM images (PPM/PGM) using nothing but numpy, so the engine has a
working eyeball test with zero install friction. A nicer OpenCV/Pygame live
viewer can land later; it should not gate week 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalise(surface: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Scale finite values into [0, 1]; invalid pixels come back as NaN."""
    s = surface.astype(np.float64).copy()
    if valid is not None:
        s[~valid] = np.nan
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s)
    lo = np.nanmin(s[finite])
    hi = np.nanmax(s[finite])
    if hi <= lo:
        out = np.zeros_like(s)
        out[~finite] = np.nan
        return out
    return (s - lo) / (hi - lo)


def colourise_age(surface: np.ndarray, valid: np.ndarray | None = None
                  ) -> np.ndarray:
    """Map a time surface to RGB: recent = hot, stale = cold, unfired = black.

    Deliberately a hand-rolled ramp rather than a perceptual colormap -- this
    is a debugging aid, not a paper figure. Figures come later with real
    plotting.
    """
    n = _normalise(surface, valid)
    finite = np.isfinite(n)
    v = np.where(finite, n, 0.0)

    r = np.clip(1.5 * v - 0.25, 0, 1)
    g = np.clip(1.5 * v - 0.6, 0, 1)
    b = np.clip(1.2 * (1.0 - v) - 0.1, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    rgb[~finite] = 0.0
    return (rgb * 255).astype(np.uint8)


def write_ppm(path: str | Path, rgb: np.ndarray) -> Path:
    """Write an (H, W, 3) uint8 array as a binary PPM."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {rgb.shape}")
    h, w = rgb.shape[:2]
    path = Path(path)
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(rgb.tobytes())
    return path


def save_surface(path: str | Path, surface: np.ndarray,
                 valid: np.ndarray | None = None) -> Path:
    """Colourise a time surface and write it to `path`."""
    return write_ppm(path, colourise_age(surface, valid))
