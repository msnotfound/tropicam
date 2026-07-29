"""Render each oracle scene's time surface to out/ as a PPM.

The eyeball test: the bar should be a clean left-to-right ramp, the dot a
radial bloom, and the corner two ramps meeting at a visible crease -- that
crease is what motion-regime decomposition (C2) has to find.

Usage:
    python3 scripts/demo_oracle.py [--out out]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tropicam.events import event_rate, duration_us  # noqa: E402
from tropicam.render import save_surface  # noqa: E402
from tropicam.synthetic import (  # noqa: E402
    translating_bar, moving_dot, translating_corner, inject_background_noise,
)
from tropicam.time_surface import TimeSurface  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = args.size
    scenes = [
        translating_bar(n, n, vx=0.05),
        moving_dot(n, n, vx=0.04, vy=0.03, radius=12, c0=(20.0, 20.0)),
        translating_corner(n, n, vx=0.05, vy=0.03),
    ]
    scenes.append(inject_background_noise(
        translating_bar(n, n, vx=0.05), rate_hz_per_px=3.0,
        rng=np.random.default_rng(0)))
    scenes[-1].name = "translating_bar_noisy"

    for scene in scenes:
        ts = TimeSurface(scene.height, scene.width)
        ts.update(scene.events)
        surface = ts.merged()
        path = save_surface(args.out / f"{scene.name}.ppm", surface,
                            valid=ts.has_fired().any(axis=0))
        print(f"{scene.name:26s} {scene.events.size:8,} ev  "
              f"{duration_us(scene.events)/1000:7.1f} ms  "
              f"{event_rate(scene.events)/1e6:6.2f} Mev/s  -> {path}")

    # The corner scene carries the regime ground truth; dump it too, since E2
    # scores recovered regimes against exactly this map.
    corner = scenes[2]
    save_surface(args.out / "corner_regimes.ppm",
                 corner.extra["regime_map"].astype(float),
                 valid=corner.extra["regime_map"] >= 0)
    save_surface(args.out / "corner_crease.ppm",
                 corner.extra["crease_mask"].astype(float))
    print(f"\nground truth: corner has {corner.n_regimes} regimes, "
          f"{corner.extra['crease_mask'].sum()} crease pixels")
    print(f"wrote {len(list(args.out.glob('*.ppm')))} images to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
