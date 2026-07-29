"""The synthetic oracle must match its own closed form.

If these fail, nothing downstream can be trusted -- every operator in the
engine is validated against these surfaces.

Runs under pytest, or standalone: ``python3 tests/test_synthetic.py``
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tropicam.events import is_sorted  # noqa: E402
from tropicam.synthetic import (  # noqa: E402
    translating_bar, moving_dot, translating_corner,
    inject_background_noise, inject_threshold_mismatch,
)
from tropicam.time_surface import TimeSurface  # noqa: E402


def _built_surface(scene, polarity=1):
    ts = TimeSurface(scene.height, scene.width)
    ts.update(scene.events)
    return ts.plane(polarity), ts


def test_all_scenes_are_time_sorted_and_nonempty():
    for scene in (translating_bar(), moving_dot(), translating_corner()):
        assert scene.events.size > 0, scene.name
        assert is_sorted(scene.events), scene.name
        assert scene.events["t"].min() > 0, "t=0 collides with the sentinel"


def test_bar_surface_is_an_affine_ramp():
    """The time surface of a translating bar is a flat ramp with slope 1/vx.

    This is Benosman's observation, not ours (doc section 6.1) -- but the
    oracle still has to reproduce it exactly.
    """
    vx, x0, t0 = 0.05, 5.0, 1000
    scene = translating_bar(vx=vx, x0=x0, t0=t0)
    built, _ = _built_surface(scene)

    swept = np.isfinite(scene.analytic_surface)
    err = np.abs(built[swept].astype(float) - scene.analytic_surface[swept])
    assert err.max() <= 1.0, f"max deviation {err.max()} us exceeds rounding"

    # Slope along x is 1/vx; slope along y is zero.
    row = built[64].astype(float)
    dx = np.diff(row[10:])
    assert np.allclose(dx, 1.0 / vx, atol=1.0)
    col = built[:, 64].astype(float)
    assert np.allclose(np.diff(col), 0.0, atol=1.0)


def test_bar_constrains_normal_flow_only():
    """Aperture problem, stated as a test rather than discovered as a bug.

    A vertical edge carries no evidence about vertical motion. The surface is
    exactly invariant along y, so any v_y is consistent with it -- which is
    why the bar is labelled `constrains="normal"` and why the dot scene
    exists.
    """
    scene = translating_bar()
    assert scene.constrains == "normal"
    built, _ = _built_surface(scene)
    swept = np.isfinite(scene.analytic_surface)
    rows = [built[r][swept[r]] for r in (20, 60, 100)]
    assert all(np.array_equal(rows[0], r) for r in rows[1:])


def test_dot_constrains_full_flow():
    """A curved boundary pins both components; check against the closed form."""
    scene = moving_dot(vx=0.04, vy=0.03)
    assert scene.constrains == "full"
    built, _ = _built_surface(scene)

    swept = np.isfinite(scene.analytic_surface)
    err = np.abs(built[swept].astype(float) - scene.analytic_surface[swept])
    assert err.max() <= 1.0

    # Unlike the bar, the surface genuinely varies in both directions.
    gy, gx = np.gradient(np.where(swept, built.astype(float), np.nan))
    assert np.nanstd(gy) > 0.5 and np.nanstd(gx) > 0.5


def test_corner_is_a_max_of_two_affine_pieces():
    """The C2 / E2 oracle: exactly two regimes and a known crease.

    T(x, y) = max((x - cx0)/vx, (y - cy0)/vy) is a tropical polynomial with
    one crease. We know the regime count is 2 and where the crease lies, which
    is what experiment E2 needs to score against.
    """
    vx, vy = 0.05, 0.03
    scene = translating_corner(vx=vx, vy=vy)
    assert scene.n_regimes == 2

    built, _ = _built_surface(scene)
    swept = np.isfinite(scene.analytic_surface)
    err = np.abs(built[swept].astype(float) - scene.analytic_surface[swept])
    assert err.max() <= 1.0

    regime = scene.extra["regime_map"]
    assert set(np.unique(regime)) <= {-1, 0, 1}
    assert (regime == 0).sum() > 100 and (regime == 1).sum() > 100

    # Each regime carries the gradient of its own affine piece.
    surf = scene.analytic_surface
    gy, gx = np.gradient(surf)
    interior0 = (regime == 0) & swept
    interior1 = (regime == 1) & swept
    assert np.isclose(np.nanmedian(gx[interior0]), 1.0 / vx, rtol=0.05)
    assert np.isclose(np.nanmedian(gy[interior1]), 1.0 / vy, rtol=0.05)

    assert scene.extra["crease_mask"].any()


def test_background_noise_only_adds_events():
    scene = translating_bar()
    noisy = inject_background_noise(scene, rate_hz_per_px=2.0,
                                   rng=np.random.default_rng(1))
    assert noisy.events.size > scene.events.size
    assert is_sorted(noisy.events)
    assert noisy.extra["n_noise_events"] > 0
    # Ground truth is untouched -- E1 compares against the same oracle.
    assert noisy.velocity == scene.velocity


def test_threshold_mismatch_is_a_bounded_time_offset():
    """C1's premise, as an executable assertion.

    Mismatch must shift *when* a pixel fires by a bounded amount, and must not
    reorder that pixel's own events. If this ever fails, the certificate's
    model of the perturbation is wrong.
    """
    scene = translating_bar()
    sigma = 0.03
    perturbed, offsets = inject_threshold_mismatch(
        scene, sigma_contrast=sigma, contrast_rate=1e-3,
        rng=np.random.default_rng(3))

    assert perturbed.events.size == scene.events.size
    assert offsets.shape == (scene.height, scene.width)

    bound = np.abs(offsets).max() + 1.0  # +1 for rounding to integer us
    # Compare per pixel: deviation of each pixel's ON timestamp.
    a = TimeSurface(scene.height, scene.width); a.update(scene.events)
    b = TimeSurface(scene.height, scene.width); b.update(perturbed.events)
    fired = a.has_fired() & b.has_fired()
    dev = np.abs(a.timestamps[fired].astype(float)
                 - b.timestamps[fired].astype(float))
    assert dev.max() <= bound, f"deviation {dev.max()} exceeded bound {bound}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
