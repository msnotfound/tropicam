"""Synthetic oracle: event streams with analytically known ground truth.

Every operator in the engine validates against these before it is allowed near
real data (doc section 7, build order step 1). Each generator emits events at
the *exact* time a moving shape's boundary crosses a pixel, so the resulting
time surface has a closed form we can compare against.

Three scenes, and the distinction between them is load-bearing:

``translating_bar``
    A vertical bar moving in +x. **Constrains normal flow only.** A vertical
    edge carries no information about its own vertical motion, so a "wrong"
    v_y recovered here may be correct behaviour, not a bug. This is the
    aperture problem, and it is the classic way to fool yourself with a bar
    test.

``moving_dot``
    A disc. Its boundary curves, so **both flow components are recoverable**.
    This is the test that actually validates full flow.

``translating_corner``
    A quadrant. Its time surface is exactly ``max(affine, affine)`` -- a
    tropical polynomial with one analytically known crease. This is the
    oracle for motion-regime decomposition (C2 / experiment E2): we know the
    true regime count is 2 and we know where the crease lies.

All streams start at ``t0 > 0`` because timestamp 0 is the "never fired"
sentinel in `TimeSurface`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .events import ON, OFF, make_events


@dataclass
class Scene:
    """A synthetic stream plus everything we know analytically about it."""

    name: str
    events: np.ndarray
    height: int
    width: int

    #: True velocity in pixels per microsecond, (vx, vy).
    velocity: tuple[float, float]

    #: Number of distinct locally-affine motion regimes in the time surface.
    n_regimes: int

    #: Which flow components this scene actually constrains. A bar constrains
    #: only the edge-normal component; asking it for full flow is a category
    #: error, not a test failure.
    constrains: str  # "normal" | "full"

    #: T(x, y) in microseconds for swept pixels, NaN elsewhere. The closed-form
    #: surface the engine should reproduce.
    analytic_surface: np.ndarray | None = None

    #: Extra scene-specific ground truth (e.g. the crease mask).
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Scene({self.name!r}, {self.events.size} events, "
                f"{self.width}x{self.height}, v={self.velocity}, "
                f"{self.n_regimes} regime(s), constrains={self.constrains})")


def _pack(xs, ys, ts, ps, t0):
    """Round exact float times to integer microseconds and pack."""
    ts = np.rint(ts).astype(np.int64) + int(t0)
    keep = ts > 0
    return make_events(xs[keep], ys[keep], ts[keep].astype(np.uint32), ps[keep])


def translating_bar(height: int = 128, width: int = 128, *, vx: float = 0.05,
                    bar_width: int = 4, x0: float = 5.0, t0: int = 1000,
                    emit_off: bool = True) -> Scene:
    """Vertical bar of width `bar_width` translating in +x at `vx` px/us.

    Leading edge fires ON, trailing edge fires OFF. The ON-plane time surface
    is exactly ``t0 + (x - x0) / vx`` -- a flat affine ramp whose slope in x is
    ``1 / vx``. Reading that slope is the whole flow idea in one line.

    Constrains normal flow only; see module docstring.
    """
    if vx <= 0:
        raise ValueError("vx must be positive for this generator")

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xs = xx.ravel().astype(np.int64)
    ys = yy.ravel().astype(np.int64)

    t_lead = (xs - x0) / vx
    valid = t_lead >= 0

    ev_x = xs[valid]
    ev_y = ys[valid]
    ev_t = t_lead[valid]
    ev_p = np.full(ev_x.size, ON)

    if emit_off:
        t_trail = (xs - x0 - bar_width) / vx
        v2 = t_trail >= 0
        ev_x = np.concatenate([ev_x, xs[v2]])
        ev_y = np.concatenate([ev_y, ys[v2]])
        ev_t = np.concatenate([ev_t, t_trail[v2]])
        ev_p = np.concatenate([ev_p, np.full(int(v2.sum()), OFF)])

    surface = np.full((height, width), np.nan)
    lead_grid = (xx - x0) / vx
    surface[lead_grid >= 0] = (lead_grid + t0)[lead_grid >= 0]

    return Scene(
        name="translating_bar",
        events=_pack(ev_x, ev_y, ev_t, ev_p, t0),
        height=height, width=width,
        velocity=(vx, 0.0),
        n_regimes=1,
        constrains="normal",
        analytic_surface=surface,
        extra={"bar_width": bar_width, "x0": x0, "normal": (1.0, 0.0)},
    )


def moving_dot(height: int = 128, width: int = 128, *, vx: float = 0.04,
               vy: float = 0.03, radius: float = 6.0,
               c0: tuple[float, float] = (10.0, 10.0), t0: int = 1000,
               emit_off: bool = True) -> Scene:
    """Disc of `radius` translating at (vx, vy) px/us.

    A pixel is covered while ``|p - c(t)| <= radius``. That is a quadratic in
    t; the earlier root is the ON (entry) event, the later root the OFF (exit)
    event. Curved boundary means **both** flow components are constrained, so
    this is the scene that validates full flow.
    """
    v = np.array([vx, vy], dtype=float)
    speed_sq = float(v @ v)
    if speed_sq == 0:
        raise ValueError("dot must be moving")

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    dx = xx.ravel().astype(float) - c0[0]
    dy = yy.ravel().astype(float) - c0[1]

    dv = dx * vx + dy * vy
    disc = dv * dv - speed_sq * (dx * dx + dy * dy - radius * radius)
    swept = disc >= 0

    root = np.sqrt(np.maximum(disc, 0.0))
    t_enter = (dv - root) / speed_sq
    t_exit = (dv + root) / speed_sq

    xs = xx.ravel().astype(np.int64)
    ys = yy.ravel().astype(np.int64)

    on = swept & (t_enter >= 0)
    ev_x, ev_y, ev_t = xs[on], ys[on], t_enter[on]
    ev_p = np.full(int(on.sum()), ON)

    if emit_off:
        off = swept & (t_exit >= 0)
        ev_x = np.concatenate([ev_x, xs[off]])
        ev_y = np.concatenate([ev_y, ys[off]])
        ev_t = np.concatenate([ev_t, t_exit[off]])
        ev_p = np.concatenate([ev_p, np.full(int(off.sum()), OFF)])

    surface = np.full((height, width), np.nan)
    flat = surface.ravel()
    flat[on] = t_enter[on] + t0

    return Scene(
        name="moving_dot",
        events=_pack(ev_x, ev_y, ev_t, ev_p, t0),
        height=height, width=width,
        velocity=(vx, vy),
        n_regimes=1,
        constrains="full",
        analytic_surface=surface,
        extra={"radius": radius, "c0": c0},
    )


def translating_corner(height: int = 128, width: int = 128, *,
                       vx: float = 0.05, vy: float = 0.03,
                       c0: tuple[float, float] = (4.0, 4.0),
                       t0: int = 1000) -> Scene:
    """A filled quadrant whose corner translates at (vx, vy) px/us.

    The quadrant covers ``x <= cx(t) and y <= cy(t)``, so a pixel is first
    covered at::

        T(x, y) = max( (x - cx0) / vx , (y - cy0) / vy )

    which is a max of two affine functions -- a tropical polynomial with
    exactly one crease. Two locally-affine motion regimes, analytically known,
    with the crease at the locus where the two arguments are equal.

    This is the ground truth for experiment E2: the engine should recover
    ``n_regimes == 2`` and place the crease where we know it is.
    """
    if vx <= 0 or vy <= 0:
        raise ValueError("vx and vy must be positive for this generator")

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    a = (xx - c0[0]) / vx  # vertical-edge regime: gradient (1/vx, 0)
    b = (yy - c0[1]) / vy  # horizontal-edge regime: gradient (0, 1/vy)

    t_cover = np.maximum(a, b)
    valid = t_cover >= 0

    # Which affine piece attains the max -- the regime label.
    regime = (b > a).astype(np.int8)  # 0 -> vertical edge, 1 -> horizontal
    # Crease: pixels where the two pieces are within one timestep of each other.
    crease = np.abs(a - b) < (0.5 / max(vx, vy))

    xs = xx[valid].astype(np.int64)
    ys = yy[valid].astype(np.int64)
    ts = t_cover[valid]
    ps = np.full(xs.size, ON)

    surface = np.full((height, width), np.nan)
    surface[valid] = t_cover[valid] + t0

    return Scene(
        name="translating_corner",
        events=_pack(xs, ys, ts, ps, t0),
        height=height, width=width,
        velocity=(vx, vy),
        n_regimes=2,
        constrains="full",
        analytic_surface=surface,
        extra={
            "c0": c0,
            "regime_map": np.where(valid, regime, -1),
            "crease_mask": crease & valid,
            "regime_gradients": ((1.0 / vx, 0.0), (0.0, 1.0 / vy)),
        },
    )


# -- perturbations ------------------------------------------------------
#
# Both live here rather than in a separate module because the doc's build
# order is explicit that noise injection should ride on the oracle
# infrastructure we needed anyway -- experiments E1 and E3 then come close to
# free.


def inject_background_noise(scene: Scene, rate_hz_per_px: float = 1.0, *,
                            rng: np.random.Generator | None = None) -> Scene:
    """Add uniformly-distributed background-activity events (experiment E1).

    Realistic rates are ~0.1-10 events/px/s, higher in low light. Each spurious
    event writes a high timestamp into an otherwise quiet pixel, which is
    exactly the 1/n breakdown point max-plus suffers from -- and exactly what
    the max-plus opening is meant to remove.
    """
    rng = np.random.default_rng() if rng is None else rng
    ev = scene.events
    if ev.size == 0:
        return scene

    span_us = int(ev["t"][-1]) - int(ev["t"][0])
    n_px = scene.height * scene.width
    n_noise = int(rate_hz_per_px * n_px * span_us * 1e-6)
    if n_noise == 0:
        return scene

    nx = rng.integers(0, scene.width, n_noise)
    ny = rng.integers(0, scene.height, n_noise)
    nt = rng.integers(int(ev["t"][0]), int(ev["t"][-1]) + 1, n_noise)
    np_ = rng.integers(0, 2, n_noise)

    merged = make_events(
        np.concatenate([ev["x"], nx]),
        np.concatenate([ev["y"], ny]),
        np.concatenate([ev["t"], nt]),
        np.concatenate([ev["p"], np_]),
    )
    out = Scene(**{**scene.__dict__, "events": merged})
    out.extra = {**scene.extra, "noise_rate_hz_per_px": rate_hz_per_px,
                 "n_noise_events": n_noise}
    return out


def inject_threshold_mismatch(scene: Scene, sigma_contrast: float = 0.03, *,
                              contrast_rate: float = 1.0,
                              rng: np.random.Generator | None = None
                              ) -> tuple[Scene, np.ndarray]:
    """Per-pixel contrast-threshold mismatch as a time-axis offset (E3 / C1).

    This is the first-order model the certificate is built on: a pixel whose
    threshold is off by dC fires early or late by roughly ``dC / (dlogI/dt)``,
    a *static per-pixel offset along the time axis*. It shifts **when** a pixel
    fires, never the ordering of that pixel's own events -- which is precisely
    why it lands in max-plus as a non-flat structuring function rather than as
    unstructured noise.

    Returns the perturbed scene and the offset field (microseconds), so a test
    can check the measured output deviation against a bound derived from the
    field's magnitude.

    Caveat: this is the idealised model for fast iteration and for sanity-
    checking the bound. Reported E3 numbers must come from EVIS or v2e, which
    simulate the sensor circuit rather than assuming the linearisation.
    `contrast_rate` stands in for the local temporal contrast slope and is a
    scene-wide constant here, which real data will not respect.
    """
    rng = np.random.default_rng() if rng is None else rng
    offsets_us = rng.normal(0.0, sigma_contrast / contrast_rate,
                            size=(scene.height, scene.width))

    ev = scene.events.copy()
    per_event = offsets_us[ev["y"].astype(np.intp), ev["x"].astype(np.intp)]
    shifted = ev["t"].astype(np.int64) + np.rint(per_event).astype(np.int64)
    shifted = np.clip(shifted, 1, np.iinfo(np.uint32).max)

    perturbed = make_events(ev["x"], ev["y"], shifted.astype(np.uint32), ev["p"])
    out = Scene(**{**scene.__dict__, "events": perturbed})
    out.extra = {**scene.extra, "sigma_contrast": sigma_contrast}
    return out, offsets_us


ALL_SCENES: dict[str, Callable[..., Scene]] = {
    "bar": translating_bar,
    "dot": moving_dot,
    "corner": translating_corner,
}
