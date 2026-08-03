"""Metrics for E1, E2 and E3.

Deliberately estimator-agnostic: these take arrays, not engine objects, so the
same measurement layer serves whichever tropical regression formulation wins.

Two of these encode corrections to the master doc rather than implementing it
literally. Both are flagged in their docstrings and in HAZARDS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -- flow error (E1) ----------------------------------------------------


def angular_error(est: np.ndarray, gt: np.ndarray, *,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel angular error in degrees, the standard event-flow metric.

    `est` and `gt` are (..., 2) arrays of (vx, vy). Uses the usual 3D
    embedding (u, v, 1) so that zero-motion pixels stay well defined.
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if est.shape != gt.shape or est.shape[-1] != 2:
        raise ValueError("est and gt must share shape (..., 2)")

    e = np.concatenate([est, np.ones(est.shape[:-1] + (1,))], axis=-1)
    g = np.concatenate([gt, np.ones(gt.shape[:-1] + (1,))], axis=-1)
    e /= np.linalg.norm(e, axis=-1, keepdims=True)
    g /= np.linalg.norm(g, axis=-1, keepdims=True)

    ae = np.degrees(np.arccos(np.clip((e * g).sum(-1), -1.0, 1.0)))
    return ae if mask is None else ae[mask]


def endpoint_error(est: np.ndarray, gt: np.ndarray, *,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel L2 endpoint error."""
    err = np.linalg.norm(np.asarray(est, float) - np.asarray(gt, float),
                         axis=-1)
    return err if mask is None else err[mask]


def normal_flow_error(est: np.ndarray, gt: np.ndarray, normal: tuple,
                      *, mask: np.ndarray | None = None) -> np.ndarray:
    """Error in the edge-normal component only.

    The metric to use on the `translating_bar` scene. Scoring full flow there
    is a category error: a vertical edge carries no evidence about vertical
    motion, so a "wrong" v_y is unconstrained rather than incorrect. Every
    Scene declares `constrains`; honour it.
    """
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    err = np.abs((np.asarray(est, float) - np.asarray(gt, float)) @ n)
    return err if mask is None else err[mask]


# -- motion regimes (E2) ------------------------------------------------


@dataclass
class RegimeAgreement:
    """Refinement-based scoring of a regime map against object labels."""

    purity: float           # fraction of regime pixels whose regime is object-pure
    n_regimes: int
    n_objects: int
    count_ratio: float      # descriptive ONLY -- see docstring
    impure_regimes: int

    def __repr__(self) -> str:
        return (f"RegimeAgreement(purity={self.purity:.3f}, "
                f"{self.n_regimes} regimes / {self.n_objects} objects, "
                f"ratio={self.count_ratio:.2f}, "
                f"impure={self.impure_regimes})")


def regime_agreement(regimes: np.ndarray, objects: np.ndarray, *,
                     ignore: int = -1,
                     purity_threshold: float = 0.9) -> RegimeAgreement:
    """Score a regime decomposition as a *refinement* of the object map.

    **This deliberately does not implement E2 as the master doc words it.**
    The doc says to compare "recovered piece count vs. true object count", but
    its own section 6.2 establishes that pieces are not objects: rotation,
    depth gradients and occlusion boundaries all produce creases with no
    second object present. Comparing those counts scores two incomparable
    quantities, and a reviewer will quote 6.2 back at us.

    The defensible claim is a **refinement** one: every recovered regime should
    lie within a single object (a regime must not straddle an object
    boundary), while one object may legitimately decompose into several
    regimes. That is exactly `purity`, and it is the number to lead with.

    `count_ratio` is retained as *descriptive* output -- useful for spotting
    runaway over-segmentation -- but it is not an accuracy score and must
    never be reported as one.
    """
    regimes = np.asarray(regimes)
    objects = np.asarray(objects)
    if regimes.shape != objects.shape:
        raise ValueError("regimes and objects must share shape")

    valid = (regimes != ignore) & (objects != ignore)
    if not valid.any():
        return RegimeAgreement(float("nan"), 0, 0, float("nan"), 0)

    r_labels = np.unique(regimes[valid])
    o_labels = np.unique(objects[valid])

    pure_pixels = 0
    impure = 0
    for r in r_labels:
        sel = valid & (regimes == r)
        counts = np.bincount(objects[sel].astype(np.int64).ravel())
        dominant = counts.max()
        if dominant / sel.sum() >= purity_threshold:
            pure_pixels += int(sel.sum())
        else:
            impure += 1

    n_r, n_o = len(r_labels), len(o_labels)
    return RegimeAgreement(
        purity=pure_pixels / int(valid.sum()),
        n_regimes=n_r,
        n_objects=n_o,
        count_ratio=n_r / n_o if n_o else float("nan"),
        impure_regimes=impure,
    )


# -- certificate (E3) ---------------------------------------------------


@dataclass
class CertificateCheck:
    """Result of testing a derived bound against measured deviation."""

    bound: float
    max_deviation: float
    p99_deviation: float
    n_violations: int
    n_samples: int
    slack: float            # bound - max_deviation; negative means violated

    @property
    def holds(self) -> bool:
        return self.n_violations == 0

    @property
    def tightness(self) -> float:
        """max_deviation / bound. Near 1 is tight; near 0 is vacuous."""
        return self.max_deviation / self.bound if self.bound > 0 else float("nan")

    def __repr__(self) -> str:
        verdict = "HOLDS" if self.holds else f"VIOLATED x{self.n_violations}"
        return (f"CertificateCheck({verdict}, bound={self.bound:.4g}, "
                f"max_dev={self.max_deviation:.4g}, "
                f"tightness={self.tightness:.3f})")


def check_certificate(deviation: np.ndarray, bound: float, *,
                      tol: float = 0.0) -> CertificateCheck:
    """Test measured output deviation against a derived bound.

    A certificate that merely *holds* proves nothing on its own -- a bound of
    infinity holds too. Always report `tightness` alongside: a bound the data
    never approaches is vacuous, and a reviewer will say so. The interesting
    result is a bound that holds AND is approached.
    """
    dev = np.abs(np.asarray(deviation, float).ravel())
    dev = dev[np.isfinite(dev)]
    if dev.size == 0:
        raise ValueError("no finite deviation samples")

    violations = int((dev > bound + tol).sum())
    return CertificateCheck(
        bound=float(bound),
        max_deviation=float(dev.max()),
        p99_deviation=float(np.percentile(dev, 99)),
        n_violations=violations,
        n_samples=int(dev.size),
        slack=float(bound - dev.max()),
    )


def contrast_rate_floor(surface: np.ndarray, valid: np.ndarray,
                        dt_us: float = 1.0) -> float:
    """Estimate the minimum local temporal contrast rate over a surface.

    The mismatch bound scales as ``sigma_C / r``, so it **diverges as r -> 0**:
    in slow or low-contrast regions the certificate is vacuous. Any bound
    reported without the r-floor it was computed under is not a certificate,
    it is a hope. This returns the conditioning quantity so experiments can
    state it explicitly.

    Crude proxy: the inverse of the largest local timestamp gap, i.e. the
    slowest observed firing. Replace with a proper photometric estimate once
    real data is in.
    """
    s = np.where(valid, surface.astype(float), np.nan)
    gy, gx = np.gradient(s)
    grad = np.sqrt(np.nan_to_num(gy) ** 2 + np.nan_to_num(gx) ** 2)
    finite = grad[np.isfinite(grad) & (grad > 0)]
    if finite.size == 0:
        return float("nan")
    return float(dt_us / finite.max())
