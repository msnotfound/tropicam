"""The time surface: T(x, y) <- max(T(x, y), t).

That update rule is tropical (max-plus) addition. The whole project follows
from taking it literally, so this module is deliberately small and exact.

Design decisions locked in by the master doc (section 7):

* **Pre-allocated flat contiguous array**, shape (polarities, H, W), row-major.
  Neighbourhood access is what buys locality; fitting in L2 is not the goal.

* **uint32 timestamps with wrapping unsigned subtraction** for age. `now - T`
  is correct mod 2**32 as long as the true age is below ~71.6 minutes at
  microsecond resolution.

* **No rolling epoch.** Rebasing timestamps to keep them small forces a
  periodic O(N) stop-the-world sweep over the entire surface, and that pause
  lands squarely in the p99 latency plot we care about. Wrapping subtraction
  gives the same memory saving with no sweep and no spike.

* **Scatter-max, not last-write-wins.** See `scatter_max` below; this is the
  correctness footgun the doc calls out by name.
"""

from __future__ import annotations

import numpy as np

#: Timestamps wrap at this value; ages above it are ambiguous.
WRAP_US = 1 << 32

#: ~71.58 minutes. Ages beyond this cannot be represented unambiguously.
MAX_UNAMBIGUOUS_AGE_US = WRAP_US - 1


def scatter_max(surface_flat: np.ndarray, flat_idx: np.ndarray, ts: np.ndarray,
                method: str = "sort") -> None:
    """In-place ``surface_flat[i] = max(surface_flat[i], t)`` over duplicates.

    The obvious spelling is wrong::

        T[ys, xs] = np.maximum(T[ys, xs], ts)   # WRONG

    Fancy-index assignment is **last-write-wins** when `flat_idx` repeats
    within a batch, so a later-arriving *older* timestamp silently overwrites a
    newer one. It is not a max. A clean translating bar fires each pixel
    exactly once, so the synthetic oracle will NOT catch this -- hence the
    explicit duplicate-pixel test in the suite.

    Two correct implementations, kept side by side because the choice matters
    once the hot loop gets ported:

    ``"ufunc"``
        ``np.maximum.at`` -- unbuffered, unambiguously correct, slow.
    ``"sort"``
        Sort by (index, timestamp), keep the last row per index, then assign.
        Indices are unique after the reduction, so plain fancy assignment is
        safe. Usually several times faster on realistic batch sizes.
    """
    if flat_idx.size == 0:
        return

    if method == "ufunc":
        np.maximum.at(surface_flat, flat_idx, ts)
        return

    if method != "sort":
        raise ValueError(f"unknown scatter_max method: {method!r}")

    # lexsort orders by the LAST key first, so this is (idx asc, then t asc).
    order = np.lexsort((ts, flat_idx))
    idx_sorted = flat_idx[order]
    ts_sorted = ts[order]

    # Keep the final row of each run of equal indices == the max timestamp.
    keep = np.empty(idx_sorted.shape, dtype=bool)
    keep[-1] = True
    np.not_equal(idx_sorted[1:], idx_sorted[:-1], out=keep[:-1])

    uniq_idx = idx_sorted[keep]
    uniq_ts = ts_sorted[keep]

    # Indices are unique now, so this is a genuine element-wise max.
    surface_flat[uniq_idx] = np.maximum(surface_flat[uniq_idx], uniq_ts)


class TimeSurface:
    """Per-pixel most-recent-event timestamps, one plane per polarity.

    Polarity planes are kept separate by default (doc section 7.3): merging
    them discards the sign of the brightness change, and the decision changes
    the memory footprint enough to matter (~7.4 MB at 1280x720, 2 planes,
    32-bit).
    """

    def __init__(self, height: int, width: int, n_polarities: int = 2,
                 scatter_method: str = "sort"):
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        self.height = int(height)
        self.width = int(width)
        self.n_polarities = int(n_polarities)
        self.scatter_method = scatter_method

        # Contiguous, pre-allocated, never reallocated.
        self._t = np.zeros((self.n_polarities, self.height, self.width),
                           dtype=np.uint32)
        self._flat = self._t.reshape(-1)
        self._plane_stride = self.height * self.width

        #: Timestamp of the most recent event consumed, for age queries.
        self.now = np.uint32(0)

    # -- ingest ---------------------------------------------------------

    def update(self, events: np.ndarray) -> None:
        """Consume a batch of events. Order within the batch does not matter."""
        if events.size == 0:
            return
        x = events["x"].astype(np.intp)
        y = events["y"].astype(np.intp)
        p = events["p"].astype(np.intp)
        t = events["t"]

        flat_idx = (p * self._plane_stride) + (y * self.width) + x
        scatter_max(self._flat, flat_idx, t, method=self.scatter_method)

        batch_max = t.max()
        if batch_max > self.now:
            self.now = np.uint32(batch_max)

    def reset(self) -> None:
        self._t.fill(0)
        self.now = np.uint32(0)

    # -- read -----------------------------------------------------------

    @property
    def timestamps(self) -> np.ndarray:
        """(polarities, H, W) view of raw timestamps. Do not mutate."""
        return self._t

    def plane(self, polarity: int) -> np.ndarray:
        """(H, W) view of one polarity plane."""
        return self._t[polarity]

    def merged(self) -> np.ndarray:
        """(H, W) polarity-agnostic surface -- the max across planes.

        Tropical addition across planes, which is the only merge that respects
        the semiring. Averaging the planes would be exactly the clock-time
        mistake.
        """
        return self._t.max(axis=0)

    def age(self, now: int | None = None) -> np.ndarray:
        """Age of each pixel in microseconds, via wrapping unsigned subtraction.

        Valid only where the true age is below `MAX_UNAMBIGUOUS_AGE_US`; older
        pixels alias. Quiet regions therefore report garbage ages, which is
        acceptable because they carry no motion evidence -- but do not treat a
        large age as meaningful without checking `has_fired`.
        """
        ref = self.now if now is None else np.uint32(now)
        return ref - self._t  # uint32 arithmetic wraps; this is intentional.

    def has_fired(self) -> np.ndarray:
        """Boolean mask of pixels that have received at least one event.

        Timestamp 0 is the sentinel for "never fired". A genuine event at
        t == 0 is indistinguishable from it; the synthetic oracle therefore
        starts streams at t > 0.
        """
        return self._t != 0

    # -- introspection --------------------------------------------------

    @property
    def nbytes(self) -> int:
        return self._t.nbytes

    def __repr__(self) -> str:
        return (f"TimeSurface({self.height}x{self.width}, "
                f"{self.n_polarities} planes, {self.nbytes / 1e6:.1f} MB, "
                f"now={int(self.now)}us)")
