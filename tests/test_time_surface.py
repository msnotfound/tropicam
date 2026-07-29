"""Time surface correctness, including the two named footguns.

Runs under pytest, or standalone: ``python3 tests/test_time_surface.py``
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tropicam.events import ON, OFF, make_events  # noqa: E402
from tropicam.time_surface import TimeSurface, scatter_max  # noqa: E402


def test_basic_max_update():
    ts = TimeSurface(8, 8)
    ts.update(make_events([1, 1], [2, 2], [500, 900], [ON, ON]))
    assert ts.plane(1)[2, 1] == 900

    # An older event must NOT overwrite a newer one.
    ts.update(make_events([1], [2], [300], [ON]))
    assert ts.plane(1)[2, 1] == 900, "max was not respected across batches"


def test_duplicate_pixels_within_batch_take_the_max():
    """The footgun: fancy-index assignment is last-write-wins, not a max.

    The doc calls this out by name -- and warns that a clean translating bar
    fires each pixel exactly once, so the synthetic oracle will never catch
    it. Hence this explicit test with repeated pixels in one batch, ordered so
    that the OLDEST timestamp arrives last. A last-write-wins implementation
    leaves 100 here; a real max leaves 900.
    """
    ts = TimeSurface(8, 8)
    ts.update(make_events([3, 3, 3], [4, 4, 4], [500, 900, 100],
                          [ON, ON, ON]))
    assert ts.plane(1)[4, 3] == 900

    # And demonstrate that the naive spelling really would have been wrong,
    # so this test documents the bug rather than merely guarding against it.
    naive = np.zeros((8, 8), dtype=np.uint32)
    ys = np.array([4, 4, 4]); xs = np.array([3, 3, 3])
    vals = np.array([500, 900, 100], dtype=np.uint32)
    naive[ys, xs] = np.maximum(naive[ys, xs], vals)
    assert naive[4, 3] == 100, "expected the naive version to be wrong"


def test_scatter_max_methods_agree():
    rng = np.random.default_rng(0)
    n, size = 20_000, 512
    idx = rng.integers(0, size, n).astype(np.intp)
    vals = rng.integers(1, 10_000, n).astype(np.uint32)

    a = np.zeros(size, dtype=np.uint32)
    b = np.zeros(size, dtype=np.uint32)
    scatter_max(a, idx, vals, method="sort")
    scatter_max(b, idx, vals, method="ufunc")
    assert np.array_equal(a, b)

    # Ground truth by brute force.
    ref = np.zeros(size, dtype=np.uint32)
    for i, v in zip(idx, vals):
        ref[i] = max(ref[i], v)
    assert np.array_equal(a, ref)


def test_batch_order_does_not_matter():
    rng = np.random.default_rng(7)
    ev = make_events(rng.integers(0, 32, 5000), rng.integers(0, 32, 5000),
                     rng.integers(1, 50_000, 5000), rng.integers(0, 2, 5000))

    whole = TimeSurface(32, 32)
    whole.update(ev)

    chunked = TimeSurface(32, 32)
    for i in range(0, ev.size, 137):
        chunked.update(ev[i:i + 137])

    assert np.array_equal(whole.timestamps, chunked.timestamps)

    shuffled = TimeSurface(32, 32)
    shuffled.update(ev[rng.permutation(ev.size)])
    assert np.array_equal(whole.timestamps, shuffled.timestamps)


def test_polarity_planes_are_independent():
    ts = TimeSurface(8, 8)
    ts.update(make_events([2, 2], [3, 3], [400, 800], [ON, OFF]))
    assert ts.plane(1)[3, 2] == 400
    assert ts.plane(0)[3, 2] == 800
    # merged() is a max across planes -- tropical addition, not an average.
    assert ts.merged()[3, 2] == 800


def test_wrapping_age_no_rolling_epoch():
    """Age via wrapping unsigned subtraction stays correct across the wrap."""
    ts = TimeSurface(4, 4)
    near_wrap = np.uint32(2**32 - 100)
    ts.update(make_events([0], [0], [near_wrap], [ON]))

    # 'now' has wrapped past zero; true age is 350us.
    now = np.uint32(250)
    assert int(ts.age(now)[1, 0, 0]) == 350

    # Same arithmetic away from the boundary.
    ts2 = TimeSurface(4, 4)
    ts2.update(make_events([0], [0], [1000], [ON]))
    assert int(ts2.age(1350)[1, 0, 0]) == 350


def test_has_fired_sentinel():
    ts = TimeSurface(4, 4)
    ts.update(make_events([1], [1], [55], [ON]))
    fired = ts.has_fired()
    assert fired[1, 1, 1]
    assert not fired[1, 0, 0]
    assert fired.sum() == 1


def test_reset_and_footprint():
    ts = TimeSurface(720, 1280)
    # Doc section 7.3 budget: ~7.4 MB at 1280x720, two 32-bit polarity planes.
    assert 7.0e6 < ts.nbytes < 7.5e6
    ts.update(make_events([5], [5], [10], [ON]))
    ts.reset()
    assert ts.timestamps.max() == 0 and int(ts.now) == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
