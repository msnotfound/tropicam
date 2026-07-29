"""The Rust core must match the NumPy reference bit for bit.

The NumPy implementation is the *executable specification*. It is slower and
it cannot make the O(1)-per-event claim, but it is simple enough to audit by
eye, and it is validated against the analytic oracle. Differential testing
against it is what lets the fast path be trusted.

Where the two disagree, one is a bug -- that is the point of keeping both.

Skips cleanly (exit 0) if the extension is not built; run
``scripts/build_rust.sh`` first.

Runs under pytest, or standalone: ``python3 tests/test_rust_parity.py``
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tropicam.events import EVENT_DTYPE, make_events  # noqa: E402
from tropicam.synthetic import (  # noqa: E402
    translating_bar, moving_dot, translating_corner, inject_background_noise,
)
from tropicam.time_surface import TimeSurface as NumpyTS  # noqa: E402

try:
    import tropicam_rs
except ImportError:  # pragma: no cover
    tropicam_rs = None

SKIP_MSG = "tropicam_rs not built -- run scripts/build_rust.sh"


def _rust_surface(h, w, events, planes=2):
    ts = tropicam_rs.TimeSurface(h, w, planes)
    ts.update_bytes(events.view(np.uint8))
    raw = np.frombuffer(ts.timestamps_bytes(), dtype=np.uint32)
    return ts, raw.reshape(planes, h, w)


def test_event_layout_agrees_across_the_boundary():
    """A 9-byte packed dtype on both sides, asserted, not assumed."""
    assert EVENT_DTYPE.itemsize == tropicam_rs.event_size() == 9
    offsets = [EVENT_DTYPE.fields[f][1] for f in EVENT_DTYPE.names]
    assert offsets == [0, 2, 4, 8]


def test_parity_on_every_oracle_scene():
    for scene in (translating_bar(), moving_dot(), translating_corner()):
        ref = NumpyTS(scene.height, scene.width)
        ref.update(scene.events)
        _, got = _rust_surface(scene.height, scene.width, scene.events)
        assert np.array_equal(ref.timestamps, got), f"mismatch on {scene.name}"


def test_parity_under_noise():
    scene = inject_background_noise(translating_bar(), rate_hz_per_px=5.0,
                                    rng=np.random.default_rng(11))
    ref = NumpyTS(scene.height, scene.width)
    ref.update(scene.events)
    _, got = _rust_surface(scene.height, scene.width, scene.events)
    assert np.array_equal(ref.timestamps, got)


def test_parity_on_adversarial_duplicates():
    """Heavy duplicate pressure -- the case that breaks last-write-wins.

    Only 64 pixels for 200k events, so nearly every event collides. This is
    the scatter-max footgun at maximum intensity.
    """
    rng = np.random.default_rng(5)
    n = 200_000
    ev = make_events(rng.integers(0, 8, n), rng.integers(0, 8, n),
                     rng.integers(1, 5_000, n), rng.integers(0, 2, n))
    ref = NumpyTS(8, 8)
    ref.update(ev)
    _, got = _rust_surface(8, 8, ev)
    assert np.array_equal(ref.timestamps, got)


def test_parity_across_batching_and_shuffling():
    """Batch boundaries and arrival order must not change the result."""
    rng = np.random.default_rng(2)
    n = 50_000
    ev = make_events(rng.integers(0, 64, n), rng.integers(0, 48, n),
                     rng.integers(1, 100_000, n), rng.integers(0, 2, n))

    ref = NumpyTS(48, 64)
    ref.update(ev)

    chunked = tropicam_rs.TimeSurface(48, 64, 2)
    for i in range(0, ev.size, 999):
        chunked.update_bytes(ev[i:i + 999].view(np.uint8))
    got = np.frombuffer(chunked.timestamps_bytes(),
                        dtype=np.uint32).reshape(2, 48, 64)
    assert np.array_equal(ref.timestamps, got)

    _, shuffled = _rust_surface(48, 64, ev[rng.permutation(ev.size)])
    assert np.array_equal(ref.timestamps, shuffled)


def test_parity_of_merged_and_age():
    scene = translating_corner()
    ref = NumpyTS(scene.height, scene.width)
    ref.update(scene.events)
    ts, _ = _rust_surface(scene.height, scene.width, scene.events)

    merged = np.frombuffer(ts.merged_bytes(), dtype=np.uint32).reshape(
        scene.height, scene.width)
    assert np.array_equal(ref.merged(), merged)

    assert ts.now == int(ref.now)
    now = int(ref.now)
    age = np.frombuffer(ts.age_bytes(now), dtype=np.uint32).reshape(
        2, scene.height, scene.width)
    assert np.array_equal(ref.age(now), age)


def test_wrapping_age_parity_across_the_boundary():
    ev = make_events([0], [0], [np.uint32(2**32 - 100)], [1])
    ref = NumpyTS(4, 4)
    ref.update(ev)
    ts, _ = _rust_surface(4, 4, ev)
    age = np.frombuffer(ts.age_bytes(250), dtype=np.uint32).reshape(2, 4, 4)
    assert age[1, 0, 0] == 350
    assert np.array_equal(ref.age(250), age)


def test_out_of_bounds_events_are_dropped_not_wrapped():
    ev = make_events([99, 0], [0, 99], [500, 500], [1, 1])
    ts = tropicam_rs.TimeSurface(8, 8, 2)
    consumed, updated, oob = ts.update_bytes(ev.view(np.uint8))
    assert (consumed, updated, oob) == (2, 0, 2)


def test_ragged_buffer_raises():
    ts = tropicam_rs.TimeSurface(8, 8, 2)
    try:
        ts.update_bytes(b"\x00" * 10)
    except Exception as e:
        assert "9-byte events" in str(e)
    else:
        raise AssertionError("expected a buffer error")


if __name__ == "__main__":
    if tropicam_rs is None:
        print(f"SKIP: {SKIP_MSG}")
        raise SystemExit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
