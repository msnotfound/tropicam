"""Event stream representation.

An event stream is a flat, contiguous, time-sorted structured array. This is
the only input format the engine accepts; whether the events came from a
sensor, a dataset file, or the synthetic oracle is irrelevant downstream.

Timestamps are microseconds since stream start, stored as uint32. See
`time_surface` for why uint32 (and why there is no rolling epoch).
"""

from __future__ import annotations

import numpy as np

#: One event: pixel coordinates, timestamp (us), polarity (0 = OFF, 1 = ON).
EVENT_DTYPE = np.dtype(
    [
        ("x", np.uint16),
        ("y", np.uint16),
        ("t", np.uint32),
        ("p", np.uint8),
    ]
)

OFF = np.uint8(0)
ON = np.uint8(1)


def make_events(x, y, t, p) -> np.ndarray:
    """Pack parallel arrays into a time-sorted event array.

    Sorting is stable so that events sharing a timestamp keep their relative
    order, which keeps the synthetic oracle reproducible.
    """
    x = np.asarray(x, dtype=np.uint16)
    y = np.asarray(y, dtype=np.uint16)
    t = np.asarray(t, dtype=np.uint32)
    p = np.asarray(p, dtype=np.uint8)
    if not (len(x) == len(y) == len(t) == len(p)):
        raise ValueError("x, y, t, p must have equal length")

    ev = np.empty(len(x), dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["t"], ev["p"] = x, y, t, p
    ev.sort(order="t", kind="stable")
    return ev


def is_sorted(events: np.ndarray) -> bool:
    """True if timestamps are non-decreasing."""
    return bool(np.all(np.diff(events["t"].astype(np.int64)) >= 0))


def duration_us(events: np.ndarray) -> int:
    """Span of the stream in microseconds."""
    if events.size == 0:
        return 0
    return int(events["t"][-1]) - int(events["t"][0])


def event_rate(events: np.ndarray) -> float:
    """Mean events per second over the stream's span."""
    span = duration_us(events)
    if span == 0:
        return float("nan")
    return events.size / (span * 1e-6)
