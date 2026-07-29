"""Measure the time-surface ingest ceiling (week 1: 'throughput ceiling measured').

Reports **processing** throughput on a replayed stream. Without hardware we
cannot and do not claim sensor-to-decision latency (doc section 6.9) -- every
number here is time spent inside the engine on events already in memory.

Usage:
    python3 scripts/bench_time_surface.py [--events 2000000] [--batch 8192]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tropicam.events import make_events  # noqa: E402
from tropicam.time_surface import TimeSurface  # noqa: E402


def synth_stream(n: int, h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return make_events(
        rng.integers(0, w, n), rng.integers(0, h, n),
        np.sort(rng.integers(1, 10_000_000, n)), rng.integers(0, 2, n),
    )


def bench(events, h, w, batch, method, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        ts = TimeSurface(h, w, scatter_method=method)
        t0 = time.perf_counter()
        for i in range(0, events.size, batch):
            ts.update(events[i:i + batch])
        best = min(best, time.perf_counter() - t0)
    return events.size / best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=2_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    ev = synth_stream(args.events, args.height, args.width)
    surf = TimeSurface(args.height, args.width)

    print(f"resolution   {args.width}x{args.height}   surface {surf.nbytes/1e6:.1f} MB")
    print(f"stream       {ev.size:,} events, batch {args.batch}")
    print("(processing throughput only -- not sensor-to-decision latency)\n")

    results = {}
    for method in ("sort", "ufunc"):
        rate = bench(ev, args.height, args.width, args.batch, method)
        results[method] = rate
        print(f"  numpy/{method:6s}  {rate/1e6:7.2f} M events/s   "
              f"{1e9/rate:6.1f} ns/event")

    try:
        import tropicam_rs
    except ImportError:
        print("\n  (rust core not built -- run scripts/build_rust.sh)")
    else:
        raw = ev.view(np.uint8)
        stride = args.batch * ev.dtype.itemsize
        best = float("inf")
        for _ in range(3):
            ts = tropicam_rs.TimeSurface(args.height, args.width, 2)
            t0 = time.perf_counter()
            for i in range(0, raw.size, stride):
                ts.update_bytes(raw[i:i + stride])
            best = min(best, time.perf_counter() - t0)
        results["rust"] = ev.size / best
        print(f"  rust          {results['rust']/1e6:7.2f} M events/s   "
              f"{1e9/results['rust']:6.1f} ns/event")
        print(f"\n  rust is {results['rust']/results['sort']:.1f}x the best "
              f"numpy path")

    print("\n  NumPy needs an O(n log n) sort per batch to make scatter-max\n"
          "  correct; the Rust core is a compare-and-store, i.e. strictly\n"
          "  constant work per event.")

    # Per-batch latency distribution -- the p99 plot the doc cares about. Note
    # there is no periodic spike here by construction: no rolling epoch means
    # no O(N) stop-the-world sweep.
    ts = TimeSurface(args.height, args.width)
    lat = []
    for i in range(0, ev.size, args.batch):
        t0 = time.perf_counter()
        ts.update(ev[i:i + args.batch])
        lat.append((time.perf_counter() - t0) * 1e6)
    lat = np.array(lat)
    print(f"\nper-batch latency (us):  p50 {np.percentile(lat,50):7.1f}   "
          f"p99 {np.percentile(lat,99):7.1f}   max {lat.max():7.1f}")
    print(f"max/p50 ratio {lat.max()/np.percentile(lat,50):.1f}x "
          f"(no epoch-rebase sweep by design)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
