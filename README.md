# TropiCam

A deterministic, CPU-only, training-free event-vision engine whose native
arithmetic is the tropical (max-plus) semiring.

The canonical data structure in event vision is the **time surface**, built by
one rule:

```
T(x, y) <- max( T(x, y), t )
```

That `max` is the *addition operation of the tropical semiring*. The field
builds the surface with tropical arithmetic and then analyses it with ordinary
linear algebra. TropiCam uses the matching algebra throughout.

The project's headline contribution is not the engine. It is a **provable
invariance certificate** against per-pixel contrast-threshold mismatch — the
fixed-pattern sensor defect (sigma ~2.5-4% contrast) that every event camera
permanently has and that no existing method certifies against.

Full project document: [`docs/tropicam-master-doc.md`](docs/tropicam-master-doc.md).
Build status and next steps: [`ROADMAP.md`](ROADMAP.md).

## Status

Week 1 scaffold. The synthetic oracle and the time surface are built and
tested; no flow estimator yet.

## Quick start

Requires Python 3.10+ and numpy. Nothing else.

```bash
python3 tests/test_time_surface.py     # 8 tests
python3 tests/test_synthetic.py        # 7 tests
python3 scripts/demo_oracle.py         # renders time surfaces to out/*.ppm
python3 scripts/bench_time_surface.py  # ingest throughput
```

`pytest tests/` also works if you have it; the suites are written to run both
ways so nothing is gated on an install.

## Layout

```
src/tropicam/
  events.py        event stream dtype and helpers
  time_surface.py  T <- max(T, t), the max-plus accumulator
  synthetic.py     analytic-ground-truth scenes + perturbations
  render.py        dependency-free PPM visualisation
tests/             correctness, including the named footguns
scripts/           demo and benchmark entry points
docs/              the master project document
```

## The three oracle scenes

Built before any real data, per the master doc's build order. Each has a
closed-form time surface the engine must reproduce.

| Scene | Constrains | Regimes | Purpose |
|---|---|---|---|
| `translating_bar` | **normal flow only** | 1 | affine ramp, slope `1/vx` |
| `moving_dot` | full flow | 1 | curved boundary pins both components |
| `translating_corner` | full flow | 2 | `max(affine, affine)` — a known crease |

The bar/dot split is deliberate. A vertical edge carries no information about
its own vertical motion, so a "wrong" `v_y` from the bar scene may be correct
behaviour rather than a bug — the aperture problem. `test_bar_constrains_normal_flow_only`
asserts this explicitly so it can't be rediscovered as a mystery later.

The corner scene is the ground truth for motion-regime decomposition: its
surface is literally a max of two affine functions, i.e. a tropical polynomial
with one analytically known crease.

## Design decisions already locked in

- **uint32 timestamps + wrapping unsigned subtraction** for age. Correct up to
  ~71.6 minutes at microsecond resolution.
- **No rolling epoch.** Rebasing timestamps forces a periodic O(N)
  stop-the-world sweep, and that pause lands directly in the p99 latency plot.
- **Scatter-max, not last-write-wins.** `T[ys, xs] = np.maximum(T[ys, xs], ts)`
  is *wrong* — fancy-index assignment resolves duplicate pixels by last write,
  not by max. A clean translating bar fires each pixel once, so the oracle will
  never catch it; `test_duplicate_pixels_within_batch_take_the_max` does.
- **Separate ON/OFF polarity planes.** ~7.4 MB at 1280x720. Merging is a max
  across planes (tropical addition), never an average.
- **NumPy first.** Porting the hot loop to Rust/C++ comes after the pipeline is
  correct and the demo runs, not before.

## Honest scoping

Carried over from the master doc (section 6) and binding on any writeup:

- The affine-ramp observation is **Benosman's**, ~12 years old. Conceded up front.
- A crease means a change in *local velocity*, not a new object — rotation,
  depth gradients, and occlusion edges all create creases with no second
  object. The claim is "locally-affine motion regimes", never "objects".
- Max-plus has no averaging, so its breakdown point is 1/n: one spurious high
  timestamp corrupts a neighbourhood. The corruption is transient (the next
  genuine event overwrites it) and the fix is native (max-plus opening) — but
  opening starts with an erosion, which *injects delay*. That cost gets
  measured and reported, not hidden.
- Local sliding windows only. A real time surface is not globally convex, so
  full-frame tropical regression is the wrong shape for the problem.
- **No hardware means no end-to-end latency claim.** Everything here is
  *processing* throughput on replayed streams, never sensor-to-decision time.

## Licence

Open source, permissive. Publication is the sole IP strategy — the engine's
whole rationale is that it is open and inspectable.
