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

Week 1-2. Synthetic oracle, time surface, and the native Rust core with Python
bindings are built and tested. No flow estimator yet.

**Ingest: 157 M events/s, 6.4 ns/event** at 1280x720 (Rust core, 11x the best
NumPy path). Processing throughput on replayed streams -- see the note on
latency claims below.

## Quick start

Python 3.11+ and numpy. The pure-Python path needs nothing else.

```bash
python3 tests/test_time_surface.py     # 8 tests  (numpy reference)
python3 tests/test_synthetic.py        # 7 tests  (oracle)
python3 scripts/demo_oracle.py         # renders time surfaces to out/*.ppm
```

For the native core, a Rust toolchain (1.70+):

```bash
./scripts/build_rust.sh                # cargo test + build + install the .so
python3 tests/test_rust_parity.py      # 9 tests  (rust == numpy, bit for bit)
python3 scripts/bench_time_surface.py  # numpy vs rust throughput
```

`pytest tests/` also works if you have it; suites run both ways, and the
parity suite skips cleanly when the extension isn't built, so nothing is gated
on an install.

## Layout

```
crates/tropicam-core/   native engine -- no Python dependency
  events.rs             Event, ABI-compatible with the NumPy dtype
  time_surface.rs       T <- max(T, t), O(1) per event
crates/tropicam-py/     PyO3 bindings, marshalling only
src/tropicam/           NumPy reference implementation
  events.py             event stream dtype and helpers
  time_surface.py       the max-plus accumulator
  synthetic.py          analytic-ground-truth scenes + perturbations
  render.py             dependency-free PPM visualisation
tests/                  correctness, footguns, and rust/numpy parity
scripts/                build, demo, and benchmark entry points
docs/                   the master project document
```

## Two implementations, on purpose

The NumPy package is **not** dead prototype code. It is the *executable
specification*: small enough to audit by eye, validated against the analytic
oracle, and differentially tested against the Rust core on every oracle scene
plus adversarial duplicate-pressure streams (`tests/test_rust_parity.py`).
Where the two disagree, one of them is a bug.

Events cross the FFI boundary **zero-copy**. NumPy's packed structured dtype
is 9 bytes at offsets 0/2/4/8; the Rust `Event` is `#[repr(C, packed)]` to
match exactly, and both sides assert the layout so drift fails loudly instead
of silently misreading every field.

## Why the native core is not just "faster"

It changes what can be claimed, which is the point.

A correct scatter-max in NumPy needs `np.maximum.at` (unbuffered, slow) or a
lexsort-and-reduce -- **O(n log n)** in the batch, because vectorised
fancy-index assignment resolves duplicate pixels by last write rather than by
max. In Rust the same operation is a load, a compare, and a conditional
store: no sort, no allocation, no batching required.

That is **strictly constant work per event**, independent of batch size,
resolution, and stream history. The NumPy path cannot make that claim at all;
it was always standing in for this.

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
- **NumPy first, then Rust.** The native port landed only after the pipeline was
  correct and the demo ran -- so the reference implementation could become the
  parity oracle instead of being thrown away. Rust, not Go: GC pauses would
  land in exactly the p99 plot the engine is judged on.

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
