# Roadmap

Derived from the master doc, section 9. Ten-week window.

**Hard rule (doc section 10): no theorem section gets written before E2 and E3
have numbers.**

---

## Week 0 — confirm the gap is empty

- [ ] Exact-phrase search: *"certified event-based vision"*, *"threshold
      mismatch invariance"*, *"event camera certified robustness"*. ~1 hour.
      This must happen **now**, not in October — if the certificate is already
      published, the opening narrows rather than dies, but only if you know
      early.
- [ ] Read **Nagata, Sekikawa & Aoki 2021**, *Optical Flow Estimation by
      Matching Time Surface*, in full. It operates directly on the time surface
      with an L1 loss, so it kills any "first non-L2 treatment" claim. Draft the
      paragraph distinguishing closed-form-slope + certificate from
      L1-optimised matching.
- [ ] Check whether `tonic` covers DSEC before building around it.

## Week 1 — oracle and engine floor

- [x] Synthetic oracle: bar (normal flow), dot (full flow), corner (two
      regimes + known crease).
- [x] NumPy time surface: uint32, wrapping age, no rolling epoch, scatter-max.
- [x] Duplicate-pixel test — the footgun the oracle cannot catch.
- [x] Throughput ceiling measured (NumPy: ~14 M events/s ingest, 1280x720).
- [x] Visualiser (dependency-free PPM).

## Native core (brought forward from weeks 7-8)

The submitted project scope commits to a **C++/Rust engine with Python
bindings** as the deliverable, and to **strictly O(1) work per event**. Both
are properties of the native core, so it moved ahead of schedule -- but only
*after* the NumPy pipeline was correct and demoable, which is what the doc's
"native-first trap" warning actually guards against.

- [x] `crates/tropicam-core`: `Event` (repr(C, packed), ABI-matched to the
      NumPy dtype) and `TimeSurface` with compare-and-store ingest -- constant
      work per event, no allocation, no sort.
- [x] `crates/tropicam-py`: PyO3 bindings, zero-copy events via the buffer
      protocol. Built with plain cargo; maturin not required.
- [x] Differential parity suite: Rust == NumPy bit for bit on every oracle
      scene, under noise, under adversarial duplicate pressure, across batch
      boundaries and shuffling.
- [x] **157 M events/s, 6.4 ns/event** (11x the best NumPy path).
- [ ] Max-plus morphology in the core (erosion/dilation/opening) once the
      NumPy versions exist to check against.
- [ ] p99 latency histograms from the native path.
- [ ] Consider building on **PALMA** (N'guessan 2026, MIT, dependency-free C
      tropical linear algebra) rather than reimplementing. The engineering
      novelty of "tropical algebra runs fast on a CPU" is already gone; all
      novelty must live in the vision formulation and the certificate.
- [ ] One real DVS-Gesture sequence rendering end to end. Start on DVS-Gesture,
      **not** DSEC — DSEC is enormous. Convert to a flat binary of packed
      structs and mmap it; don't write parsers, use `tonic`.

## Week 2 — E2 early, gate on C2

- [ ] Tropical regression on a local sliding window: read the slope of the
      max-plus affine fit. Validate against `translating_bar` (normal flow) and
      `moving_dot` (full flow).
- [ ] Crease detection / regime decomposition. Validate against
      `translating_corner`, where the regime map and crease mask are known
      exactly.
- [ ] **E2 on EVIMO**: recovered piece count vs. true object count, including
      rotation and depth-varying scenes — not just fronto-parallel translation.
- [ ] **Decision gate.** If the ratio is wild, C2 collapses. The honest move is
      to report it as a negative result and lean entirely on C1. Not fatal.

## Weeks 3-4 — morphology and E1

- [ ] Max-plus erosion, dilation, opening; vectorised.
- [ ] **E1 noise sweep** on MVSEC/DSEC at 0.1-10 events/px/s: tropical raw vs.
      tropical + opening vs. Benosman plane fit + iterative outlier rejection.
      Both robustified — comparing against a naked least-squares fit is a
      strawman a reviewer will catch.
- [ ] Measure and plot **erosion-induced delay** alongside the error curves.
      Name the cost before a reviewer does.

## Weeks 4-6 — C1 derivation (the paper's core)

- [ ] Formalise: mismatch -> bounded additive offset field on the time axis.
- [ ] Flat structuring element -> non-flat structuring **function** (Maragos).
- [ ] Closed-form bound on output deviation as a function of mismatch magnitude.
- [ ] Two theorems, not one: exact invariance under *spatially uniform*
      monotone transforms (Matheron), and bounded/calibratable degradation
      under *spatially varying* per-pixel mismatch, which Matheron does not
      cover.

## Weeks 6-7 — E3, the certificate validation

- [ ] Per-pixel mismatch at measured hardware sigma (2.5-4%) via **EVIS** or
      **v2e**. The idealised injector in `synthetic.py` is for fast iteration
      only; reported numbers come from a sensor simulator.
- [ ] (a) measured deviation stays within the derived bound.
- [ ] (b) a learned baseline (E-RAFT or similar) has no such guarantee —
      measure its worst-case degradation for contrast.

## Weeks 7-8 — performance

Core ingest already ported (see above). Remaining:

- [ ] Port the flow and regime operators once they exist and their NumPy
      versions can serve as parity oracles.
- [ ] Benchmark harness, latency histograms. All framed as *processing*
      latency.
- [ ] Real-time visualisation interface (committed in the project scope).

## Weeks 9-10 — writeup

- [ ] Paper draft, figures, repo polish, demo video, prof deliverable.

---

## Contributions, in order of weight

| | Contribution | Risk |
|---|---|---|
| **C1** | Closed-form invariance certificate for per-pixel threshold mismatch | Medium — the theorem *is* the project |
| **C2** | Deterministic motion-regime decomposition, provable per-event work bound | **High — hinges on E2** |
| **C3** | The engine + benchmark harness | Low — ships regardless |

Do not invert this ordering. C1 survives even if C2 comes back weak; the
flow-speed framing is scooped and cannot carry a prestige venue.

## Kill criteria

| Risk | Trigger | Response |
|---|---|---|
| C2 collapses | E2 piece count >> object count on rotation/depth | Honest negative result; rest on C1. Not fatal. |
| C1 doesn't close | Bound vacuous or unprovable in time | Fall back to measured invariance + structural argument; downgrade venue. |
| Certificate already published | Week-0 search finds it | Narrow to a sharper delta — but know **now**. |
| Theory-only drift | Theorem written, experiments unrun | Hard rule above. |
| Native-first trap | Starting in Rust week 1 | Forbidden. NumPy first. |

**Floor:** C3 ships regardless — a working, open, deterministic event-vision
stack with a benchmark and a demo, contingent on no theorem.
