# TropiCam — Master Project Document

**Full title (working):** Certified Event Vision: Provable Invariance to Per-Pixel
Threshold Mismatch via Tropical (Max-Plus) Algebra

**Status:** 5th-semester minor project + target paper. Prof expects a
high-prestige venue.
**Date:** July 2026. **Build window:** ~10 weeks.
**Supersedes:** the earlier "TropiCam Master Project Document" (Gemini). See
§12 for the specific corrections — that version contains three errors you
should not build against.

---

## 1. The project in one paragraph

An event camera has no frames. Each pixel independently emits a timestamped
spike the microsecond its log-brightness changes, and stays silent otherwise —
microsecond timing, tiny bandwidth, works in near-darkness and glare. TropiCam
is a deterministic, CPU-only, training-free engine that reads that raw
asynchronous stream and, using **tropical (max-plus) algebra as its native
arithmetic**, recovers optical flow and decomposes locally-affine motion
regimes — with a provable constant work bound per event and no neural network
anywhere in the pipeline. The *paper's* contribution is not the engine: it is
a **provable invariance certificate** against the one perturbation event
cameras permanently suffer from and nobody can certify against.

---

## 2. Background: why the time surface is secretly tropical

The canonical data structure in event vision is the **time surface**: a 2D grid
where each pixel stores the timestamp of its most recent event, maintained by
one rule:

```
T(x, y) ← max( T(x, y), t )
```

Everyone builds this, then runs *ordinary linear algebra* on it — least-squares
plane fits, averaging, clustering, CNNs.

But that `max` update **is the addition operation of the tropical semiring** —
the algebra where `max` plays the role of `+` and `+` plays the role of `×`.
The field has been doing linear algebra on an object whose native arithmetic
is a different algebra entirely. (Analogy: averaging clock times. 11pm and 1am
"average" to noon, which is exactly wrong, because clock times are not ordinary
numbers.)

Three consequences follow once you switch to the matching algebra:

- **Flow → tropical regression.** A single object at constant local velocity
  produces a flat affine ramp in the time surface; its slope is the motion. In
  tropical terms this is fitting a max-plus affine function — reading the slope
  of the roof, in closed form.
- **Segmentation → polytope geometry.** Two motions produce a crease: a
  piecewise-affine structure, i.e. a tropical polytope. Finding the creases
  *is* the segmentation — combinatorial, no cluster count supplied, no EM
  initialization, no iteration.
- **Invariance → theorems.** Via Matheron's theorem, translation-invariant
  monotone operators are exactly the max-plus morphological ones. Certain
  contrast/threshold invariances become provable rather than measured.

That third point is the one the paper is built on. See §3.

---

## 3. The headline claim (this is the paper)

### The hardware fact nobody can engineer away

Every event camera's contrast threshold is assumed to be a single scalar `C`
(typically ~10%). It isn't. Circuit bias and manufacturing imperfection give
**every pixel a slightly different threshold** — fixed-pattern noise with a
measured standard deviation of roughly **2.5–4% contrast between pixels**
(Gallego et al., TPAMI survey). Hardware work that raises threshold sensitivity
still does *not* solve mismatch. It is a permanent property of the sensor.

The field's response has been: treat it as noise, filter or calibrate it, move
on. Every flow and segmentation method — classical and learned — silently
absorbs it as error.

### The empty gap

Meanwhile, the **certified robustness** community has spent a decade arguing
that empirical robustness is insufficient for safety-critical deployment and
that provable certificates are required. But every result in that literature
certifies **neural networks** against **ℓ_p pixel perturbations** (PROVEN,
auto-LiRPA/CROWN, formally-verified certifiers, Lipschitz homography
verification).

**Nobody has produced a robustness certificate against a *sensor-parameter*
perturbation, and nobody has a training-free operator whose structure makes
such a certificate derivable.** That intersection is empty.

### The claim

> Per-pixel threshold mismatch enters the max-plus time surface as a bounded
> **additive offset field along the time axis** (i.e. fixed-pattern noise
> shifts *when* a pixel fires, not the ordering of its own events). For a
> max-plus flow / motion-regime estimator, we prove a closed-form bound on
> output deviation as a function of mismatch magnitude — an invariance
> certificate — and validate it at measured hardware σ. Learned baselines
> (E-RAFT and similar) admit no such guarantee, and we measure their
> worst-case degradation to show the gap is real, not rhetorical.

### Why tropical algebra is the *right* tool, not a costume

This is the argument that elevates the paper above a reframing, so state it
explicitly in the intro:

Threshold mismatch is **spatially varying and non-differentiable**, so the
standard linear-relaxation machinery (LiRPA/CROWN) that powers NN certification
**does not apply**. Max-plus morphology, by contrast, handles *non-flat
structuring functions* natively — a per-pixel offset field is just a
structuring function instead of a flat structuring element (Maragos). The
tropical formalism is plausibly the only tractable route to this certificate.
"Right tool for a problem the incumbent tools structurally cannot touch" is
the spine of the contribution.

---

## 4. Contributions, in order of weight

| # | Contribution | Type | Risk |
|---|---|---|---|
| **C1** | Closed-form invariance certificate for per-pixel threshold mismatch on a training-free max-plus flow operator, validated at measured hardware σ | Theoretical + empirical | Medium — the theorem is the project |
| **C2** | Deterministic motion-regime decomposition via tropical polytope structure, with a provable per-event work bound; no supplied regime count, no iteration | Computational + theoretical | **High — hinges on E2 (§8)** |
| **C3** | TropiCam: open-source, CPU-only, deterministic event-vision engine + benchmark harness | Artifact | Low — ships regardless |

**C1 is the foundation.** It survives even if C2 comes back weak. Do not invert
this ordering; the earlier framing (flow speed / engine performance) is scooped
and cannot carry a prestige venue.

---

## 5. Prior art map, with the required delta against each

This is the section reviewers read first. Every entry below is a real
collision — each needs an explicit distinguishing sentence in related work.

### Flow (the crowded side)

- **Benosman et al. (~2014), local plane fitting / surface of active events
  (`LocalPlanesFlow`).** Fits affine planes to the time surface; least-squares;
  documented as noise-sensitive and requiring outlier rejection. FPGA
  implementations exist at sub-µs latency.
  **Delta:** the affine-ramp observation is *theirs, not yours*. Concede in
  the first related-work paragraph.
- **Nagata, Sekikawa & Aoki (2021), "Optical Flow Estimation by Matching Time
  Surface."** ⚠️ **Read this first.** Operates directly on the time surface
  (not a plane fit), with an **L1** smoothness-regularized loss.
  **Delta:** kills any "first non-L2 treatment of the time surface" claim.
  Your distinction is (a) closed-form slope read vs. optimized loss, and
  (b) the *certificate* — they have no invariance guarantee.

### Segmentation (your C2 territory)

- **Stoffregen et al. (2019), motion-compensation segmentation, ICCV.** First
  per-event method; jointly estimates event-object association and motion by
  maximizing an objective; ~90% accuracy at 4px displacement. Iterative.
- **Gallego et al. (2020), spatio-temporal graph cuts, TNNLS.** SOTA; notably
  **does not require predetermining the number of moving objects**.
  **Delta (important):** "no cluster count needed" is **already taken**. Your
  differentiator must be the *how* — deterministic, non-iterative, closed-form
  decomposition with a worst-case work bound — not the *what*.

### Tropical / max-plus (your formalism's neighbors)

- **S. & Iyer (2025), "Tropical Geometry Based Edge Detection Using Min-Plus
  and Max-Plus Algebra."** Reformulates convolution and gradients in min/max-plus
  on static grayscale images.
  **Delta:** tropical-algebra-for-vision is no longer novel per se. Your line
  is drawn at **asynchronous event data + certification**, and must be drawn
  explicitly.
- **Maragos, Charisopoulos & Theodosis (2021), *Tropical Geometry and Machine
  Learning*, Proc. IEEE;** plus Maragos on max-plus ↔ mathematical morphology.
  **Delta:** this is the **toolkit you import**. Cite generously, claim
  nothing here. The non-flat-structuring-function result is what makes C1
  tractable.
- **N'guessan (2026), PALMA** — dependency-free C library for tropical linear
  algebra on ARM; MIT-licensed; ~2,274 MOPS on a Raspberry Pi 4, sub-10µs
  scheduling solves; UAV/IoT case studies.
  **Delta:** your *engineering* novelty ("tropical algebra runs fast on a CPU")
  is **gone**. Cite it; consider building on it rather than reimplementing.
  All novelty must live in the vision formulation and the certificate.
- **UltraLIF (Miñoza, ICML 2026; arXiv Feb 2026).** Max-plus /
  ultradiscretization to make spiking neurons differentiable.
  **Delta:** orthogonal (training, not geometry). Cite as *motivation* — it
  legitimizes the max-plus/event pairing at a top venue while occupying a
  disjoint slot.

### Denoising (fully occupied — plumbing only)

- Delbruck-style background-activity filters (`t − max(neighborhood) < dt`);
  Guo & Delbruck (DND21), whose best variant uses an MLP over local time
  surfaces; plus several others.
  **Delta:** your "max-plus opening as native denoiser" is a **related-work
  paragraph and an ablation**, never a contribution. The observation that the
  field's standard denoiser *is already* a max-plus predicate is a nice
  unification sentence — one sentence.

### Certified robustness (the field you're entering)

- PROVEN (2018); auto-LiRPA (2020); formally-verified robustness certifier
  (2025); Lipschitz-based homography verification (2026).
  **Delta:** all NN + ℓ_p. **None touch sensor-parameter perturbations or
  training-free operators.** This absence *is* your opening — say so directly.

---

## 6. Scoping discipline — what you must NOT claim

Each of these survived stress-testing only in scoped form. Overclaiming any one
is a desk-reject.

1. **The affine ramp is not yours.** Benosman, ~12 years. Concede early.
2. **"Pieces ≠ objects."** A crease means a change in *local velocity*, not a
   new object — rotation, depth gradients, and occlusion boundaries all create
   creases with no second object. The honest claim is **"number of
   locally-affine motion regimes,"** never "number of objects."
3. **Convexity mismatch.** Off-the-shelf tropical regression fits a *max of
   hyperplanes* = convex piecewise-linear. A real time surface is **not**
   globally convex (plateaus in quiet regions, cliffs at boundaries, concave
   creases). Therefore: **sliding local window, never full-frame.** Architect
   this from day one. Genuinely non-convex piecewise-linear structure would
   need max-min or difference-of-tropical formulations — more mathematics than
   one semester holds.
4. **Noise: breakdown point is 1/n.** Max-plus does not average, so a single
   spurious high timestamp corrupts the local surface. Two mitigating facts,
   both of which belong in the paper:
   - The corruption is **transient and self-healing**: spurious timestamps are
     bounded above by `now`, and the next genuine event at that pixel
     overwrites it. Damage lifetime = local inter-event interval (µs in active
     regions). It persists only in quiet regions, which contribute no motion
     evidence anyway. It is **not** permanent.
   - The fix is **native**: a max-plus **opening** (erosion → dilation) is the
     canonical isolated-spike remover. Same semiring, no new mathematics.
5. **⚠️ The erosion latency cost — name it before a reviewer does.** Opening
   *begins* with an erosion, which replaces a timestamp with an older
   neighbourhood minimum. On a time surface that is **literally injecting
   delay**, proportional to the structuring radius and the local inter-event
   interval. It eats directly into the low-latency selling point. Measure it
   and report it as the explicit price of L∞ robustness.
6. **Invariance is two theorems, not one.**
   - Exact invariance under *spatially uniform* monotone transforms (Matheron).
   - Per-pixel mismatch is spatially *varying* and **not** covered by Matheron
     — it enters as a static per-pixel offset field (fixed-pattern noise),
     calibratable once, which in max-plus means moving from a flat structuring
     element to a non-flat structuring **function**. Bounded, calibratable
     degradation, with the residual measured.
7. **The honest baseline matchup** is *L2-plus-outlier-rejection* vs.
   *L∞-plus-opening* — both robustified, neither naked. Benosman's method is
   documented as needing outlier rejection; comparing your robustified method
   against a naked least-squares fit is a strawman a reviewer will catch.
8. **Terminology.** Say **"Newton polytope vertex count,"** not "tropical rank."
   There are several inequivalent tropical ranks (tropical, Kapranov, Barvinok)
   and they disagree on the same matrix; the quantity you want is the number of
   monomials attaining the max on some region — a vertex count on the upper hull
   of the Newton polytope. A geometry reviewer will catch loose usage.
9. **No hardware ⇒ no end-to-end latency claim.** You can measure *processing*
   throughput and per-event work on replayed streams. You **cannot** claim
   sensor-to-decision latency. Frame all timing results as processing latency.

---

## 7. Engine architecture & build order

CPU-only, deterministic, no GPU, no training.

### Build order (this ordering avoids the classic traps)

1. **Synthetic oracle first, before any real data.** Generator producing events
   for a bar translating at known velocity → analytic ground-truth flow,
   exactly known creases, zero noise. Every operator validates against it.
   Later, inject noise into the same generator and the robustness sweep comes
   for free on infrastructure you needed anyway.
   - ⚠️ **Aperture problem:** a translating vertical bar constrains only the
     component *normal* to the edge — it validates **normal flow**, not full
     flow. A "wrong" v_y may be correct behaviour. Add a **moving dot or
     corner** case where both components are recoverable, and label which test
     checks which quantity.
2. **NumPy-vectorized end-to-end, then port the hot loop.** Do **not** open in
   Rust/C++ with Python bindings — three weeks of FFI plumbing before anything
   moves on screen. Get the pipeline correct and the demo running first; port
   only the identified hot loop afterward. You will write a tenth as much
   native code and it will be the right tenth.
   - If **p99 latency** becomes a headline metric, port to **Rust or C++**, not
     Go — GC pauses land in exactly the plot you care about.
3. **Time surface = pre-allocated flat contiguous array.**
   - `uint32` timestamps with **wrapping unsigned subtraction** for age:
     `now − T` is correct mod 2³² as long as true age < ~71 min at µs
     resolution.
   - ⚠️ **Do NOT use a rolling epoch.** Rebasing forces a periodic O(N)
     stop-the-world sweep over the whole surface — a pause landing directly in
     the p99 plot. Same memory saving, no sweep, no spike.
   - Budget: ~7.4 MB at 1280×720 with **separate ON/OFF polarity surfaces** at
     32-bit. Decide polarity handling early — it changes whether you're
     arguing about L3 or main memory. Locality (row-major + neighbourhood
     access) is what saves you, not fitting in L2.
4. **⚠️ Correctness footgun.** `T[ys, xs] = np.maximum(T[ys, xs], ts)` is
   **last-write-wins** on duplicate pixels within a batch, *not* a max. Use
   `np.maximum.at`, or sort-and-reduce. **A clean translating bar fires each
   pixel exactly once, so your oracle will NOT catch this bug.** Add an
   explicit test with repeated events at the same pixel.
5. **Don't write parsers.** Use `tonic` for standard datasets (confirm DSEC
   coverage before building around it). Convert to a flat binary of packed
   structs and mmap it. Start on DVS-Gesture, not full DSEC — DSEC is enormous.

### Datasets

| Dataset | Use |
|---|---|
| **DVS-Gesture** | Starting sequence — small, fast iteration |
| **N-Caltech101** | Secondary sanity |
| **MVSEC / DSEC** | Flow with ground truth; the noise sweep (E1) |
| **EVIMO** | Motion-regime validation (E2) — has rotation & depth-varying scenes |
| **v2e / ESIM** | Synthesize events from ordinary video |
| **EVIS** (Isaac Sim plugin) | Per-pixel threshold-mismatch injection at controlled σ — directly supports E3 |

---

## 8. The three experiments that decide everything

### E1 — Noise sweep (robustness)
Inject background activity at measured rates (~0.1–10 events/px/s) into
MVSEC/DSEC. Plot angular error for three estimators:
1. tropical fit, raw
2. tropical fit + max-plus opening
3. Benosman plane fit + iterative outlier rejection

Plot the **erosion-induced delay** alongside the error curves.
**Pass criterion:** opened-tropical stays competitive past realistic low-light
noise rates.

### E2 — Motion-regime validation (**load-bearing for C2**)
EVIMO. Recovered **piece count vs. true object count**, deliberately including
**rotation and depth-varying scenes**, not just clean fronto-parallel
translation.
**Run this early — before investing in the full engine.** If the ratio is wild,
C2 collapses and the honest move is to report it as a negative result and lean
entirely on C1.

### E3 — Certificate validation (**load-bearing for C1**)
Simulate per-pixel threshold mismatch at measured hardware σ (2.5–4%) using
EVIS or v2e. Show:
- (a) measured output deviation stays **within the derived bound** — the
  certificate holds;
- (b) a learned baseline (E-RAFT or similar) has **no** such guarantee, with
  its worst-case degradation measured for contrast.

---

## 9. Ten-week plan

| Weeks | Focus |
|---|---|
| **0 (this week)** | Confirmatory prior-art pass on exact phrases *"certified event-based vision"* and *"threshold mismatch invariance"*. Read **Nagata 2021** in full. |
| **1** | Synthetic oracle (bar + dot/corner), NumPy time surface, OpenCV/Pygame visualizer, throughput ceiling measured, one real DVS-Gesture sequence rendering. |
| **2** | **E2 early** — tropical decomposition prototype, EVIMO regime-count check. Gate on C2. |
| **3–4** | Max-plus morphological operators (erosion/dilation/opening), vectorized; E1 noise sweep; erosion-delay measurement. |
| **4–6** | **C1 derivation** — mismatch → time-axis offset field → bounded deviation via non-flat structuring function. This is the paper's core; give it real time. |
| **6–7** | E3 certificate validation; learned baseline comparison. |
| **7–8** | Port hot loops to Rust/C++ if p99 is a headline; benchmark harness; latency histograms. |
| **9–10** | Paper draft, figures, repo polish, demo video, prof deliverable. |

---

## 10. Risks & kill criteria

| Risk | Trigger | Response |
|---|---|---|
| C2 collapses | E2 shows piece count ≫ object count on rotation/depth | Report as honest negative result; paper rests on C1 alone. **Not fatal.** |
| C1 theorem doesn't close | Bound is vacuous or unprovable in available time | Fall back to *empirically measured* invariance + the structural argument; downgrade venue target. Weakens but doesn't kill. |
| Someone published the certificate | Week-0 search finds it | Opening narrows to a sharper delta rather than dying — but you must know **now**, not in October. |
| Scope creep into theory-only | Pattern risk: theory fully written, experiments unrun (cf. SWG) | Hard rule: **no theorem section written before E2 and E3 have numbers.** |
| Native-first trap | Starting in Rust week 1 | Explicitly forbidden in §7. NumPy first. |

**What ships regardless:** the engine (C3) — a working, open, deterministic
event-vision stack with a benchmark and a demo. That artifact is the floor and
it is not contingent on any theorem.

---

## 11. Venue strategy

| Target | Framing to lead with |
|---|---|
| **NeurIPS / ICLR** | Certified robustness against a **novel perturbation class** (sensor-parameter), on a training-free operator. Their exact register. **Primary target.** |
| **CVPR / ICCV** | Lead with the vision result (flow + motion-regime segmentation), certificate as the theoretical backbone. |
| **TPAMI** | Journal-length treatment combining C1 + C2 + engine. Good fallback with a longer horizon. |
| **RSS / CoRL** | Only if you foreground the worst-case latency guarantee for robotics. |

**The positioning move that fixes the prestige problem:** stop framing this as
an optical-flow method (crowded, scooped — competing on angular error against
E-RAFT and Nagata) and frame it as **certified robustness against a
sensor-parameter perturbation** (empty, uncontested, and the incumbents
structurally cannot follow). Same math, same engine, same datasets. The
contribution moves from *incremental and already-touched* to *significant and
uncontested*, at zero implementation cost.

---

## 12. Corrections to the earlier (Gemini) master doc

If you're working from that PDF, fix these three before coding:

1. **§9 "Intellectual Property Strategy" — delete it.** It recommends keeping
   the core closed-source as a precompiled binary to preserve patentability.
   This guts the project's entire rationale (an *open, inspectable*
   deterministic engine), kills the demo's visibility and the preprint, and
   contradicts the standing IP position (publication as sole strategy).
   Separately, TropiCam is **not patentable in India** — max-plus regression on
   a CPU is math + computer-programme *per se*, the textbook §3(k) exclusion.
   Replace with: open-source under a permissive licence; publish.
2. **The "32-bit epoch trick" is the wrong version.** Rolling-epoch rebasing
   forces a periodic O(N) stop-the-world sweep. Use raw uint32 + wrapping
   unsigned subtraction (§7.3).
3. **Missing validation.** That doc lists "motion regimes" as a rendered
   deliverable but contains **no experiment validating that piece count tracks
   object count**. That's E2 — the single experiment separating a real
   contribution from a reframing. Add it, or downgrade the deliverable to
   "exploratory."

Minor: the ADAS "millisecond reaction time, surpassing 33ms" application
overclaims — without hardware you measure *processing* latency, not reaction
time.

---

## 13. Immediate next actions

1. **Week-0 search** (≈1 hour): exact phrases *"certified event-based vision"*,
   *"threshold mismatch invariance"*, *"event camera certified robustness"*.
   Confirm the gap is empty.
2. **Read Nagata 2021** in full (one evening). Draft the paragraph
   distinguishing your closed-form + certified approach from their
   L1-optimized time-surface matching.
3. **Write the synthetic bar generator tonight** (~50 lines). Nothing about the
   reframe changes the week-1 code.

---

## Appendix — key sources

- Gallego et al., 2019. *Event-based Vision: A Survey.* TPAMI. arxiv.org/abs/1904.08405
- *Event Camera Calibration of Per-pixel Biased Contrast Threshold*, 2020. arxiv.org/abs/2012.09378
- *EVIS: A Physics-Grounded Event Camera Plugin for NVIDIA Isaac Sim*, 2026.
- Nagata, Sekikawa & Aoki, 2021. *Optical Flow Estimation by Matching Time Surface.* Sensors.
- Benosman et al. *Event-based visual flow / surface of active events.*
- Stoffregen et al., 2019. *Event-Based Motion Segmentation by Motion Compensation.* ICCV.
- Gallego et al., 2020. *Event-Based Motion Segmentation with Spatio-Temporal Graph Cuts.* TNNLS.
- Maragos, Charisopoulos & Theodosis, 2021. *Tropical Geometry and Machine Learning.* Proc. IEEE.
- S. & Iyer, 2025. *Tropical Geometry Based Edge Detection Using Min-Plus and Max-Plus Algebra.*
- N'guessan, 2026. *PALMA* — tropical linear algebra for ARM embedded systems.
- Miñoza, 2026. *UltraLIF.* ICML 2026 (arXiv Feb 2026).
- Guo & Delbruck. *Low-cost event denoising (DND21).*
- Certified-robustness cluster: PROVEN (2018), auto-LiRPA (2020),
  *A Formally Verified Robustness Certifier* (2025), Lipschitz homography
  verification (2026).

*Sources gathered via targeted search, not an exhaustive systematic review.
The week-0 confirmatory pass in §13 is not optional.*
