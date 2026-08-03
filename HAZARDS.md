# Hazard register

Claims discipline for TropiCam. Read before writing any paper text or
reporting any number. Complements the master doc's section 6, which this
register **corrects in two places** (H5, H8).

Severity: **S1** could invalidate a contribution · **S2** reviewer will catch
it · **S3** we are underselling something.

---

## S1 — could invalidate a contribution

### H1. The mismatch model is conditional, and the condition is not yet stated
A pixel fires when `|Δ log I| >= C_i`. With local log-intensity rate `r`, the
inter-event interval is `C_i / r`, so threshold mismatch is a **rate** effect,
not a constant time offset. Nominal drift over k events is `k·δC/r`, unbounded
in k.

The rescue: the time surface keeps only the most recent event, so deviation in
`T(x,y)` is bounded by roughly one inter-event interval. But then

```
deviation  <~  sigma_C / r_local
```

**and this diverges as `r -> 0`.** In slow or low-contrast regions the
certificate is vacuous. This is physically real, not an artifact.

*Action:* C1 must be conditioned on a contrast-rate floor, or stated
per-region. Never report a bound without the `r_min` it assumed —
`metrics.contrast_rate_floor` exists to force this.

*Known gap:* `synthetic.inject_threshold_mismatch` uses a scene-wide constant
`contrast_rate`, so it **cannot currently exhibit this failure mode**. Needs a
spatially varying `r` before it is fit for E3 rehearsal.

### H2. "Closed form" is not yet established
The natural tropical loss is L-infinity, and **L-inf plane fitting is a linear
program, not a closed form**. Max-plus residuation (`x = A \ b`) *is* closed
form but solves the one-sided problem: it returns the greatest subsolution of
`A (x) x <= b`, with equality only when the system is consistent.

*Action:* settle the estimator's formulation before writing it. Defensible
wording until then: "non-iterative", "single-pass", or "closed form given a
fixed direction set". A tropical-geometry reviewer knows residuation's exact
semantics.

### H3. The certificate may not cover the shipped pipeline
Section 6.4 ships a max-plus opening as the denoiser; section 3 derives the
certificate on the flow estimator. Erosion over a **non-flat** structuring
function interacts with the offset field, so a bound proved for the raw
estimator does not automatically transfer to the opened pipeline.

*Action:* derive C1 for the pipeline *including* the opening, or ship
un-opened and report denoising separately. Certifying one system and shipping
another is a desk reject.

---

## S2 — reviewer will catch it

### H4. Novelty that is already gone
Affine ramp (Benosman) · non-L2 time surface (Nagata 2021) · tropical algebra
for vision (S. & Iyer 2025) · no-cluster-count segmentation (Gallego 2020) ·
tropical algebra fast on CPU (PALMA) · max-plus opening as denoiser (Delbruck
filters already are this).

Surviving novelty is one intersection: **sensor-parameter perturbation x
training-free operator x provable certificate.**

### H5. E2 as worded contradicts section 6.2
The doc says compare "recovered piece count vs. true object count", but 6.2
establishes pieces are not objects. Those counts are incomparable *by our own
argument*.

*Correction, implemented in `metrics.regime_agreement`:* score
**refinement/purity** — every regime should lie within one object; one object
may contain several regimes. Count ratio is descriptive output only and must
never be reported as accuracy.

### H6. Matheron is likely the wrong citation
Matheron's theorem: increasing + translation-invariant operators admit a
sup-of-erosions representation. The property actually invoked — invariance
under monotone transforms of the *value* range — is commutation with
anamorphosis, a property of **flat** morphological operators. Both true,
different results. Same class of error as the "tropical rank" caution in 6.8.

### H7. The 157 M events/s number is ingest-only
It measures `T <- max(T,t)` and nothing else, single-threaded, on synthetic
uniform-random events. A local-window flow estimator at W=9 does ~81 ops per
event; the full pipeline will be **1-2 orders of magnitude slower**.

*Action:* never let this number appear near the word "pipeline". Always label
it *time-surface ingest*.

### H8. "O(1) per event" needs its constant stated
Ingest is genuinely O(1). Flow over a WxW window is O(W^2); opening at radius
r is O(r^2) — constant only because W and r are fixed. Correct phrasing:
"constant work per event, independent of stream length and sensor resolution,
constant = window area".

### H9. Evaluation is not CPU-only
E3(b) needs a learned baseline (E-RAFT), which needs a GPU and torch. The
*artifact* is CPU-only; the *evaluation* is not. Keep the words apart.

### H10. Missing prior art: semantic / photometric perturbation certification
Section 5 lists NN + l_p certification but omits certified robustness to
brightness, contrast and geometric transforms. A reviewer will ask "isn't
threshold mismatch just a contrast perturbation, already covered?"

*Answer to have ready:* those certify *networks* against *image-space*
photometric change; this is a *sensor parameter* manifesting as *timing*, on a
*training-free* operator. Defensible — but only if cited and stated.

### H11. Prof-facing scope vs. paper framing
The submitted description sells speed and edge deployment; section 11 says
that framing is scooped. Section 11's point is that the reframe costs zero
implementation — but the two audiences are being told different stories.
Manage deliberately.

### H12. Citations are unverified
The master doc states its sources were "gathered via targeted search, not an
exhaustive systematic review". Several entries are 2025-2026 and none have
been checked: PALMA (N'guessan 2026), UltraLIF (Minoza, ICML 2026), EVIS
(2026), S. & Iyer (2025), formally-verified certifier (2025), Lipschitz
homography (2026).

*Action:* verify every one against a real DOI/arXiv ID before it enters a .bib
file. A hallucinated citation is fatal in a way nothing else here is.

---

## S3 — we are underselling

### H13. The certificate is global; NN certificates are local
Certified robustness for networks is almost always *per-input local*
robustness — a ball around one input, recomputed per input. A closed-form
bound on a training-free operator holds **for all inputs, unconditionally**
(subject to H1's conditioning). That is a categorically stronger class of
guarantee and is currently not claimed at all. Foreground it.

### H14. No training implies no distribution shift, no dataset bias
The certificate cannot be invalidated by retraining, fine-tuning or
deployment drift, and is architecture-free.

### H15. Determinism + per-event work bound is a WCET argument
That is the language safety certification actually speaks. Currently framed as
performance; it is really a *certifiability* property, and pairs with C1.

### H16. Same-pixel temporal differences are exactly mismatch-invariant
Because the offset field is static, it cancels identically in
`T_k(x,y) - T_{k-1}(x,y)`. Part of the estimator may be **exactly** invariant
while only the spatial part needs a bound. "Exact on this component, bounded
on that one" is far stronger than one global bound. (Degrades where `r` varies
between successive events — see H1.)

### H17. The corner oracle is a strong asset
`T = max(affine, affine)` is an exact tropical polynomial with an
analytically known crease: regime ground truth needing no dataset and no
annotation.

---

## Code-level landmines

- **`t = 0` sentinel.** `has_fired()` treats `T == 0` as "never fired". Real
  datasets that start at `t = 0` would have their first events made invisible.
  Normalise timestamps to `t >= 1` on ingest.
- **Polarity encoding.** Datasets using `-1/+1` wrap to `255` in our `u8` and
  are silently dropped. The out-of-bounds counter catches it *only if read*.
- **Event dtype layout.** 9 bytes at offsets 0/2/4/8, asserted on both sides.
  Changing `EVENT_DTYPE` without changing the Rust `Event` breaks every field
  silently — the layout assertions exist to prevent this; do not remove them.
