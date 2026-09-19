# Stage 4: small-angle mixed diagonal synthesis (Appendix C)

> This is the "Stage 4" section referenced by the (untracked, local-only)
> `mixed_fallback_synthesis.md` design doc's staging plan. It lives here, as a tracked file,
> because that doc is untracked in the main checkout and outside this change's worktree;
> merge the two by hand if you keep that doc around.

## Context

Every protocol implemented so far in this crate (`diagonal`, `fallback`, `mixed_diagonal`,
`mixed_fallback`) has an **angle-independent** T-count formula: the cost of synthesizing
`R_z(1e-6)` is the same as `R_z(1.0)` at the same accuracy. The two mixed protocols achieve
their factor-of-2 improvement by splitting the diamond-norm error budget *evenly* between an
under- and an over-rotation (Kliuchnikov Prop. 3.13/3.18) -- see
`MixedDiagonalRegion`'s symmetric cap at `Re(w) >= sqrt(1 - eps/2)`.

Kliuchnikov et al.'s Appendix C observes that the even split leaves value on the table: if one
branch can be pinned to something free (the identity gate, valid whenever
`2*|sin(theta/2)| > delta`), the *entire* error budget can go to the other branch, and because
the resulting mixture weight `p` on that branch is itself small for small `theta`, the mean
T-count collapses -- multiplied down twice (larger search region *and* small `p`). Bothe,
Sünderhauf, Witham, Campbell, Blunt (arXiv:2605.31544v2) works this regime out fully; their
Appendix D ("Adapted gridsynth implementation") specifies the resulting region against
exactly the three-function `Region` interface (`ellipse`/`inside`/`intersect`) this crate
already exposes (`src/tdgp.rs`).

This document is the comparison and design rationale for `src/protocol/small_angle.rs`,
which implements this. See that module's own doc comments for the concrete derivation.

## The regions are one family

| | Condition on top-left entry | Shape | Implemented as |
|---|---|---|---|
| Diagonal (Prop. 3.7) | `\|Re(w)\| >= sqrt(1-eps^2/4)` | circular cap, height ~eps^2/8 | `EpsilonRegion` |
| Mixed diagonal (Prop. 3.13) | `\|Re(w)\| >= sqrt(1-eps/2)`, sign-split | circular cap, height ~eps/4 | `MixedDiagonalRegion` |
| Appendix C / Bothe §V | unit disk ∩ hyperbola ∩ sign constraints | curved sliver | `SmallAngleRegion` |

The first two differ only in the offset of a *straight* cutting line. Once the split stops
being even, the admissible set for the searched branch is no longer a half-plane -- it's a
hyperbola, because the mixture weight `p` (and hence the total error after mixing) depends
nonlinearly on the searched candidate's position. `SmallAngleRegion`'s doc comments give the
derivation directly from this crate's own `mixture_weight` closed form (working in the
`w`-frame, i.e. `WFrame`'s rotated coordinates, rather than Bothe's absolute `u`-coordinates)
-- reaching the same curve as Bothe's Eq. (D3) independently, related by the
`theta_paper = -theta_crate/2` identification.

**The T-count savings come from the asymmetric error split, not from quasi-probabilities.**
Bothe's own Appendix F shows the probability and quasi-probability formulations of this same
region agree to `O(theta^2)`.

## What's reused unchanged

No new mixing mathematics was needed: `mixture_weight` (`src/protocol/mixing.rs`) already
implements the fully general closed form (it never assumed an even split -- that assumption
lived in the *region*, not the weight formula). Also reused as-is: `WFrame`, the `{I,S,Z,SZ}`
twirl, `setup_regions_and_transform`, `solve_tdgp`, `process_solution_candidate`,
`assemble_result`/`StraddleOutcome` (both made reachable from `small_angle.rs` via their
existing `pub(crate)` visibility), and `AchievedDiamondError`.

## Worked example

`theta = 1e-3`, `delta = 1e-4` (comfortably in `delta >> theta^2 = 1e-6`):
- existing angle-independent mixed diagonal: `1.52*log2(1e4) - 0.01 ~= 20` T gates (mean)
- measured (`small_angle_beats_mixed_diagonal_for_small_theta` test): mean T-count ~0.044,
  vs. ~4.43 for `synth_mixed_diagonal` at the same `(theta, delta)` -- a ~100x reduction,
  consistent with the paper's own two-orders-of-magnitude claims for this regime.

## Simplifications made relative to the papers (and why)

1. **No `phi_0` angular performance floor** (Bothe Eq. 64/E10). That floor only trades region
   area against average success probability to shave a lower-order term off the *asymptotic*
   T-count estimate; omitting it costs some efficiency in the very-small-`(theta, delta)`
   tail, never correctness. `SmallAngleRegion` always searches the full accuracy-limited area.
2. **Bounding ellipse is the plain isotropic circle**, not Bothe's 5-point Khachiyan MVEE
   construction. Per this crate's `Region` contract, `ellipse` may over-report without
   affecting correctness -- only `to_upright` efficiency. `CandidateStats`
   (`src/gridsynth.rs`) exists to measure this trade-off; it was not wired up in this change
   (follow-up, if the simpler ellipse turns out costly in practice).
3. **`intersect`'s hyperbola clip over-approximates the non-convex (`q2 < 0`) branch** by
   leaving the interval unclipped rather than computing the exact two-ray union. Same
   justification as (2): `solve_tdgp` always re-checks `inside` on every candidate, so this
   only costs efficiency, never correctness. This is the concern flagged (before this region
   existed) in the local design doc's "Region::intersect convexity question" section.
4. **Bounded-search scoring, not full enumeration.** Bothe explores "all solutions inside the
   region up to a certain T count" and scores each by `p * T-count`; this implementation
   scores every candidate found within a fixed window (`EXTRA_K_STEPS_AFTER_FIRST_HIT = 3`)
   past the first hit, not an unbounded or T-count-targeted search. A tighter search
   (targeting a specific T-count budget rather than a step count) is a possible future
   refinement.

## Deferred (analysis only, not implemented)

- **Bothe's exact lookup tables** (Tables I-III, up to T-count 35): a static, theta-independent
  table that dominates the `delta >~ theta/500` regime with no synthesis at all. Cheap
  follow-up; doubles as a correctness oracle for this region.
- **Asymptotic cost formulas** (Eq. 10/11/63/66): closed-form T-count estimates for resource
  estimation without running synthesis. Independent of everything here; worth doing only once
  there's a named cost-model consumer.
- **Quasi-probability mode**: buys nothing in T-count at equal `delta` for a single rotation
  (per Appendix F); its real benefit is circuit-level (multiplicative rather than additive
  sampling-overhead composition across many rotations), which is explicitly a caller's
  concern, not this crate's (see `small_angle.rs`'s module doc on per-rotation scope).
- **Fallback small-angle variant** (`delta -> 2*delta` substitution onto `synth_mixed_fallback`):
  nearly free once this region exists; best value-per-effort follow-up of the four.

## Verification performed

- `make ci` (fmt-check + clippy `-D warnings` + full test suite, `--all-features --all-targets`,
  plus doc tests) is green with this change.
- Absolute correctness: every synthesized branch's achieved diamond-norm error is recomputed
  independently via `AchievedDiamondError` and checked against the requested `delta`
  (`synth_small_angle_finds_a_mixed_result_for_small_theta`,
  `negative_angle_achieves_its_own_target_after_x_conjugation`).
- Regression guard: the combined entry point (`synth_small_angle_or_mixed`) never costs more
  than the existing `synth_mixed_diagonal` alone
  (`combined_entry_point_never_worse_than_mixed_diagonal`).
- Regime win: mean T-count is measurably lower than the angle-independent formula for small
  `theta` (`small_angle_beats_mixed_diagonal_for_small_theta`).
- Canonicalization: `theta` and `theta - 2*pi` (same channel) synthesize to the same T-count;
  negative `theta` round-trips correctly through the `X`-conjugation branch-swap fix (the one
  real bug caught during implementation -- see `apply_x_conjugation`'s doc comment).

## Not implemented from the original plan

- `max_t_count()` was added to `MixedDiagonalResult` (mean *and* max, as requested), but no
  new region-search instrumentation was wired to `CandidateStats` (see simplification 2
  above).
- `PhiFloor` enum (the `phi_0` tuning knob) was dropped per simplification 1.
