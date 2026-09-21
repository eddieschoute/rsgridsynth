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
- measured (`small_angle_beats_mixed_diagonal_for_small_theta` test): mean T-count ~0.10,
  vs. ~4.43 for `synth_mixed_diagonal` at the same `(theta, delta)` -- a ~44x reduction,
  consistent with the paper's own claims for this regime (the exact ratio moves as the
  implementation changes -- e.g. removing the early-stop-after-first-hit optimization, see
  simplification 4 below, changed it from an earlier ~100x -- but the qualitative win is
  stable).

## Simplifications made relative to the papers (and why)

1. **No `phi_0` angular performance floor** (Bothe Eq. 64/E10). That floor only trades region
   area against average success probability to shave a lower-order term off the *asymptotic*
   T-count estimate; omitting it costs some efficiency in the very-small-`(theta, delta)`
   tail, never correctness. `SmallAngleRegion` always searches the full accuracy-limited area.
2. **Bounding ellipse is an exact circular cap, not Bothe's 5-point Khachiyan MVEE
   construction** (updated from an earlier, much looser "plain isotropic circle" -- see
   `SmallAngleRegion`'s own doc comment for the `x_min` derivation and why it matters:
   the full-disk version made the live lattice search pay the cost of searching the entire
   disk on every call). Still not the tightest possible bound (Bothe's MVEE would be tighter
   still), but a large, measured improvement over the original choice. `CandidateStats`
   (`src/gridsynth.rs`) exists to measure any further gap; not wired up in this change.
3. **`intersect`'s hyperbola clip over-approximates the non-convex (`q2 < 0`) branch** by
   leaving the interval unclipped rather than computing the exact two-ray union. Per this
   crate's `Region` contract, `intersect` may over-report without affecting correctness --
   `solve_tdgp` always re-checks `inside` on every candidate, so this only costs efficiency.
   This is the concern flagged (before this region existed) in the local design doc's
   "Region::intersect convexity question" section.
4. **Bounded-depth search, not full enumeration.** Bothe explores "all solutions inside the
   region up to a certain T count" and scores each by `p * T-count`; this implementation
   caps the search at `DEFAULT_MAX_LIVE_SEARCH_K` steps (default 8; callable with an explicit
   override, or no cap at all, via `synth_small_angle_with_max_k`) rather than an unbounded or
   T-count-targeted search, and -- unlike an earlier version of this change, which stopped
   early a fixed number of steps after the first hit -- always explores the *entire*
   `0..=max_k` range once started, since the cap is already tuned to a small, acceptable
   latency bound and stopping early would only risk returning a worse candidate for no
   latency benefit. A tighter search (targeting a specific T-count budget rather than a step
   count) remains a possible future refinement.

## Implemented beyond the original plan

The original version of this document deferred three of Bothe's techniques as "analysis
only." One of them has since been implemented, because the live search above turned out to
be too slow for practical use without it:

- **Bothe's exact lookup table** (`src/protocol/small_angle_table.rs`): all 56 rows of
  Tables II/III (theta-independent, up to T-count 35 -- 55 from the paper's original
  publication, plus one, `tan(alpha)=1.32e-02`, supplied directly by the paper's authors
  after we found it missing via a Table-II/Table-III cross-diff), transcribed from the
  paper's own LaTeX source. `synth_small_angle` tries every row (via this crate's own exact
  `mixture_weight`, not the paper's quasi-probability bookkeeping) before running the live
  search, accepting a hit only if it beats a cheap even-split cost estimate. This resolves
  the common case (roughly `delta` in the `1e-4`-`1e-6` range, for `theta` above the table's
  own floor of ~1.93e-3 rad) near-instantly; the live search remains necessary for smaller
  `theta` (below that floor, where no table row is ever close enough) and is the reason the
  bounded-depth search above (simplification 4) exists at all. Still deferred, as analysis
  only: the asymptotic cost formulas and quasi-probability mode, for the same reasons as
  originally noted -- and now also the small-angle fallback variant, not revisited in this
  round of work.

## Verification performed

- `make ci` (fmt-check + clippy `-D warnings` + full test suite, `--all-features --all-targets`,
  plus doc tests, including the pre-existing `protocol_accuracy_fuzz_test` fuzz suite) is
  green.
- Absolute correctness: every synthesized branch's achieved diamond-norm error is recomputed
  independently via `AchievedDiamondError` and checked against the requested `delta`
  (`synth_small_angle_finds_a_mixed_result_for_small_theta`,
  `negative_angle_achieves_its_own_target_after_x_conjugation`).
- Regression guard: the dispatcher (`synth_mixed_diagonal`) never costs more than
  `even_split_search` alone when `small_angle_could_help` says yes
  (`small_angle_never_worse_than_even_split_when_pre_check_says_yes`).
- Regime win: mean T-count is measurably lower than the angle-independent formula for small
  `theta` (`small_angle_beats_mixed_diagonal_for_small_theta`), with a log-log slope fit
  against Bothe's Eq. (66) power law (`small_angle_slope_fit_and_per_point_accuracy`).
- Canonicalization: `theta` and `theta - 2*pi` (same channel) synthesize to the same T-count;
  negative `theta` round-trips correctly through the `X`-conjugation branch-swap fix (the one
  real bug this specific check caught during the original implementation -- see
  `apply_x_conjugation`'s doc comment).
- Search-depth cap: a case confirmed to exceed `DEFAULT_MAX_LIVE_SEARCH_K` falls back to
  `even_split_search` within a bounded time and still meets its own budget
  (`deep_search_falls_back_to_even_split_within_a_bounded_time`).
- Table correctness: every row decodes to its stated T-count, the `Y`/`Z`-letter
  substitutions are checked as direct algebraic identities (not just against the table), and
  every row's decoded magnitude matches Table II's independently-transcribed value (see
  `small_angle_table.rs`'s own test suite).
- Two further, independent correctness bugs were found and fixed while making this feature
  practically usable, not part of the original plan: `mixture_weight`'s closed form silently
  assumed unit modulus (`src/protocol/mixing.rs`), and this in turn exposed a second bug in
  `mixed_fallback`'s error accounting (`src/protocol/mixed_fallback.rs`) that had been masked
  by the first. See the PR description for detail on both.

## Not implemented from the original plan

- `max_t_count()` was added to `MixedDiagonalResult` (mean *and* max, as requested), but no
  new region-search instrumentation was wired to `CandidateStats` (see simplification 2
  above).
- `PhiFloor` enum (the `phi_0` tuning knob) was dropped per simplification 1.
