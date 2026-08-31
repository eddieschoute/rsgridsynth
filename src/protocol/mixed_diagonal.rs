// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Stage 1: the mixed-diagonal region and the single-enumeration straddling-pair search.
//!
//! Implements the "mixed diagonal" rotation-synthesis protocol of Kliuchnikov, Lauter,
//! Minko, Paetznick, Petit, "Shorter quantum circuits via single-qubit gate approximation"
//! (Quantum 7, 1208 (2023), arXiv:2203.10064v2). Instead of finding one candidate close to
//! the target direction, this finds two candidates that straddle it (one under-rotated, one
//! over-rotated) and mixes them with a classical probability chosen so their first-order
//! rotation errors cancel exactly, at roughly half the T-count of the plain single-candidate
//! ("diagonal") protocol for the same diamond-norm accuracy.

use crate::accuracy::{diagonal_diamond_distance, AchievedDiamondError, WFrame};
use crate::common::Prec;
use crate::config::{config_from_theta_epsilon, GridSynthConfig};
use crate::diophantine::diophantine_dyadic;
use crate::gate::GateSeq;
use crate::gridsynth::{process_solution_candidate, setup_regions_and_transform, PhaseMode};
use crate::gridsynth::{UnitDisk, UprightTransform};
use crate::math::solve_quadratic;
use crate::normal_form::{conjugate_by_clifford, Clifford};
use crate::protocol::mixing::{diamond_to_spec_epsilon, mixture_weight};
use crate::region::Ellipse;
use crate::ring::{DOmega, DRootTwo, ZRootTwo};
use crate::synthesis_of_clifford_t::decompose_domega_unitary;
use crate::tdgp::{solve_tdgp, Region};
use crate::unitary::DOmegaUnitary;

use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;
use dashu_int::ops::Abs;
use dashu_int::IBig;
use nalgebra::{Matrix2, Vector2};

/// Same 2x2 matrix product helper as `gridsynth::matrix_multiply_2x2` (that one is private to
/// its module and not reachable from here, so it is duplicated verbatim rather than plumbed
/// through a new `pub(crate)` export, per the "only touch `mixed_diagonal.rs`" constraint).
fn matrix_multiply_2x2(
    prec: Prec,
    a: &Matrix2<FBig<HalfEven>>,
    b: &Matrix2<FBig<HalfEven>>,
) -> Matrix2<FBig<HalfEven>> {
    let mut result = Matrix2::from_element(prec.ib(IBig::ZERO));
    for i in 0..2 {
        for j in 0..2 {
            let mut sum = prec.ib(IBig::ZERO);
            for k in 0..2 {
                sum += &a[(i, k)] * &b[(k, j)];
            }
            result[(i, j)] = sum;
        }
    }
    result
}

/// The relaxed-offset cap region used by the mixed-diagonal protocol's straddling-pair
/// search. Structurally identical to [`crate::gridsynth::EpsilonRegion`] (same ellipse
/// construction pattern, same disc+half-plane `inside`/`intersect` predicates), but built
/// from the *exact* (not asymptotic) closed forms for the offset, radial semi-axis, and
/// tangential semi-axis of a circular cap cut at `Re(w) = d`:
///
/// - `d = sqrt(s) * sqrt(1 - eps/2)` (offset of the cutting line from the origin, along the
///   target direction; note `eps/2`, not the plain-diagonal protocol's `eps^2/4`).
/// - `h = sqrt(s) - d` (radial semi-axis: exact, not the asymptotic `eps^2*sqrt(s)/8`).
/// - `c = sqrt(s * eps/2)` (tangential semi-axis: exact circle-chord half-length at `Re(w) =
///   d`, derived from `sqrt(s - d^2) = sqrt(s - s*(1-eps/2)) = sqrt(s*eps/2)`).
///
/// Both branches of the mixed protocol (under- and over-rotation) share this single region;
/// which branch a solved candidate belongs to is decided later by the sign of `Im(w)`, not
/// by the region itself -- see [`search_for_straddling_pair`].
#[derive(Debug)]
pub struct MixedDiagonalRegion {
    scale: ZRootTwo,
    d: FBig<HalfEven>,
    z_x: FBig<HalfEven>,
    z_y: FBig<HalfEven>,
    ellipse: Ellipse,
    prec: Prec,
}

impl MixedDiagonalRegion {
    pub fn new(
        prec: Prec,
        theta: &FBig<HalfEven>,
        epsilon: &FBig<HalfEven>,
        scale: ZRootTwo,
    ) -> Self {
        let two = prec.fb(FBig::try_from(2.0).unwrap());
        let theta_half = prec.fb(theta / &two);
        let neg_theta_half = -prec.fb(theta_half);
        let z_x: FBig<HalfEven> = prec.fb(neg_theta_half.cos());
        let z_y: FBig<HalfEven> = prec.fb(neg_theta_half.sin());
        Self::from_target_direction_impl(prec, z_x, z_y, epsilon, scale)
    }

    /// Builds the same region as [`MixedDiagonalRegion::new`], but from the target
    /// direction's half-angle `(cos(-phi/2), sin(-phi/2))` directly, avoiding an `atan2`-style
    /// angle round-trip -- mirrors
    /// [`crate::gridsynth::EpsilonRegion::from_target_direction`]. Used by "mixed fallback"
    /// (a later stage) to build a mixed-diagonal *correction* region for a residual angle
    /// that only exists as an algebraically-derived `(cos, sin)` pair, not a raw `theta`.
    pub(crate) fn from_target_direction(
        prec: Prec,
        z_x: FBig<HalfEven>,
        z_y: FBig<HalfEven>,
        epsilon: &FBig<HalfEven>,
        scale: ZRootTwo,
    ) -> Self {
        Self::from_target_direction_impl(prec, z_x, z_y, epsilon, scale)
    }

    fn from_target_direction_impl(
        prec: Prec,
        z_x: FBig<HalfEven>,
        z_y: FBig<HalfEven>,
        epsilon: &FBig<HalfEven>,
        scale: ZRootTwo,
    ) -> Self {
        let two = prec.fb(FBig::try_from(2.0).unwrap());
        let one = prec.ib(IBig::ONE);
        let zero = prec.ib(IBig::ZERO);
        let half_eps = epsilon / &two;
        // `epsilon` >= 2 (e.g. a derived, rescaled correction-step epsilon on a
        // near-degenerate candidate, as in `mixed_fallback::build_side`) is already past the
        // point where any point of the disk fails to qualify; the sane mathematical limit of
        // the formula below is `one_minus_half_eps = 0`, not a negative radicand. Clamp
        // rather than let that panic in `FBig::sqrt`, matching the analogous clamp in
        // `gridsynth::EpsilonRegion::from_target_direction_impl`.
        let one_minus_half_eps = (&one - &half_eps).max(zero.clone());
        let scale_to_real = scale.to_real(prec);

        // Exact offset, radial semi-axis, and tangential semi-axis -- see struct docs.
        let sqrt_s = scale_to_real.sqrt();
        let sqrt_one_minus_half_eps = one_minus_half_eps.sqrt();
        let d = &sqrt_s * &sqrt_one_minus_half_eps;
        let h = &sqrt_s - &d;
        let s_half_eps = &scale_to_real * &half_eps;
        let c = s_half_eps.sqrt();

        let h_sq = &h * &h;
        let c_sq = &c * &c;
        let inv_h_sq = &one / &h_sq;
        let inv_c_sq = &one / &c_sq;

        // Same d1/d2/d3 matrix-product pattern as `EpsilonRegion::new`: d1/d3 rotate into and
        // out of the (radial, tangential) frame, d2 is the diagonal quadratic form in that
        // frame with the radial (thinner) direction first.
        let neg_z_y: FBig<HalfEven> = -(z_y.clone());
        let d1: Matrix2<FBig<HalfEven>> =
            Matrix2::new(z_x.clone(), neg_z_y.clone(), z_y.clone(), z_x.clone());
        let d2: Matrix2<FBig<HalfEven>> =
            Matrix2::new(inv_h_sq, zero.clone(), zero.clone(), inv_c_sq);
        let d3: Matrix2<FBig<HalfEven>> =
            Matrix2::new(z_x.clone(), z_y.clone(), neg_z_y, z_x.clone());

        let px = &d * &z_x;
        let py = &d * &z_y;
        let p = Vector2::new(px, py);
        let m1: Matrix2<FBig<HalfEven>> = matrix_multiply_2x2(prec, &d1, &d2);
        let m: Matrix2<FBig<HalfEven>> = matrix_multiply_2x2(prec, &m1, &d3);
        let ellipse = Ellipse::new(m, p, prec);

        Self {
            scale,
            d,
            z_x,
            z_y,
            ellipse,
            prec,
        }
    }
}

impl Region for MixedDiagonalRegion {
    fn ellipse(&self) -> Ellipse {
        self.ellipse.clone()
    }

    fn inside(&self, u: &DOmega) -> bool {
        let prec = self.prec;
        let cos_term1 = &self.z_x * u.real(prec);
        let cos_term2 = &self.z_y * u.imag(prec);
        let cos_similarity = &cos_term1 + &cos_term2;

        DRootTwo::from_domega(u.conj() * u) <= DRootTwo::from_zroottwo(self.scale.clone())
            && cos_similarity >= self.d
    }

    fn intersect(&self, u0: &DOmega, v: &DOmega) -> Option<(FBig<HalfEven>, FBig<HalfEven>)> {
        let prec = self.prec;
        let a = v.conj() * v;
        let b = 2 * (v.conj() * u0);
        let c = u0.conj() * u0 - DOmega::from_zroottwo(&self.scale);
        let vz_term1 = &self.z_x * v.real(prec);
        let vz_term2 = &self.z_y * v.imag(prec);
        let vz = &vz_term1 + &vz_term2;

        let term1 = &self.z_x * u0.real(prec);
        let term2 = &self.z_y * u0.imag(prec);
        let temp_sub = &self.d - &term1;
        let rhs = &temp_sub - &term2;
        // t0 <= t1
        let (t0, t1) = solve_quadratic(prec, a.real(prec), b.real(prec), c.real(prec))?;
        let zero = prec.ib(IBig::ZERO);

        if vz > zero {
            let t2 = &rhs / &vz;
            Some(if t0 > t2 { (t0, t1) } else { (t2, t1) })
        } else if vz < zero {
            let t2 = &rhs / &vz;
            Some(if t1 < t2 { (t0, t1) } else { (t0, t2) })
        } else if rhs <= zero {
            Some((t0, t1))
        } else {
            None
        }
    }
}

/// Result of [`search_for_straddling_pair`].
#[derive(Debug)]
pub(crate) enum StraddleOutcome {
    /// A candidate with `Im(w) == 0` was found: it IS the exact target direction, so no
    /// mixing is needed (or possible).
    Unmixed(DOmegaUnitary),
    /// `(lo, hi)`: `Im(w_lo) <= 0 <= Im(w_hi)`, i.e. under- and over-rotation respectively.
    /// `hi` is boxed purely to keep this enum's variants closer in size (clippy
    /// `large_enum_variant`); there is no semantic difference from an unboxed field.
    Mixed(DOmegaUnitary, Box<DOmegaUnitary>),
    /// Exceeded `max_k` without completing a pair. This should not happen for well-formed
    /// inputs; see [`crate::gridsynth::search_for_solution`] for why.
    NotFound,
}

/// Single-enumeration search for a straddling pair of candidates (one under-rotating, one
/// over-rotating the target direction), partitioned by the sign of `Im(w)` as the `k`-loop's
/// lazy candidate stream is consumed -- deliberately not two separate calls to
/// `search_for_solution`, which would duplicate the (expensive) `solve_tdgp` enumeration.
///
/// Generic over the region type (`A: Region + Debug`, matching the bound
/// `crate::gridsynth`'s own generic driver functions use) rather than hardcoded to
/// `MixedDiagonalRegion`, so a sibling region shape (e.g. the annulus sector used by "mixed
/// fallback") can reuse this same search unchanged -- only the region's `inside`/`intersect`
/// predicates differ; the bucket-by-sign-of-`Im(w)` logic here is region-agnostic.
///
/// `phase_tolerance` bounds how far off-angle (in `|Im(w)|`) an exact-ring-unitary candidate
/// (`|z| == 1`, see below) may be before the "no mixing needed" fast path is allowed to claim
/// it as the final answer -- callers should pass the same spec epsilon used to build `region`.
/// See the fast path's inline comment for why this check exists.
pub(crate) fn search_for_straddling_pair<A: Region + std::fmt::Debug>(
    region: &A,
    unit_disk: &UnitDisk,
    transformed: &UprightTransform,
    config: &mut GridSynthConfig,
    wframe: &WFrame,
    phase_tolerance: &FBig<HalfEven>,
) -> StraddleOutcome {
    // See `gridsynth::search_for_solution` for the rationale behind this bound: it is
    // effectively unreachable for well-formed inputs and exists only to fail loudly (rather
    // than hang) if a `Region` predicate is broken.
    let max_k = 4 * config.prec.bits() as i64;
    let zero = config.prec.ib(IBig::ZERO);

    let mut lo: Option<DOmegaUnitary> = None;
    let mut hi: Option<DOmegaUnitary> = None;

    let mut k = 0;
    while k <= max_k {
        let solutions = solve_tdgp(
            region,
            unit_disk,
            &transformed.op_g,
            &transformed.bbox_a,
            &transformed.bbox_b,
            k,
            config.verbose,
        );

        if let Some(solutions) = solutions {
            for z in solutions {
                if (&z * z.conj()).residue() == 0 {
                    continue;
                }

                let xi = DRootTwo::from_int(IBig::ONE) - DRootTwo::from_domega(z.conj() * &z);

                // Floating-point `im_w` is used both to bucket non-exact candidates into the
                // under-/over-rotation slots below, and (see immediately below) to gate the
                // ring-exactness fast path. A landed-on-the-boundary `im == 0` for bucketing
                // purposes is just rounding noise, so it is folded into the `lo` (`<= 0`)
                // bucket, which still satisfies `mixture_weight`'s `im_lo <= 0 <= im_hi`
                // precondition.
                let im = wframe.im_w(&z);

                // Ring-exact zero-*synthesis*-error check: `xi == 1 - |z|^2` is computed
                // purely from exact ring arithmetic on `z`, so it is genuinely, bit-exactly
                // zero whenever `|z| == 1`, i.e. `z` is already a unitary and needs no `w`
                // correction. This is NOT, by itself, "z is the target direction": `z` only
                // has to satisfy the *search region's* (at loose `epsilon` values, sometimes
                // wide) containment check, not the tight target tolerance. So an exact-ring
                // point can land inside the region while still being measurably off-angle
                // (see issue #8) -- hence the additional `im.abs() <= *phase_tolerance` gate
                // below, using floating `im_w` (the only place that measures the *angle*, as
                // opposed to just the magnitude `xi`) to confirm the phase genuinely matches
                // before short-circuiting on it. `im_w` mixes two independently rounded
                // floating approximations of the same irrational value (`Prec::cos`/`Prec::sin`
                // for `z_x`/`z_y` vs. `Prec::sqrt2`'s Newton iteration inside
                // `u.real()`/`u.imag()`), so it is not bit-exact even for a genuine exact
                // match -- but that rounding noise is many orders of magnitude below
                // `phase_tolerance` (working precision scales with `-log(epsilon)`), while a
                // genuinely off-angle candidate that merely passed the region's wide
                // containment check sits well above it, so the threshold cleanly separates the
                // two cases without needing bit-exact equality.
                if xi.to_real(config.prec) == zero && im.clone().abs() <= *phase_tolerance {
                    if let Some(w_val) =
                        diophantine_dyadic(xi.clone(), &mut config.diophantine_data)
                    {
                        return StraddleOutcome::Unmixed(process_solution_candidate(
                            z,
                            w_val,
                            PhaseMode::Exact,
                        ));
                    }
                    // Falls through to the ordinary lo/hi bucketing below in the (should not
                    // normally happen) case that the trivial xi=0 diophantine solve fails.
                }

                if im <= zero {
                    if lo.is_some() {
                        // Slot already filled: don't waste a factoring attempt on it.
                        continue;
                    }
                    if let Some(w_val) = diophantine_dyadic(xi, &mut config.diophantine_data) {
                        lo = Some(process_solution_candidate(z, w_val, PhaseMode::Exact));
                        if let Some(hi_val) = hi {
                            return StraddleOutcome::Mixed(lo.unwrap(), Box::new(hi_val));
                        }
                    }
                } else {
                    if hi.is_some() {
                        continue;
                    }
                    if let Some(w_val) = diophantine_dyadic(xi, &mut config.diophantine_data) {
                        hi = Some(process_solution_candidate(z, w_val, PhaseMode::Exact));
                        if let Some(lo_val) = lo {
                            return StraddleOutcome::Mixed(lo_val, Box::new(hi.unwrap()));
                        }
                    }
                }
            }
        }
        k += 1;
    }

    match (lo, hi) {
        (Some(l), Some(h)) => StraddleOutcome::Mixed(l, Box::new(h)),
        _ => StraddleOutcome::NotFound,
    }
}

/// The four `{I, S, Z, SZ}`-twirl variants of a solved `DOmegaUnitary`: conjugating
/// `U = [[z, ...], [w, ...]]` by a diagonal Clifford `diag(1, e^{i*beta})` for `beta in
/// {0, pi/2, pi, 3pi/2}` leaves `z` (the top-left entry, which is all that matters for the
/// diagonal-rotation approximation) exactly unchanged, and multiplies `w` by `e^{i*beta}`.
/// Since `omega^2` is a primitive 4th root of unity (regardless of this crate's `omega` sign
/// convention), `w.mul_by_omega_power(2*m)` for `m in 0..4` realizes all four phases on `w`
/// as an unordered set.
/// Kept as the canonical, directly-from-the-matrix definition of the twirl even though
/// production code (`assemble_result`, via [`twirl_variant_gates`]) no longer calls it --
/// [`relabel_by_twirl`] is a *derived* shortcut, and this function is what
/// `relabel_by_twirl_matches_independent_decomposition` checks it against. Hence
/// `#[allow(dead_code)]` rather than deleting it: it is dead in the non-test build only, not
/// unused.
#[allow(dead_code)]
pub(crate) fn twirl_variants(u: &DOmegaUnitary) -> [DOmegaUnitary; 4] {
    std::array::from_fn(|m| {
        let twirled_w = u.w().mul_by_omega_power(2 * m);
        DOmegaUnitary::new(u.z().clone(), twirled_w, u.n() as usize, Some(u.k()))
    })
}

/// The `{I, S, Z, SZ}` diagonal-Clifford twirl group used by the mixed protocols: `S^m` for
/// `m in 0..4`. Not a `const`/`static` because `Clifford::new` normalizes its arguments (cheap,
/// but not a `const fn`).
///
/// An implementer with only fair coins draws two bits `(b1, b0)` and conjugates by
/// `twirl_cliffords()[2*b1 + b0]`.
pub fn twirl_cliffords() -> [Clifford; 4] {
    std::array::from_fn(|m| Clifford::new(0, 0, m as i32, 0))
}

/// The output of [`synth_mixed_diagonal`]: a classical probabilistic-channel approximation of
/// `R_z(theta)`. Call [`AchievedDiamondError::achieved_diamond_error`] to compute the achieved
/// projective-step diamond-norm error on demand.
///
/// The two variants spell out exactly what randomness a caller needs to sample this channel:
/// `Exact` needs none, `Mixed` needs one biased coin and two fair coins.
#[derive(Debug, Clone)]
pub enum MixedDiagonalResult {
    /// The target direction was ring-exactly representable (e.g. `theta` a multiple of
    /// `pi/2`): one gate word suffices, with zero synthesis error and no randomness at all.
    Exact { gates: GateSeq, prec: Prec },
    /// The general case. Sampling this channel, once per use:
    ///   1. flip one **biased** coin: `lo` with probability `p`, else `hi`;
    ///   2. flip two **fair** coins to get `m = 2*b1 + b0 in 0..4`;
    ///   3. run [`conjugate_by_clifford`]`(chosen_side, twirl_cliffords()[m])` (or, equivalently,
    ///      the same convenience via [`MixedDiagonalResult::gates_for`]).
    ///
    /// The twirl changes neither the decoded top-left entry `z` (hence not the rotation error)
    /// nor the T-count (conjugation by a Clifford preserves both) -- see
    /// `mixed_diagonal_twirl_preserves_z_and_t_count` -- so `lo`/`hi` are stored untwirled, and
    /// error/cost computations never need to materialize a conjugate.
    Mixed {
        /// `P(lo)`. Strictly between 0 and 1 -- a degenerate mixture (`p` exactly 0 or 1)
        /// collapses to `Exact` instead (see [`assemble_result`]).
        p: FBig<HalfEven>,
        /// The under-rotation side's untwirled gate word.
        lo: GateSeq,
        /// The over-rotation side's untwirled gate word.
        hi: GateSeq,
        /// The working precision this result was synthesized at.
        prec: Prec,
    },
}

impl MixedDiagonalResult {
    /// The working precision this result was synthesized at.
    fn prec(&self) -> Prec {
        match self {
            MixedDiagonalResult::Exact { prec, .. } => *prec,
            MixedDiagonalResult::Mixed { prec, .. } => *prec,
        }
    }

    /// The word to actually run, given the two draws described on [`MixedDiagonalResult::Mixed`].
    /// `Exact` ignores both arguments. `twirl` is taken mod 4 (any of the 4 fair-coin encodings
    /// of `m` works).
    pub fn gates_for(&self, take_lo: bool, twirl: usize) -> GateSeq {
        match self {
            MixedDiagonalResult::Exact { gates, .. } => gates.clone(),
            MixedDiagonalResult::Mixed { lo, hi, .. } => {
                let side = if take_lo { lo } else { hi };
                let m = twirl % 4;
                if m == 0 {
                    side.clone()
                } else {
                    conjugate_by_clifford(side, twirl_cliffords()[m])
                }
            }
        }
    }

    /// Flat categorical view: every `(weight, gate word)` pair, weights summing to 1 --
    /// `Exact` yields one pair at weight 1; `Mixed` yields 8 (four of `lo` at `p/4` each, four
    /// of `hi` at `(1-p)/4` each), materializing each twirl conjugate via
    /// [`conjugate_by_clifford`].
    ///
    /// This is the *derived* view, for error/cost math and verification -- it does the
    /// conjugation work `Mixed`'s own docs say is unnecessary for those computations, so
    /// prefer `Mixed`'s fields directly, or [`MixedDiagonalResult::expected_t_count`], when
    /// only the error or the average cost is needed.
    pub fn weighted_branches(&self) -> Vec<(FBig<HalfEven>, GateSeq)> {
        match self {
            MixedDiagonalResult::Exact { gates, prec } => {
                vec![(prec.ib(IBig::ONE), gates.clone())]
            }
            MixedDiagonalResult::Mixed { p, prec, .. } => {
                let one = prec.ib(IBig::ONE);
                let four = prec.fb(FBig::try_from(4.0).unwrap());
                let one_minus_p = &one - p;
                let p_over_4 = p / &four;
                let one_minus_p_over_4 = &one_minus_p / &four;

                let mut branches = Vec::with_capacity(8);
                for m in 0..4 {
                    branches.push((p_over_4.clone(), self.gates_for(true, m)));
                }
                for m in 0..4 {
                    branches.push((one_minus_p_over_4.clone(), self.gates_for(false, m)));
                }
                branches
            }
        }
    }

    /// Weight-averaged T-count -- the protocol's mean cost. `Exact` is that one word's T-count;
    /// `Mixed` is `p*t_count(lo) + (1-p)*t_count(hi)`, since a twirl preserves T-count (no
    /// conjugation needed to compute this).
    pub fn expected_t_count(&self) -> FBig<HalfEven> {
        match self {
            MixedDiagonalResult::Exact { gates, prec } => prec.ib(IBig::from(gates.t_count())),
            MixedDiagonalResult::Mixed { p, lo, hi, prec } => {
                let one = prec.ib(IBig::ONE);
                let one_minus_p = &one - p;
                let lo_t = prec.ib(IBig::from(lo.t_count()));
                let hi_t = prec.ib(IBig::from(hi.t_count()));
                (p * &lo_t) + (&one_minus_p * &hi_t)
            }
        }
    }

    /// Recomputes the achieved diamond-norm error to the target direction encoded by `wframe`,
    /// directly from the public `lo`/`hi` gate strings (no conjugation needed -- see `Mixed`'s
    /// own docs on why a twirl doesn't change `z`). [`AchievedDiamondError::achieved_diamond_error`]
    /// is a thin wrapper over this that builds `wframe` from a raw `theta` via [`WFrame::new`]
    /// -- this lower-level entry point exists so a caller that already has a target direction
    /// as a `(cos, sin)` half-angle pair (e.g. a fallback correction's residual angle) can
    /// reuse the exact same mixture-aware computation without a lossy angle round-trip.
    pub(crate) fn achieved_diamond_error_with_frame(&self, wframe: &WFrame) -> FBig<HalfEven> {
        let prec = self.prec();
        match self {
            MixedDiagonalResult::Exact { gates, .. } => {
                let u = DOmegaUnitary::from_gates(gates);
                let re_w = wframe.re_w(u.z());
                diagonal_diamond_distance(prec, &re_w)
            }
            MixedDiagonalResult::Mixed { lo, hi, .. } => {
                let lo_u = DOmegaUnitary::from_gates(lo);
                let hi_u = DOmegaUnitary::from_gates(hi);
                let re_lo = wframe.re_w(lo_u.z());
                let im_lo = wframe.im_w(lo_u.z());
                let re_hi = wframe.re_w(hi_u.z());
                let im_hi = wframe.im_w(hi_u.z());

                mixture_weight(prec, (&re_lo, &im_lo), (&re_hi, &im_hi))
                    .expect("a real assembled Mixed result must yield a valid mixture")
                    .projective_diamond_error
            }
        }
    }
}

impl AchievedDiamondError for MixedDiagonalResult {
    fn achieved_diamond_error(&self, theta: &FBig<HalfEven>) -> FBig<HalfEven> {
        self.achieved_diamond_error_with_frame(&WFrame::new(self.prec(), theta))
    }
}

/// Turns a [`StraddleOutcome`] into the final [`MixedDiagonalResult`].
///
/// For the `Unmixed` (exact-angle) case, this emits `Exact`: twirling an exact solution (zero
/// rotation error) is harmless but has no error-cancellation purpose, so the untwirled form is
/// the simpler, equally-correct choice.
///
/// For the `Mixed` case, [`mixture_weight`] can itself return `p` exactly 0 or exactly 1 when
/// one side is already an exact solution ("Degenerate exact solutions" in its own docs) -- in
/// that case this also collapses to `Exact` on the surviving side, rather than emitting a
/// `Mixed` result with a side that can never be sampled. Otherwise this emits `Mixed { p, lo,
/// hi }`; the classical mixture is invariant under an additional random `{Z,S}` twirl on
/// whichever side gets sampled -- required for the mixture to implement a genuine
/// depolarizing-style probabilistic channel rather than leak phase information -- but nothing
/// is precomputed for it (see [`MixedDiagonalResult::gates_for`]).
pub(crate) fn assemble_result(
    prec: Prec,
    outcome: StraddleOutcome,
    wframe: &WFrame,
) -> MixedDiagonalResult {
    match outcome {
        StraddleOutcome::NotFound => panic!(
            "search_for_straddling_pair: exceeded max_k without finding a straddling pair \
             (or an exact solution) -- region predicate is likely incorrect"
        ),
        StraddleOutcome::Unmixed(u) => MixedDiagonalResult::Exact {
            gates: decompose_domega_unitary(u),
            prec,
        },
        StraddleOutcome::Mixed(lo, hi) => {
            let hi = *hi;
            let re_lo = wframe.re_w(lo.z());
            let im_lo = wframe.im_w(lo.z());
            let re_hi = wframe.re_w(hi.z());
            let im_hi = wframe.im_w(hi.z());

            let mw = mixture_weight(prec, (&re_lo, &im_lo), (&re_hi, &im_hi)).expect(
                "mixture_weight returned None for a real solved straddling pair -- this \
                 indicates a genuine bug (search_for_straddling_pair only ever produces \
                 im_lo <= 0 <= im_hi pairs), not an expected degenerate input",
            );

            let one = prec.ib(IBig::ONE);
            let zero = prec.ib(IBig::ZERO);
            if mw.p == zero {
                return MixedDiagonalResult::Exact {
                    gates: decompose_domega_unitary(hi),
                    prec,
                };
            }
            if mw.p == one {
                return MixedDiagonalResult::Exact {
                    gates: decompose_domega_unitary(lo),
                    prec,
                };
            }

            MixedDiagonalResult::Mixed {
                p: mw.p,
                lo: decompose_domega_unitary(lo),
                hi: decompose_domega_unitary(hi),
                prec,
            }
        }
    }
}

/// Synthesizes a mixed-diagonal probabilistic-channel approximation of `R_z(theta)` to
/// diamond-norm accuracy `epsilon_diamond`.
///
/// `epsilon_diamond` is converted to this crate's operator-norm-style `epsilon` convention
/// (via [`diamond_to_spec_epsilon`]) before building the search region. Only exact-phase
/// synthesis (`PhaseMode::Exact`) is implemented at this stage; `up_to_phase` mixing is out
/// of scope.
///
/// # Panics
/// Panics if the internal search exceeds its (very generous) bound on `k` without finding a
/// solution; see [`search_for_straddling_pair`]. Not expected to trigger for any well-formed
/// input.
pub fn synth_mixed_diagonal(
    theta: f64,
    epsilon_diamond: f64,
    seed: u64,
    verbose: bool,
) -> MixedDiagonalResult {
    // `config_from_theta_epsilon` is reused purely as a scaffold: it parses `theta` exactly,
    // and sizes working precision from the decimal magnitude of its `epsilon` argument, which
    // is close enough to the actual (post-conversion) spec epsilon for that purpose. The
    // *value* stored in `config.epsilon` here is still the diamond-norm epsilon; the spec
    // epsilon actually used to build the region is derived from it just below.
    let mut config = config_from_theta_epsilon(theta, epsilon_diamond, seed, verbose, false);
    let prec = config.prec;
    let epsilon_spec = diamond_to_spec_epsilon(prec, &config.epsilon);

    let scale = ZRootTwo::new(IBig::from(1), IBig::from(0));
    let region = MixedDiagonalRegion::new(prec, &config.theta, &epsilon_spec, scale.clone());
    let unit_disk = UnitDisk::new(prec, scale);
    let wframe = WFrame::new(prec, &config.theta);

    let transformed =
        setup_regions_and_transform(&region, &unit_disk, config.verbose, config.measure_time);

    let outcome = search_for_straddling_pair(
        &region,
        &unit_disk,
        &transformed,
        &mut config,
        &wframe,
        &epsilon_spec,
    );

    assemble_result(prec, outcome, &wframe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashu_base::Approximation;
    use dashu_int::ops::Abs;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::f64::consts::PI;

    /// Fixed precision for tests that build values without going through
    /// `config_from_theta_epsilon` (which carries its own derived `Prec`).
    const PREC: Prec = Prec(1000);

    fn to_fbig(prec: Prec, x: f64) -> FBig<HalfEven> {
        FBig::<HalfEven>::try_from(x)
            .unwrap()
            .with_precision(prec.bits())
            .value()
    }

    fn fbig_to_f64(x: &FBig<HalfEven>) -> f64 {
        match x.to_f64() {
            Approximation::Inexact(v, _) => v,
            Approximation::Exact(v) => v,
        }
    }

    /// A tolerance (in bits) safely below the *actual* working precision configured by
    /// `config_from_theta_epsilon` for a given epsilon (which allocates working precision
    /// roughly proportional to epsilon's decimal digit count -- e.g. only ~60 bits for
    /// epsilon=1e-5, not this crate's default 1000-bit `PREC_BITS_INITIAL`). Using a fixed,
    /// generous tolerance like 200 bits regardless of the actual precision would spuriously
    /// fail whenever the configured working precision is lower than that, even though the
    /// underlying arithmetic identity holds exactly up to genuine rounding.
    fn safe_tol_bits(prec: Prec) -> usize {
        prec.bits().saturating_sub(30)
    }

    fn approx_eq(a: &FBig<HalfEven>, b: &FBig<HalfEven>, tol_bits: usize) -> bool {
        let diff = (a - b).abs();
        let tol = PREC.ib(IBig::ONE) / PREC.ib(IBig::ONE << tol_bits);
        diff <= tol
    }

    /// Builds a `(MixedDiagonalRegion, UnitDisk, UprightTransform, WFrame, GridSynthConfig)`
    /// tuple for a given `(theta, epsilon)` pair, mirroring `synth_mixed_diagonal`'s setup but
    /// exposing the intermediate pieces for tests that need to inspect the raw search.
    fn setup(
        theta_f64: f64,
        epsilon: f64,
        seed: u64,
    ) -> (
        MixedDiagonalRegion,
        UnitDisk,
        UprightTransform,
        WFrame,
        GridSynthConfig,
    ) {
        let config = config_from_theta_epsilon(theta_f64, epsilon, seed, false, false);
        let prec = config.prec;
        let scale = ZRootTwo::new(IBig::from(1), IBig::from(0));
        let region = MixedDiagonalRegion::new(prec, &config.theta, &config.epsilon, scale.clone());
        let unit_disk = UnitDisk::new(prec, scale);
        let wframe = WFrame::new(prec, &config.theta);
        let transformed =
            setup_regions_and_transform(&region, &unit_disk, config.verbose, config.measure_time);
        (region, unit_disk, transformed, wframe, config)
    }

    // ---- Task 1: ellipse containment ----

    // Samples points on the TRUE boundary of the circular cap (the two chord endpoints where
    // the line `Re(w) = d` meets the circle `|u| = sqrt(s)`, and the midpoint of the outer arc
    // between them) and confirms they lie inside the bounding ellipse. All three points are,
    // by construction (see module docs), exactly on the ellipse boundary in exact arithmetic:
    // the chord endpoints sit at (radial=0, tangential=c) in the (z_x,z_y)-aligned frame, and
    // the arc midpoint sits at (radial=h, tangential=0), each giving quadratic-form value
    // exactly 1. This is precisely the check that would catch an asymptotic-height bug (using
    // eps^2/8 instead of the exact `sqrt(s) - d`), since that would move the arc-midpoint
    // radial coordinate away from `h` and push its quadratic-form value above 1.
    #[test]
    fn mixed_diagonal_region_boundary_points_inside_ellipse() {
        let theta = to_fbig(PREC, 0.7);
        let epsilon = to_fbig(PREC, 0.3);
        let scale = ZRootTwo::from_int(IBig::from(1));

        let region = MixedDiagonalRegion::new(PREC, &theta, &epsilon, scale.clone());

        let two = to_fbig(PREC, 2.0);
        let half_eps = &epsilon / &two;
        let scale_to_real = scale.to_real(PREC);
        let sqrt_s = scale_to_real.sqrt();
        let c = (&scale_to_real * &half_eps).sqrt();

        let d = region.d.clone();
        let z_x = region.z_x.clone();
        let z_y = region.z_y.clone();

        // Chord endpoints: (z_x*d -/+ z_y*c, z_y*d +/- z_x*c).
        let ep_plus = Vector2::new((&z_x * &d) - (&z_y * &c), (&z_y * &d) + (&z_x * &c));
        let ep_minus = Vector2::new((&z_x * &d) + (&z_y * &c), (&z_y * &d) - (&z_x * &c));
        // Outer-arc midpoint: sqrt(s) * (z_x, z_y).
        let cap_top = Vector2::new(&sqrt_s * &z_x, &sqrt_s * &z_y);

        let ellipse = region.ellipse();

        // These three points are, in *exact* arithmetic, precisely ON the ellipse boundary
        // (quadratic-form value exactly 1 -- see the doc comment above): the chord endpoints
        // sit at (radial=0, tangential=c) in the (z_x,z_y)-aligned frame, and the arc midpoint
        // sits at (radial=h, tangential=0). At working precision, independently-rounded sqrt
        // evaluations (for `d`/`c`/`sqrt_s`, computed once inside the region constructor and
        // once more here) can round the value to a few ULP above 1 rather than landing exactly
        // on it, purely as floating-point noise unrelated to region correctness. To avoid that
        // false negative while still exercising the true boundary geometry, nudge each point
        // a minuscule (1e-9 relative) amount towards the ellipse center before testing
        // containment -- utterly negligible next to the kind of gross error (e.g. an
        // asymptotic-height formula off by a factor of ~2) this test is meant to catch.
        let center = ellipse.p.clone();
        let shrink = &PREC.ib(IBig::ONE) - &to_fbig(PREC, 1e-9);
        let nudge_towards_center = |pt: &Vector2<FBig<HalfEven>>| -> Vector2<FBig<HalfEven>> {
            Vector2::new(
                &center[0] + (&shrink * (&pt[0] - &center[0])),
                &center[1] + (&shrink * (&pt[1] - &center[1])),
            )
        };
        let ep_plus = nudge_towards_center(&ep_plus);
        let ep_minus = nudge_towards_center(&ep_minus);
        let cap_top = nudge_towards_center(&cap_top);

        assert!(
            ellipse.inside(&ep_plus),
            "chord endpoint (+) not inside bounding ellipse"
        );
        assert!(
            ellipse.inside(&ep_minus),
            "chord endpoint (-) not inside bounding ellipse"
        );
        assert!(
            ellipse.inside(&cap_top),
            "outer-arc midpoint not inside bounding ellipse"
        );

        // Sanity: a point diametrically opposite the cap must NOT be inside.
        let far_side = Vector2::new(-(&sqrt_s * &z_x), -(&sqrt_s * &z_y));
        assert!(
            !ellipse.inside(&far_side),
            "diametrically opposite point should not be inside"
        );
    }

    // ---- Task 2: single-enumeration straddling-pair search ----

    #[test]
    fn straddling_search_finds_mixed_pair_for_generic_angles() {
        for k in [1i64, 3, 5, 7] {
            let theta_f64 = k as f64 * PI / 32.0;
            let epsilon = 1e-6;
            let (region, unit_disk, transformed, wframe, mut config) =
                setup(theta_f64, epsilon, 42 + k as u64);

            let phase_tolerance = config.epsilon.clone();
            let outcome = search_for_straddling_pair(
                &region,
                &unit_disk,
                &transformed,
                &mut config,
                &wframe,
                &phase_tolerance,
            );
            match outcome {
                StraddleOutcome::Mixed(_, _) => {}
                other => panic!("expected Mixed for theta={theta_f64} (k={k}), got {other:?}"),
            }
        }
    }

    // NOTE on angle choice: the target direction `u = e^{-i*theta/2}` is exactly
    // representable in the ring Z[omega] (omega = e^{i*pi/4}) iff `theta` is a multiple of
    // pi/2 -- e.g. theta=pi/2 gives u=e^{-i*pi/4}=-omega^3 exactly, verified against this
    // crate's own `pi_over_two_test` fixture, which produces the identical trivial 8-gate
    // string "SWWWWWWW" regardless of epsilon or seed (proof of a zero-error, exact
    // solution). theta=pi/4 does NOT have this property (its target direction e^{-i*pi/8}
    // is not an omega power): `pi_over_4_exact_test`'s "exact" in its name refers to
    // `PhaseMode::Exact` (vs. `Shifted`), not to zero rotation error -- that test's long,
    // epsilon-dependent gate string is itself evidence there is no exact solution at
    // theta=pi/4. So this test (and `degenerate_angles_produce_unmixed_result` below) use
    // theta in {pi/2, pi} (both multiples of pi/2) rather than the {pi/2, pi/4} the original
    // task spec suggested -- pi/4 genuinely belongs in the "Mixed" bucket, not "Unmixed".
    #[test]
    fn straddling_search_finds_unmixed_for_exact_angles() {
        for theta_f64 in [PI / 2.0, PI] {
            let epsilon = 1e-6;
            let (region, unit_disk, transformed, wframe, mut config) = setup(theta_f64, epsilon, 7);

            let phase_tolerance = config.epsilon.clone();
            let outcome = search_for_straddling_pair(
                &region,
                &unit_disk,
                &transformed,
                &mut config,
                &wframe,
                &phase_tolerance,
            );
            match outcome {
                StraddleOutcome::Unmixed(_) => {}
                other => panic!("expected Unmixed for theta={theta_f64}, got {other:?}"),
            }
        }
    }

    // ---- Task 3: twirl construction ----

    #[test]
    fn twirl_variants_preserve_top_left_entry_exactly() {
        let theta_f64 = 3.0 * PI / 32.0;
        let epsilon = 1e-6;
        let (region, unit_disk, transformed, wframe, mut config) = setup(theta_f64, epsilon, 99);

        let phase_tolerance = config.epsilon.clone();
        let outcome = search_for_straddling_pair(
            &region,
            &unit_disk,
            &transformed,
            &mut config,
            &wframe,
            &phase_tolerance,
        );
        let (lo, hi) = match outcome {
            StraddleOutcome::Mixed(lo, hi) => (lo, *hi),
            other => panic!("expected Mixed, got {other:?}"),
        };

        for original in [lo, hi] {
            let orig_top_left = original.to_complex_matrix(PREC)[(0, 0)].clone();
            for variant in twirl_variants(&original) {
                let variant_top_left = variant.to_complex_matrix(PREC)[(0, 0)].clone();
                assert_eq!(
                    variant_top_left, orig_top_left,
                    "twirl variant changed the (0,0) entry"
                );

                let gates = decompose_domega_unitary(variant);
                let reconstructed_top_left =
                    DOmegaUnitary::from_gates(&gates).to_complex_matrix(PREC)[(0, 0)].clone();
                assert_eq!(
                    reconstructed_top_left, orig_top_left,
                    "re-multiplying the decomposed gate string changed the (0,0) entry"
                );
            }
        }
    }

    /// The load-bearing correctness proof for [`conjugate_by_clifford`] as used via
    /// [`twirl_cliffords`], run *before* it is ever used in `assemble_result`/`gates_for`
    /// below: for real solved candidates (from a genuine `search_for_straddling_pair` call,
    /// not synthetic numbers), every one of its three nontrivial outputs (`m = 1, 2, 3`) must
    /// be byte-for-byte identical to independently calling `decompose_domega_unitary` on
    /// `twirl_variants(u)[m]` -- the expensive path this function replaces. Also pins the
    /// T-count invariant (`MixedDiagonalResult::expected_t_count`'s justification for never
    /// conjugating: if this ever regresses it fails loudest here first) across several
    /// `(theta, epsilon)` pairs spanning coarse to fine tolerances.
    #[test]
    fn conjugate_by_clifford_matches_independent_decomposition() {
        let cases: [(f64, f64, u64); 4] = [
            (3.0 * PI / 32.0, 1e-6, 99),
            (0.7, 1e-4, 7),
            (2.222, 1e-8, 13),
            (5.9, 1e-10, 21),
        ];
        for (theta_f64, epsilon, seed) in cases {
            let (region, unit_disk, transformed, wframe, mut config) =
                setup(theta_f64, epsilon, seed);
            let phase_tolerance = config.epsilon.clone();
            let outcome = search_for_straddling_pair(
                &region,
                &unit_disk,
                &transformed,
                &mut config,
                &wframe,
                &phase_tolerance,
            );
            let (lo, hi) = match outcome {
                StraddleOutcome::Mixed(lo, hi) => (lo, *hi),
                StraddleOutcome::Unmixed(_) => continue, // no twirl to check at this tolerance
                other => panic!("expected Mixed or Unmixed, got {other:?}"),
            };

            let twirls = twirl_cliffords();
            for original in [lo, hi] {
                let base_gates = decompose_domega_unitary(original.clone());
                let variants = twirl_variants(&original);
                for (m, (variant, &c)) in variants.iter().zip(twirls.iter()).enumerate() {
                    let expected = decompose_domega_unitary(variant.clone());
                    let actual = if m == 0 {
                        base_gates.clone()
                    } else {
                        conjugate_by_clifford(&base_gates, c)
                    };
                    assert_eq!(
                        actual, expected,
                        "theta={theta_f64} eps={epsilon} seed={seed} m={m}: conjugate_by_clifford \
                         disagrees with independent decomposition\n  actual:   {actual}\n  \
                         expected: {expected}"
                    );
                    assert_eq!(
                        actual.t_count(),
                        base_gates.t_count(),
                        "theta={theta_f64} eps={epsilon} seed={seed} m={m}: twirl changed T-count"
                    );
                }
            }
        }
    }

    /// The invariant [`MixedDiagonalResult::achieved_diamond_error_with_frame`] and
    /// [`MixedDiagonalResult::expected_t_count`] rely on to never materialize a twirl
    /// conjugate: for a genuine `Mixed` result, conjugating either side by any of
    /// [`twirl_cliffords`] leaves both the decoded top-left entry `z` and the T-count
    /// unchanged.
    #[test]
    fn mixed_diagonal_twirl_preserves_z_and_t_count() {
        let theta_f64 = 3.0 * PI / 32.0;
        let epsilon = 1e-5;
        let result = synth_mixed_diagonal(theta_f64, epsilon, 321, false);
        let prec = result.prec();
        let MixedDiagonalResult::Mixed { lo, hi, .. } = &result else {
            panic!("expected Mixed at this (theta, epsilon)");
        };

        for side in [lo, hi] {
            let orig_z = DOmegaUnitary::from_gates(side).to_complex_matrix(prec)[(0, 0)].clone();
            let orig_t = side.t_count();
            for &c in &twirl_cliffords() {
                let conjugated = conjugate_by_clifford(side, c);
                assert_eq!(
                    conjugated.t_count(),
                    orig_t,
                    "twirl {c} changed T-count for side {side}"
                );
                let z = DOmegaUnitary::from_gates(&conjugated).to_complex_matrix(prec)[(0, 0)]
                    .clone();
                assert_eq!(
                    z, orig_z,
                    "twirl {c} changed the (0,0) entry for side {side}"
                );
            }
        }
    }

    /// Pins [`assemble_result`]'s degenerate collapse: `mixture_weight` returning `p` exactly 0
    /// or exactly 1 (one side already exact) must produce `Exact`, not a `Mixed` with a
    /// zero-probability side. For a genuine (non-degenerate) `Mixed` result, `p` must be
    /// strictly interior.
    #[test]
    fn mixed_diagonal_p_is_strictly_interior_or_collapses_to_exact() {
        let theta_f64 = 3.0 * PI / 32.0;
        let epsilon = 1e-5;
        let result = synth_mixed_diagonal(theta_f64, epsilon, 321, false);
        let prec = result.prec();
        match result {
            MixedDiagonalResult::Exact { .. } => {}
            MixedDiagonalResult::Mixed { p, .. } => {
                assert!(
                    p > prec.ib(IBig::ZERO) && p < prec.ib(IBig::ONE),
                    "p={p} is not strictly interior"
                );
            }
        }
    }

    // ---- Task 4: public entry point and validation ----

    // Required test 2: "absolute oracle" -- re-verify, using REAL solved candidates from the
    // search (not synthetic numbers), that `mixture_weight`'s reported error matches its own
    // closed form, and that the mixture strictly improves on each branch's own unmixed
    // diamond-norm error (`crate::protocol::mixing`'s own tests already cover the formula in
    // isolation; this re-checks it end-to-end against this module's real search output).
    #[test]
    fn absolute_oracle_end_to_end_error_matches_closed_form() {
        let theta_f64 = 3.0 * PI / 32.0;
        let epsilon = 1e-5;
        let (region, unit_disk, transformed, wframe, mut config) = setup(theta_f64, epsilon, 55);

        let phase_tolerance = config.epsilon.clone();
        let outcome = search_for_straddling_pair(
            &region,
            &unit_disk,
            &transformed,
            &mut config,
            &wframe,
            &phase_tolerance,
        );
        let (lo, hi) = match outcome {
            StraddleOutcome::Mixed(lo, hi) => (lo, *hi),
            other => panic!("expected Mixed, got {other:?}"),
        };

        let re_lo = wframe.re_w(lo.z());
        let im_lo = wframe.im_w(lo.z());
        let re_hi = wframe.re_w(hi.z());
        let im_hi = wframe.im_w(hi.z());

        assert!(im_lo <= PREC.ib(IBig::ZERO), "lo branch must under-rotate");
        assert!(im_hi >= PREC.ib(IBig::ZERO), "hi branch must over-rotate");

        let prec = config.prec;
        let mw = mixture_weight(prec, (&re_lo, &im_lo), (&re_hi, &im_hi))
            .expect("mixture_weight should succeed for a real straddling pair");

        // Cross-check the closed form directly: error == 2*(p*im_lo^2 + (1-p)*im_hi^2).
        let one = prec.ib(IBig::ONE);
        let two = to_fbig(prec, 2.0);
        let one_minus_p = &one - &mw.p;
        let lo_term = &mw.p * (&im_lo * &im_lo);
        let hi_term = &one_minus_p * (&im_hi * &im_hi);
        let expected_error = &two * (&lo_term + &hi_term);
        assert!(
            approx_eq(
                &mw.projective_diamond_error,
                &expected_error,
                safe_tol_bits(prec)
            ),
            "mixed error {} != closed form {}",
            mw.projective_diamond_error,
            expected_error
        );

        // Mixing must beat each branch's own unmixed diamond error (quadratic vs linear-order
        // residual), using the real solved re_w values.
        let unmixed_lo = crate::protocol::mixing::diagonal_diamond_distance(prec, &re_lo);
        let unmixed_hi = crate::protocol::mixing::diagonal_diamond_distance(prec, &re_hi);
        assert!(mw.projective_diamond_error < unmixed_lo);
        assert!(mw.projective_diamond_error < unmixed_hi);

        // Round-trip both candidates through decompose/from_gates and confirm the (0,0) entry
        // used in the error computation above matches what actually gets synthesized.
        let gates_lo = decompose_domega_unitary(lo.clone());
        let reconstructed_lo = DOmegaUnitary::from_gates(&gates_lo).to_complex_matrix(prec);
        assert_eq!(reconstructed_lo[(0, 0)], lo.to_complex_matrix(prec)[(0, 0)]);
        let gates_hi = decompose_domega_unitary(hi.clone());
        let reconstructed_hi = DOmegaUnitary::from_gates(&gates_hi).to_complex_matrix(prec);
        assert_eq!(reconstructed_hi[(0, 0)], hi.to_complex_matrix(prec)[(0, 0)]);
    }

    // Required test 4: branch weights sum to 1.
    #[test]
    fn branch_weights_sum_to_one() {
        for (theta_f64, epsilon) in [(3.0 * PI / 32.0, 1e-5), (PI / 2.0, 1e-5), (PI / 4.0, 1e-5)] {
            let result = synth_mixed_diagonal(theta_f64, epsilon, 321, false);
            let prec = result.prec();
            let mut total = prec.ib(IBig::ZERO);
            for (weight, _gates) in result.weighted_branches() {
                total = &total + &weight;
            }
            assert!(
                approx_eq(&total, &prec.ib(IBig::ONE), safe_tol_bits(prec)),
                "branch weights for theta={theta_f64} summed to {total}, not 1"
            );
        }
    }

    // Required test 5: degenerate angles produce an Exact result. See the note above
    // `straddling_search_finds_unmixed_for_exact_angles` for why pi/2 and pi (both multiples
    // of pi/2, hence ring-exact target directions) are used here rather than the original task
    // spec's {pi/2, pi/4} -- pi/4 has no exact solution and correctly goes through the Mixed
    // path (already exercised by `branch_weights_sum_to_one` above).
    #[test]
    fn degenerate_angles_produce_unmixed_result() {
        for theta_f64 in [PI / 2.0, PI] {
            let result = synth_mixed_diagonal(theta_f64, 1e-6, 654, false);
            let prec = result.prec();
            assert!(
                matches!(result, MixedDiagonalResult::Exact { .. }),
                "expected the Exact form for theta={theta_f64}"
            );
            // `achieved_diamond_error` is computed via `WFrame::re_w`/`diagonal_diamond_distance`,
            // which mixes two independently-rounded floating trig approximations
            // (`Prec::cos`/`Prec::sin`) that do not cancel to bit-exact zero even when the true
            // rotation error is exactly zero (same caveat this module already documents for
            // `im_w`) -- so this must be an approximate, not exact, equality check.
            // `diagonal_diamond_distance`'s `sqrt` roughly halves the number of reliable bits
            // (a `re_w` deviation of `d` near 1 becomes an error of order `sqrt(d)`), so the
            // usual `safe_tol_bits(prec)` is too tight here.
            let theta = to_fbig(prec, theta_f64);
            assert!(approx_eq(
                &result.achieved_diamond_error(&theta),
                &prec.ib(IBig::ZERO),
                safe_tol_bits(prec) / 2
            ));
        }
    }

    // Required test 3: slope fit. Computes the branch-weighted T-count for a batch of random
    // (non-pi/4-multiple) angles at a few epsilons spanning a couple of orders of magnitude,
    // and estimates the cost-vs-log2(1/epsilon) slope. Reported in the test's own assertion
    // message (and by the harness driving this test) rather than hard-gated at exactly 1.52,
    // since a modest sample size is inherently noisy; the assertion instead guards against a
    // gross regression (e.g. a slope near the plain-diagonal baseline's ~3.02, which would
    // indicate the mixing isn't buying anything).
    //
    // T-count is `GateSeq::t_count()`, which counts `Gate::T` occurrences directly -- no
    // string round-trip through `NormalForm` needed.
    #[test]
    fn slope_fit_cost_vs_log2_inv_epsilon() {
        let mut rng = StdRng::seed_from_u64(2024);
        let num_angles = 24;
        let mut angles = Vec::with_capacity(num_angles);
        while angles.len() < num_angles {
            let theta: f64 = rng.random_range(0.0..(2.0 * PI));
            let nearest_pi4_multiple = (theta / (PI / 4.0)).round();
            if (theta - nearest_pi4_multiple * PI / 4.0).abs() > 1e-2 {
                angles.push(theta);
            }
        }

        let epsilons = [1e-4, 1e-6, 1e-8];
        let mut mean_cost = Vec::with_capacity(epsilons.len());

        for (eps_idx, &eps) in epsilons.iter().enumerate() {
            let mut total_cost = 0.0;
            for (i, &theta) in angles.iter().enumerate() {
                let seed = 10_000 + (eps_idx * 1000 + i) as u64;
                let result = synth_mixed_diagonal(theta, eps, seed, false);
                total_cost += fbig_to_f64(&result.expected_t_count());
            }
            mean_cost.push(total_cost / angles.len() as f64);
        }

        let log2_inv_eps: Vec<f64> = epsilons.iter().map(|e| (1.0_f64 / e).log2()).collect();
        let slope = (mean_cost[mean_cost.len() - 1] - mean_cost[0])
            / (log2_inv_eps[log2_inv_eps.len() - 1] - log2_inv_eps[0]);

        eprintln!(
            "mixed-diagonal slope fit: mean_cost={mean_cost:?} at log2(1/eps)={log2_inv_eps:?} \
             -> measured slope = {slope:.4} (target ~1.52, plain-diagonal baseline ~3.02)"
        );

        assert!(
            slope > 0.5 && slope < 2.5,
            "measured slope {slope:.4} is far from the expected ~1.52 (plain-diagonal baseline \
             is ~3.02); mean_cost={mean_cost:?}"
        );
    }
}
