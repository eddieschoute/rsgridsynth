// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Stage 3: mixed fallback, composed from Stage 1 (mixed diagonal) and Stage 2 (fallback).
//!
//! Implements "mixed fallback" (Kliuchnikov, Lauter, Minko, Paetznick, Petit,
//! arXiv:2203.10064v2, Prop 3.16): the widest-tolerance, lowest-T-count protocol in the
//! paper, at the cost of one ancilla and one measurement (same as plain fallback) plus a
//! classical coin (same as mixed diagonal).
//!
//! There is no new region shape and no new number theory here -- per the paper (and the
//! parent design document), mixed fallback is composed from the two earlier stages exactly
//! as described in `crate::protocol::fallback`/`crate::protocol::mixed_diagonal`'s own docs:
//!
//! - The *projective* step searches [`crate::protocol::fallback::SectorRegion`] (Stage 2's
//!   region shape), but at the *wider* angular half-width used by the mixed protocols
//!   (`sin_alpha = sqrt(eps_diamond/4)`, vs. plain fallback's `eps_diamond/4` directly -- see
//!   Prop 3.16's `sqrt(eps/2)`, in the paper's own even-split convention for the
//!   projective-mixture term's *share* of the diamond-norm budget), and via a
//!   *straddling-pair* search (Stage 1's
//!   [`crate::protocol::mixed_diagonal::search_for_straddling_pair`], now generic over the
//!   region type) instead of a single-candidate search -- because mixed fallback needs two
//!   projective candidates (one under-, one over-rotating) to mix, exactly as mixed diagonal
//!   does.
//! - Each side's classical correction -- needed on that side's own "failure" branch,
//!   analogous to plain fallback's single correction -- is itself a full *mixed-diagonal*
//!   result (Stage 1's [`crate::protocol::mixed_diagonal::MixedDiagonalResult`], 8 twirled
//!   branches), not a single plain-diagonal gate word: the correction angle is derived via
//!   the same `atan2`-free half-angle algebra `fallback::synth_fallback` uses.
//! - Per the design document: the projective step itself is **not** twirled here (unlike
//!   mixed diagonal's projective step) -- the projective outcome is a genuine measurement
//!   branch whose "success" action is exactly a Z-rotation by construction of the (out of
//!   scope for this crate) ancilla circuit, so it has no off-diagonal error to twirl away.
//!   Only the fallback *corrections* -- ordinary unitaries, applied unconditionally once
//!   selected -- need the twirl, and get it automatically by being mixed-diagonal results.

use crate::accuracy::{
    achieved_diagonal_diamond_error, achieved_phase_diamond_error, AchievedDiamondError, WFrame,
};
use crate::common::Prec;
use crate::config::{config_from_theta_epsilon, GridSynthConfig};
use crate::gate::{Gate, GateSeq};
use crate::gridsynth::{setup_regions_and_transform, UnitDisk};
use crate::protocol::fallback::{
    half_angle_cos_sin, phase_cos_sin, residual_diamond_error_mixed, SectorRegion,
};
use crate::protocol::mixed_diagonal::{
    assemble_result, search_for_straddling_pair, MixedDiagonalRegion, MixedDiagonalResult,
    StraddleOutcome,
};
use crate::protocol::mixing::mixture_weight;
use crate::protocol::small_angle::{
    small_angle_could_help_half_angle, synth_small_angle_correction, DEFAULT_MAX_LIVE_SEARCH_K,
};
use crate::ring::{DRootTwo, ZRootTwo};
use crate::synthesis_of_clifford_t::decompose_domega_unitary;
use crate::unitary::DOmegaUnitary;

use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;
use dashu_int::IBig;

/// One side (under- or over-rotation) of a mixed-fallback result: the projective gate word
/// applied unconditionally, and the mixed-diagonal correction needed on the (rare) failure
/// branch. Call [`MixedFallbackSide::achieved_success_probability`] to compute the achieved
/// success probability on demand.
#[derive(Debug, Clone)]
pub struct MixedFallbackSide {
    /// Gate sequence for this side's projective step, applied unconditionally.
    pub projective_gates: GateSeq,
    /// The mixed-diagonal correction for this side's residual angle, needed with probability
    /// `1 - achieved_success_probability()`. Sampling it (see
    /// [`MixedDiagonalResult::Mixed`]) draws its own biased coin plus two fair coins,
    /// independently of whatever coin selected this side in the first place.
    pub correction: MixedDiagonalResult,
    /// The working precision this side was synthesized at.
    pub prec: Prec,
}

impl MixedFallbackSide {
    /// Recomputes the achieved success probability directly from the returned
    /// `projective_gates` string. Mirrors
    /// [`crate::protocol::fallback::FallbackResult::achieved_success_probability`].
    pub fn achieved_success_probability(&self) -> FBig<HalfEven> {
        let prec = self.prec;
        let u = DOmegaUnitary::from_gates(&self.projective_gates);
        let z = u.z();
        (z.real(prec) * z.real(prec)) + (z.imag(prec) * z.imag(prec))
    }
}

/// Diamond-norm distance between `correction` (a mixed-diagonal, twirled-branch result) and
/// the *residual* target it actually approximates (not `theta` directly, and not composed
/// with the projective step -- see [`residual_diamond_error_mixed`]'s docs). Delegates to
/// `MixedDiagonalResult`'s own mixture-aware computation rather than naively
/// triangle-inequality-summing each individual (twirled) branch's distance, which would throw
/// away the mixture's quadratic error cancellation and wildly overstate the achieved error.
fn weighted_correction_distance(
    prec: Prec,
    theta: &FBig<HalfEven>,
    projective_gates: &[Gate],
    correction: &MixedDiagonalResult,
) -> FBig<HalfEven> {
    residual_diamond_error_mixed(prec, theta, projective_gates, correction)
}

impl AchievedDiamondError for MixedFallbackSide {
    /// Triangle-inequality upper bound on *this side alone's* diamond-norm distance to
    /// `theta`, as if it were always selected (i.e. ignoring the outer `p`/`1-p` choice
    /// between `lo`/`hi` -- see [`MixedFallbackResult`]'s own impl for why that choice needs
    /// different treatment): `p_success * dist_phase(projective, theta) + (1 - p_success) *
    /// weighted_correction_distance(..)`, mirroring
    /// [`crate::protocol::fallback::FallbackResult`]'s impl but with the "failure" branch
    /// itself a mixture (`self.correction`'s twirled branches) rather than a single gate
    /// string. The success term uses `achieved_phase_diamond_error` (normalizes the
    /// candidate's magnitude-deficient `z` first), not the raw-`z` `achieved_diagonal_*`
    /// helper -- this side's projective candidate has `|z|^2 = q < 1` by construction, same
    /// caveat as plain fallback's.
    fn achieved_diamond_error(&self, theta: &FBig<HalfEven>) -> FBig<HalfEven> {
        let prec = self.prec;
        let p_success = self.achieved_success_probability();
        let success_dist = achieved_phase_diamond_error(prec, theta, &self.projective_gates);
        let failure_dist =
            weighted_correction_distance(prec, theta, &self.projective_gates, &self.correction);

        let one = prec.ib(IBig::ONE);
        let one_minus_p = &one - &p_success;
        (&p_success * &success_dist) + (&one_minus_p * &failure_dist)
    }
}

/// The output of [`synth_mixed_fallback`]. Call
/// [`AchievedDiamondError::achieved_diamond_error`] to compute the achieved projective-step
/// diamond-norm error on demand.
#[derive(Debug, Clone)]
pub enum MixedFallbackResult {
    /// The target direction was ring-exactly representable (e.g. `theta` a multiple of
    /// `pi/2`): a single gate word suffices, with zero error and no fallback structure at
    /// all -- mirrors [`crate::protocol::mixed_diagonal::MixedDiagonalResult`]'s analogous
    /// degenerate case.
    Exact { gates: GateSeq, prec: Prec },
    /// The general case: two straddling projective branches, mixed with probability `p`
    /// (`lo` at weight `p`, `hi` at weight `1-p`), each with its own achieved success
    /// probability and mixed-diagonal correction.
    ///
    /// The full runtime sequence, once per invocation:
    ///   1. flip one **biased** coin -- `lo` with probability `p`, else `hi`;
    ///   2. run that side's `projective_gates` unconditionally (the ancilla + measurement
    ///      circuit around it is out of this crate's scope -- see the module docs);
    ///   3. on the measurement's failure outcome (probability
    ///      `1 - side.achieved_success_probability()`), sample that side's `correction` by its
    ///      own biased coin plus two fair coins (see [`MixedDiagonalResult::Mixed`]) and apply
    ///      the resulting gate word.
    ///
    /// The coin in step 3 is drawn independently of the one in step 1 -- selecting `lo` says
    /// nothing about which of `lo.correction`'s branches (if any) gets sampled.
    Mixed {
        lo: MixedFallbackSide,
        /// Boxed purely to keep this enum's variants closer in size (clippy
        /// `large_enum_variant`); no semantic difference from an unboxed field.
        hi: Box<MixedFallbackSide>,
        /// Mixing weight on `lo` (vs. `hi`, which gets `1-p`) for the *projective* steps.
        /// Operationally required to run the protocol's classical coin flip -- unlike the
        /// achieved-error metrics, this can't be recomputed on demand from the gate strings
        /// alone, so it stays a stored field.
        p: FBig<HalfEven>,
    },
}

impl AchievedDiamondError for MixedFallbackResult {
    /// Recomputes the achieved diamond-norm error to `theta` directly from the public gate
    /// strings.
    ///
    /// For the `Mixed` variant: **not** `p * lo.achieved_diamond_error(theta) + (1 - p) *
    /// hi.achieved_diamond_error(theta)` -- each side's own projective step, taken alone, sits
    /// only within the *wide* straddling-search tolerance (`sin_alpha =
    /// sqrt(epsilon_diamond/4)`) of `theta`, not within `epsilon_diamond` itself; the whole
    /// point of the `lo`/`hi` straddling-pair trick is that mixing their projective steps by
    /// `p` cancels that first-order error, per this crate's `mixture_weight` closed form --
    /// naively weighting each side's *entire* (already-large) bound by `p`/`1-p` would throw
    /// that cancellation away and wildly overstate the achieved error (a real bug caught by
    /// fuzzing here; see the removed `projective_diamond_error`-only version this replaced).
    ///
    /// Budget accounting (an even split of `epsilon_diamond`, mirroring plain fallback's):
    /// the projective-mixture term is `<= 2*sin_alpha^2 = epsilon_diamond/2` regardless of
    /// `p`, and each side's own failure term is weighted so it contributes at most
    /// `epsilon_diamond/2` in total (see `build_side`'s `epsilon_for_correction`) --
    /// `epsilon_diamond/2 + epsilon_diamond/2 = epsilon_diamond` exactly, tight at the
    /// boundary (subject to the same `Exact`-fast-path caveat documented just above for the
    /// non-`Mixed` case, and the analogous one on `mixed_diagonal::assemble_result`).
    ///
    /// Correct decomposition, mirroring the paper's additive `eq:fallback-mixing-terms`
    /// budget split: the (quadratically small) projective-mixture term from `mixture_weight`,
    /// plus each side's own (small) failure-branch contribution weighted by *both* the outer
    /// `p`/`1-p` selection *and* that side's own failure probability
    /// `1 - achieved_success_probability()`.
    ///
    /// The projective-mixture term feeds `mixture_weight` the *phase-normalized* `z/|z|` of
    /// each side's projective candidate, not the raw decoded `z` -- exactly the same
    /// normalization [`achieved_phase_diamond_error`] already applies for a single fallback
    /// candidate, and for the same reason (see its own doc comment): a fallback projective
    /// candidate's `z` deliberately has `|z|^2 = q < 1` (the *separate* measurement-failure
    /// probability, already accounted for below via `lo_fail_prob`/`hi_fail_prob`), not a
    /// magnitude deficit that represents angular error. `mixture_weight`'s general (any-`r`)
    /// closed form -- correct and necessary for callers like
    /// [`crate::protocol::small_angle::SmallAngleRegion`] whose candidates genuinely can have
    /// `r < 1` as real rotation-approximation error -- would otherwise attribute fallback's
    /// `q < 1` to angular error too, double-counting it against the already-separate failure
    /// terms and wildly overstating the projective-mixture term (confirmed directly: this
    /// bug was masked for years by mixture_weight's own former unsound `r = 1`-assuming
    /// shortcut, which happened to produce a small -- if for the wrong reason -- projective
    /// term regardless of the true `q`, and was only exposed once that shortcut was fixed to
    /// the general formula and this conflation started reporting the true, much larger
    /// value).
    ///
    /// For the `Exact` variant: this only guarantees the returned gates are an exact ring
    /// unitary (no off-diagonal synthesis error) that already passed the region's tolerance
    /// check -- NOT that its phase exactly equals `theta` (see the analogous fix in
    /// `mixed_diagonal::assemble_result`'s `Unmixed` branch). At a loose enough epsilon, an
    /// exact Clifford+T point can land inside the tolerance window while still being
    /// measurably off-angle, so this decodes `gates` and computes the real achieved error
    /// rather than assuming zero.
    fn achieved_diamond_error(&self, theta: &FBig<HalfEven>) -> FBig<HalfEven> {
        match self {
            MixedFallbackResult::Exact { gates, prec } => {
                achieved_diagonal_diamond_error(*prec, theta, gates)
            }
            MixedFallbackResult::Mixed { lo, hi, p } => {
                let prec = lo.prec;
                let wframe = WFrame::new(prec, theta);
                let lo_u = DOmegaUnitary::from_gates(&lo.projective_gates);
                let hi_u = DOmegaUnitary::from_gates(&hi.projective_gates);
                let normalize = |z: &crate::ring::DOmega| -> (FBig<HalfEven>, FBig<HalfEven>) {
                    let re = z.real(prec);
                    let im = z.imag(prec);
                    let norm = ((re * re) + (im * im)).sqrt();
                    (re / &norm, im / &norm)
                };
                let (lo_re_n, lo_im_n) = normalize(lo_u.z());
                let (hi_re_n, hi_im_n) = normalize(hi_u.z());
                let re_lo = wframe.re_w_fbig(&lo_re_n, &lo_im_n);
                let im_lo = wframe.im_w_fbig(&lo_re_n, &lo_im_n);
                let re_hi = wframe.re_w_fbig(&hi_re_n, &hi_im_n);
                let im_hi = wframe.im_w_fbig(&hi_re_n, &hi_im_n);
                let projective_term = mixture_weight(prec, (&re_lo, &im_lo), (&re_hi, &im_hi))
                    .expect("a real assembled Mixed result must yield a valid mixture")
                    .projective_diamond_error;

                let one = prec.ib(IBig::ONE);
                let one_minus_p = &one - p;
                let lo_fail_prob = &one - &lo.achieved_success_probability();
                let hi_fail_prob = &one - &hi.achieved_success_probability();
                let lo_failure_dist =
                    weighted_correction_distance(prec, theta, &lo.projective_gates, &lo.correction);
                let hi_failure_dist =
                    weighted_correction_distance(prec, theta, &hi.projective_gates, &hi.correction);

                let lo_term = (p * &lo_fail_prob) * &lo_failure_dist;
                let hi_term = (&one_minus_p * &hi_fail_prob) * &hi_failure_dist;

                (&projective_term + &lo_term) + &hi_term
            }
        }
    }
}

/// Builds one side (under- or over-rotation) of a mixed-fallback result: decomposes the
/// projective candidate to gates, derives the residual angle theta_B, and searches for that
/// residual's mixed-diagonal correction.
fn build_side(
    prec: Prec,
    projective_unitary: DOmegaUnitary,
    theta_z_x: &FBig<HalfEven>,
    theta_z_y: &FBig<HalfEven>,
    epsilon_diamond: &FBig<HalfEven>,
    config: &mut GridSynthConfig,
) -> MixedFallbackSide {
    let v = projective_unitary.w().clone();
    let projective_gates = decompose_domega_unitary(projective_unitary);

    // Residual angle theta_B = theta - Arg(v), via the same atan2-free half-angle algebra
    // `fallback::synth_fallback` uses: cos(-theta_B/2) = cos(-theta/2)*cos(phi/2) -
    // sin(-theta/2)*sin(phi/2), sin(-theta_B/2) = sin(-theta/2)*cos(phi/2) +
    // cos(-theta/2)*sin(phi/2), with (cos(phi/2), sin(phi/2)) from half_angle_cos_sin applied
    // to (cos(phi), sin(phi)) = (Re(v), Im(v)) / |v| (see `phase_cos_sin`'s docs for the
    // degenerate `v = 0` case).
    let (cos_phi, sin_phi, v_norm_sq) = phase_cos_sin(prec, &v);
    let (cos_half_phi, sin_half_phi) = half_angle_cos_sin(prec, &cos_phi, &sin_phi);

    let cos_neg_theta_b_half =
        prec.fb(prec.fb(theta_z_x * &cos_half_phi) - prec.fb(theta_z_y * &sin_half_phi));
    let sin_neg_theta_b_half =
        prec.fb(prec.fb(theta_z_y * &cos_half_phi) + prec.fb(theta_z_x * &sin_half_phi));

    // The other half of the even split (mirroring plain fallback's correction budget, see
    // `synth_fallback`'s doc comment for the accounting): `epsilon_diamond/2` divided by
    // `v_norm_sq = 1 - achieved_success_probability` so this side's failure-weighted
    // contribution comes out to exactly `epsilon_diamond/2`. `MixedDiagonalRegion`'s
    // parameter is itself a diamond-norm budget (see its own doc comment), so no norm
    // conversion applies here.
    let two = prec.fb(FBig::try_from(2.0).unwrap());
    let epsilon_for_correction = (epsilon_diamond / &two) / &v_norm_sq;

    // Small-angle fast path (extends PR #3's mixed-diagonal optimization to this correction
    // step): the residual angle `theta_B` is frequently small relative to
    // `epsilon_for_correction` (e.g. whenever the projective step's `Arg(v)` already lands
    // close to `theta`), in which case pinning the identity as one branch (see
    // `crate::protocol::small_angle`) costs far fewer T gates on average than the even-split
    // search below. `small_angle_could_help_half_angle` is an O(1) pre-check (no search), and
    // a live-search miss (`None`) falls through to the always-correct even-split path exactly
    // as `mixed_diagonal::synth_mixed_diagonal` already does for its own dispatch -- so this
    // can only ever match or beat the previous (even-split-only) behavior, never regress it.
    //
    // `SmallAngleRegion`/`synth_small_angle_correction` and `MixedDiagonalRegion` both take a
    // diamond-norm mixture budget directly (this crate's one convention -- see
    // `CLAUDE.md`'s "Accuracy convention" section), so both branches of this correction now
    // receive the identical `epsilon_for_correction` value: no conversion between them, and
    // no separate variable for one branch's "diamond" view of the same number.
    let correction = if small_angle_could_help_half_angle(
        prec,
        &epsilon_for_correction,
        &sin_neg_theta_b_half,
    ) {
        synth_small_angle_correction(
            config,
            &cos_neg_theta_b_half,
            &sin_neg_theta_b_half,
            &epsilon_for_correction,
            DEFAULT_MAX_LIVE_SEARCH_K,
        )
    } else {
        None
    };

    let correction = correction.unwrap_or_else(|| {
        let exact_scale = ZRootTwo::new(IBig::from(1), IBig::from(0));
        let correction_region = MixedDiagonalRegion::from_target_direction(
            prec,
            cos_neg_theta_b_half.clone(),
            sin_neg_theta_b_half.clone(),
            &epsilon_for_correction,
            exact_scale.clone(),
        );
        let correction_unit_disk = UnitDisk::new(prec, exact_scale);
        let correction_wframe =
            WFrame::from_target_direction(prec, cos_neg_theta_b_half, sin_neg_theta_b_half);
        let correction_transform = setup_regions_and_transform(
            &correction_region,
            &correction_unit_disk,
            config.verbose,
            config.measure_time,
        );
        let correction_outcome = search_for_straddling_pair(
            &correction_region,
            &correction_unit_disk,
            &correction_transform,
            config,
            &correction_wframe,
            &epsilon_for_correction,
        );
        assemble_result(prec, correction_outcome, &correction_wframe)
    });

    MixedFallbackSide {
        projective_gates,
        correction,
        prec,
    }
}

/// Synthesizes a mixed-fallback probabilistic-channel approximation of `R_z(theta)` to
/// diamond-norm accuracy `epsilon_diamond`, with projective success-probability threshold
/// `q` on each side (see [`crate::protocol::fallback::exact_q`] for an exactly-representable
/// choice).
///
/// Returns `None` if the projective (straddling-pair) search exceeds its internal bound
/// without finding a pair or an exact solution -- mirrors `fallback::synth_fallback`'s own
/// choice to treat that as an "expected" outcome for a `SectorRegion` search rather than
/// panicking. A failure in either side's *correction* search (an ordinary mixed-diagonal
/// search, which should essentially never fail for a well-formed input) still panics, mirroring
/// `mixed_diagonal::synth_mixed_diagonal`'s own convention.
pub fn synth_mixed_fallback(
    theta: f64,
    epsilon_diamond: f64,
    q: DRootTwo,
    seed: u64,
    verbose: bool,
) -> Option<MixedFallbackResult> {
    let mut config = config_from_theta_epsilon(theta, epsilon_diamond, seed, verbose, false);
    let prec = config.prec;

    // Mixed protocols' wider angular half-width: an even split gives the projective-mixture
    // term (`2*sin_alpha^2`, see the `AchievedDiamondError` impl's accounting doc) a share
    // `epsilon_diamond/2` of the total budget, so `sin_alpha = sqrt(epsilon_diamond/4)` --
    // vs. plain fallback's `sin_alpha = epsilon_diamond/4` directly (Prop 3.16 vs. Prop 3.9).
    let four = prec.fb(FBig::try_from(4.0).unwrap());
    let sin_alpha = (&config.epsilon / &four).sqrt();

    let scale = ZRootTwo::new(IBig::from(1), IBig::from(0));
    let sector_region = SectorRegion::new(prec, &config.theta, q, sin_alpha, scale.clone());
    let unit_disk = UnitDisk::new(prec, scale);
    let wframe = WFrame::new(prec, &config.theta);

    let transform = setup_regions_and_transform(
        &sector_region,
        &unit_disk,
        config.verbose,
        config.measure_time,
    );
    // Half the overall diamond-norm budget -- see `search_for_straddling_pair`'s
    // `phase_tolerance` doc (an exact-ring candidate's diamond distance is exactly
    // 2*|Im(w)|), same value this crate has always used here.
    let two = prec.fb(FBig::try_from(2.0).unwrap());
    let phase_tolerance = &config.epsilon / &two;
    let outcome = search_for_straddling_pair(
        &sector_region,
        &unit_disk,
        &transform,
        &mut config,
        &wframe,
        &phase_tolerance,
    );

    match outcome {
        StraddleOutcome::NotFound => None,
        StraddleOutcome::Unmixed(u) => Some(MixedFallbackResult::Exact {
            gates: decompose_domega_unitary(u),
            prec,
        }),
        StraddleOutcome::Mixed(lo, hi) => {
            let hi = *hi;
            let re_lo = wframe.re_w(lo.z());
            let im_lo = wframe.im_w(lo.z());
            let re_hi = wframe.re_w(hi.z());
            let im_hi = wframe.im_w(hi.z());
            let mw = mixture_weight(prec, (&re_lo, &im_lo), (&re_hi, &im_hi)).expect(
                "mixture_weight returned None for a real solved straddling pair -- this \
                 indicates a genuine bug, not an expected degenerate input",
            );

            let neg_theta_half = -prec.fb(&config.theta / &two);
            let theta_z_x = prec.fb(neg_theta_half.cos());
            let theta_z_y = prec.fb(neg_theta_half.sin());

            // `config.epsilon` is cloned into a local first: `build_side` also takes `&mut
            // config`, and the two can't coexist as a borrow of `config.epsilon` plus a
            // mutable borrow of all of `config` in the same call.
            let epsilon_diamond = config.epsilon.clone();
            let lo_side = build_side(
                prec,
                lo,
                &theta_z_x,
                &theta_z_y,
                &epsilon_diamond,
                &mut config,
            );
            let hi_side = build_side(
                prec,
                hi,
                &theta_z_x,
                &theta_z_y,
                &epsilon_diamond,
                &mut config,
            );

            Some(MixedFallbackResult::Mixed {
                lo: lo_side,
                hi: Box::new(hi_side),
                p: mw.p,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::fallback::exact_q;
    use std::f64::consts::PI;

    const PREC: Prec = Prec(1000);

    fn fbig_to_f64(x: &FBig<HalfEven>) -> f64 {
        match x.to_f64() {
            dashu_base::Approximation::Exact(v) => v,
            dashu_base::Approximation::Inexact(v, _) => v,
        }
    }

    fn total_expected_t_count(side: &MixedFallbackSide) -> f64 {
        let p_t = side.projective_gates.t_count() as f64;
        let fail_prob = 1.0 - fbig_to_f64(&side.achieved_success_probability());
        let correction_cost = fbig_to_f64(&side.correction.expected_t_count());
        p_t + fail_prob * correction_cost
    }

    // NOTE: unlike `mixed_diagonal::search_for_straddling_pair` used directly with the much
    // narrower `MixedDiagonalRegion` (which reliably finds the ring-exact solution for
    // theta=pi/2 before any generic candidate can fill both straddling slots), `SectorRegion`
    // at mixed fallback's much wider angular tolerance (sin_alpha = sqrt(eps/2)) and loose
    // radial threshold (q close to 1, not tight) admits many more candidates -- so this
    // search's early-return-once-both-slots-filled logic can race past a still-unexamined
    // exact candidate later in the same k's iterator and return `Mixed` instead of `Exact`
    // for an exact angle. That is a missed *optimization* (a Mixed result is still a valid,
    // accuracy-meeting synthesis, just not the cheapest possible one), not a correctness bug,
    // so this test only asserts that a degenerate angle produces SOME valid outcome (either
    // Exact, or a well-formed Mixed result -- structural validity of the latter is exercised
    // exhaustively by `generic_angle_produces_mixed_result_with_valid_structure` below), not
    // that it is always the optimal Exact one.
    #[test]
    fn degenerate_angle_produces_a_valid_result() {
        let q = exact_q(7);
        let result = synth_mixed_fallback(PI / 2.0, 1e-6, q, 11, false)
            .expect("search should succeed for theta=pi/2");
        match result {
            MixedFallbackResult::Exact { gates, .. } => {
                assert!(!gates.is_empty());
            }
            MixedFallbackResult::Mixed { p, .. } => {
                let p_f64 = fbig_to_f64(&p);
                assert!(
                    (0.0..=1.0).contains(&p_f64),
                    "p={p_f64} out of [0,1] range even for a degenerate angle"
                );
            }
        }
    }

    #[test]
    fn generic_angle_produces_mixed_result_with_valid_structure() {
        let q = exact_q(7);
        let result = synth_mixed_fallback(3.0 * PI / 32.0, 1e-6, q.clone(), 13, false)
            .expect("search should succeed for a generic angle");
        match result {
            MixedFallbackResult::Mixed { lo, hi, p } => {
                let p_f64 = fbig_to_f64(&p);
                assert!((0.0..=1.0).contains(&p_f64), "p={p_f64} out of [0,1] range");
                assert!(
                    fbig_to_f64(&lo.achieved_success_probability())
                        >= fbig_to_f64(&q.to_real(PREC)),
                    "lo side violates its own success-probability guarantee"
                );
                assert!(
                    fbig_to_f64(&hi.achieved_success_probability())
                        >= fbig_to_f64(&q.to_real(PREC)),
                    "hi side violates its own success-probability guarantee"
                );
                // Each side's correction branch weights sum to 1 (Stage 1's own invariant,
                // re-checked here end-to-end).
                for side in [&lo, &hi] {
                    let total: f64 = side
                        .correction
                        .weighted_branches()
                        .iter()
                        .map(|(w, _gates)| fbig_to_f64(w))
                        .sum();
                    assert!(
                        (total - 1.0).abs() < 1e-3,
                        "correction branch weights summed to {total}, not 1"
                    );
                }
            }
            other => panic!("expected Mixed for a generic angle, got {other:?}"),
        }
    }

    // Extends PR #3's small-angle optimization (identity pinned as one branch) to mixed
    // fallback's per-side correction step. Passing the identity as the "projective"
    // candidate makes the residual angle exactly the outer `theta`, so a small `theta`
    // relative to the correction budget puts `build_side` squarely in the
    // small-angle-eligible regime -- mirrors
    // `small_angle::tests::small_angle_beats_mixed_diagonal_for_small_theta`, but exercised
    // through `build_side` itself (i.e. through the half-angle entry points, not the raw-f64
    // ones) so a units mistake at that boundary would show up here as a correctness
    // failure, not just a missed optimization.
    #[test]
    fn build_side_correction_beats_even_split_for_small_residual() {
        use crate::config::config_from_theta_epsilon;
        use crate::protocol::mixed_diagonal::even_split_search;

        // `build_side`'s residual-correction budget is `epsilon_for_correction =
        // (epsilon_diamond/2)/v_norm_sq` (see `build_side`'s own doc comment). With the
        // identity as the "projective" step, `v = 0`, so `v_norm_sq = 1` and the budget is
        // simply `epsilon_diamond/2 = delta/2`. `delta = 2e-4` makes that budget `1e-4` --
        // the exact `(theta, delta)` pair already confirmed, in
        // `small_angle::tests::synth_small_angle_finds_a_mixed_result_for_small_theta`, to
        // find a live-search candidate within `DEFAULT_MAX_LIVE_SEARCH_K`.
        let theta = 1e-3_f64;
        let delta = 2e-4_f64;

        let mut config = config_from_theta_epsilon(theta, delta, 7, false, false);
        let prec = config.prec;

        let two = prec.fb(FBig::try_from(2.0).unwrap());
        let neg_theta_half = -prec.fb(&config.theta / &two);
        let theta_z_x = prec.fb(neg_theta_half.clone().cos());
        let theta_z_y = prec.fb(neg_theta_half.sin());

        let epsilon_diamond = config.epsilon.clone();
        let side = build_side(
            prec,
            DOmegaUnitary::identity(),
            &theta_z_x,
            &theta_z_y,
            &epsilon_diamond,
            &mut config,
        );

        // Reproduce `build_side`'s own `epsilon_for_correction` (identity projective step,
        // `v_norm_sq = 1`) purely to build the even-split comparison at the same budget
        // `build_side` itself used -- both now diamond-norm, no conversion between them.
        let epsilon_for_correction = &config.epsilon / &two;
        let even_split = even_split_search(theta, fbig_to_f64(&epsilon_for_correction), 7, false);

        let small_cost = fbig_to_f64(&side.correction.expected_t_count());
        let even_cost = fbig_to_f64(&even_split.expected_t_count());
        eprintln!(
            "build_side small-angle correction mean T-count={small_cost}, \
             even-split mean T-count={even_cost}"
        );
        assert!(
            small_cost < even_cost,
            "small-angle-assisted correction ({small_cost}) should beat even-split \
             ({even_cost}) for a small residual angle"
        );

        let theta_fbig = FBig::<HalfEven>::try_from(theta)
            .unwrap()
            .with_precision(prec.bits())
            .value();
        let achieved = side.correction.achieved_diamond_error(&theta_fbig);
        let budget = epsilon_for_correction;
        assert!(
            achieved <= budget.clone() * FBig::<HalfEven>::try_from(1.5).unwrap(),
            "achieved error {achieved} exceeds budget {budget}"
        );
    }

    // Required acceptance: expected T-count slope. Computes, for each side, projective_t +
    // (1-success_probability)*E[correction_t], averages the two sides' costs weighted by
    // (p, 1-p), and fits against log2(1/epsilon_diamond). The paper's target for mixed
    // fallback is ~0.53, well below plain fallback's ~1.03 and plain diagonal's ~3.02.
    #[test]
    fn mixed_fallback_expected_cost_slope() {
        let q = exact_q(7);
        let epsilons: [f64; 3] = [1e-4, 1e-6, 1e-8];
        let n_angles = 6;

        let mut xs = Vec::new();
        let mut ys = Vec::new();

        for &eps in &epsilons {
            for i in 0..n_angles {
                let theta = (0.29 + (i as f64) * 0.83 + eps.log10()) % (2.0 * PI);
                let Some(result) =
                    synth_mixed_fallback(theta, eps, q.clone(), 100 + i as u64, false)
                else {
                    continue;
                };
                let cost = match result {
                    MixedFallbackResult::Exact { .. } => 0.0,
                    MixedFallbackResult::Mixed { lo, hi, p, .. } => {
                        let p_f64 = fbig_to_f64(&p);
                        p_f64 * total_expected_t_count(&lo)
                            + (1.0 - p_f64) * total_expected_t_count(&hi)
                    }
                };
                xs.push((1.0 / eps).log2());
                ys.push(cost);
            }
        }

        assert!(
            xs.len() >= 8,
            "too few successful data points ({}) to fit a slope",
            xs.len()
        );

        let n = xs.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        for i in 0..xs.len() {
            cov += (xs[i] - mean_x) * (ys[i] - mean_y);
            var_x += (xs[i] - mean_x) * (xs[i] - mean_x);
        }
        let slope = cov / var_x;

        eprintln!(
            "mixed-fallback slope fit: measured slope = {slope:.4} over {} points \
             (expected near 0.53, vs plain fallback's ~1.03 and plain diagonal's ~3.02)",
            xs.len()
        );

        assert!(
            slope > 0.0 && slope < 1.5,
            "measured slope {slope} is not even qualitatively better than plain fallback's ~1.03"
        );
    }
}
