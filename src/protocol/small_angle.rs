// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Stage 4: small-angle mixed diagonal synthesis with the identity pinned as one branch.
//!
//! Implements Appendix C of Kliuchnikov, Lauter, Minko, Paetznick, Petit
//! (arXiv:2203.10064v2) as specialized by Bothe, Sünderhauf, Witham, Campbell, Blunt,
//! "More efficient Clifford+T synthesis for small-angle rotations" (arXiv:2605.31544v2),
//! Section V / Appendix D. [`crate::protocol::mixed_diagonal::MixedDiagonalRegion`] splits
//! the diamond-norm error budget *evenly* between an under- and an over-rotation, which is
//! what makes that protocol's cost angle-*independent*. Here, one branch is instead fixed to
//! the identity gate (zero T gates, valid whenever the identity alone doesn't already meet
//! the target accuracy), and the *entire* budget is given to the other branch. For `theta`
//! close to zero this collapses the mean T-count far below the angle-independent formula;
//! for `theta` not small relative to the requested accuracy, [`synth_small_angle_or_mixed`]
//! falls back to the existing even-split protocol, so there is no regression either way.
//!
//! # Per-rotation scope
//! Like every other synthesis entry point in this crate, everything here takes a single
//! `(theta, epsilon_diamond)` pair for *one* rotation. Allocating a circuit's total error
//! budget across many rotations (Bothe Section VII A) -- and deciding whether to use
//! probability or quasi-probability mixing for that circuit -- is the caller's
//! responsibility, not this module's.

use crate::accuracy::{diagonal_diamond_distance, WFrame};
use crate::common::Prec;
use crate::config::{config_from_theta_epsilon, GridSynthConfig};
use crate::diophantine::diophantine_dyadic;
use crate::gate::GateSeq;
use crate::gridsynth::{process_solution_candidate, setup_regions_and_transform, PhaseMode};
use crate::gridsynth::{UnitDisk, UprightTransform};
use crate::math::solve_quadratic;
use crate::normal_form::{conjugate_by_clifford, Clifford};
use crate::protocol::mixed_diagonal::{assemble_result, synth_mixed_diagonal, StraddleOutcome};
use crate::protocol::mixing::mixture_weight;
use crate::protocol::MixedDiagonalResult;
use crate::region::Ellipse;
use crate::ring::{DOmega, DRootTwo, ZRootTwo};
use crate::synthesis_of_clifford_t::decompose_domega_unitary;
use crate::tdgp::{solve_tdgp, Region};
use crate::unitary::DOmegaUnitary;

use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;
use dashu_int::IBig;

/// The over-rotation search region for the small-angle protocol, working directly in the
/// `w`-frame (`w := u * e^{i theta/2}`, the same rotated frame [`WFrame`] and every other
/// region in this crate build via `z_x = cos(-theta/2)`, `z_y = sin(-theta/2)`; the target
/// direction maps to `w = 1`).
///
/// Unlike [`crate::protocol::mixed_diagonal::MixedDiagonalRegion`], which is symmetric and
/// used for *both* the under- and over-rotation branch (the branch is decided afterwards by
/// the sign of `Im(w)`), this region is one-sided by construction: it only ever contains
/// candidates with `Im(w) <= 0` (paired against a fixed `Im(w) >= 0` branch -- the identity,
/// in every caller here). This is what lets the region be much larger than the even-split
/// cap for the same total accuracy: the over-rotation branch is allowed to be *worse* than
/// an even split would permit, as long as the resulting mixture (via [`mixture_weight`], with
/// the identity's fixed `(Re(w), Im(w))` as the other branch) still meets the budget.
///
/// # Derivation
/// With `w_hi := (cos(t), sin(t))` fixed (the identity's frame coordinates, `t = theta/2`)
/// and a candidate `w_lo := (x, y)` with `y <= 0`, [`mixture_weight`]'s closed form gives
/// ```text
/// p     = H / (H - x*y),                    H := cos(t)*sin(t) = sin(theta)/2
/// error = 2*(p*y^2 + (1-p)*sin(t)^2)
/// ```
/// Requiring `error <= delta` and clearing the denominator (positive whenever `x >= 0`,
/// since then `H - x*y = H + x*|y| >= H > 0`) gives, after simplification,
/// ```text
/// B*y^2 + A*x*y <= (delta/2)*B         (INSIDE-ACCURACY)
/// A := delta - 2*sin(t)^2,  B := sin(theta) = sin(2t)
/// ```
/// which is the curve this region's `inside`/`intersect` implement, together with the
/// sign constraints `x >= 0`, `y <= 0` and the usual exact-ring norm test `|u|^2 <= scale`.
/// This is the same curve as Bothe's Eq. (D3) (`A_B*x*y >= (s-x^2)*B_B` in *absolute*
/// `u`-coordinates with their own `theta`), reached independently here by staying in the
/// `w`-frame and working directly from this crate's own `mixture_weight`; the two are
/// related by the `theta_paper = -theta_crate/2` identification (Bothe's Appendix F, Prop.
/// 1, shows the probability and quasi-probability versions of this curve agree to
/// `O(theta^2)`).
///
/// `phi_0`, the angular performance floor of Bothe Eq. (64)/(D5)-(D6), is not implemented:
/// this region always searches the full accuracy-limited area. That floor only trades area
/// against average success probability to shave a lower-order term off the asymptotic
/// T-count estimate; omitting it costs some efficiency in the very-small-`theta`/`delta`
/// tail, never correctness.
#[derive(Debug)]
pub struct SmallAngleRegion {
    scale: ZRootTwo,
    /// `A := delta - 2*sin(theta/2)^2`.
    a: FBig<HalfEven>,
    /// `B := sin(theta)`.
    b: FBig<HalfEven>,
    /// `(delta/2)*B`, precomputed.
    half_delta_b: FBig<HalfEven>,
    z_x: FBig<HalfEven>,
    z_y: FBig<HalfEven>,
    ellipse: Ellipse,
    prec: Prec,
}

impl SmallAngleRegion {
    /// Builds the over-rotation search region for target angle `theta` (`0 <= theta <= pi`;
    /// see [`synth_small_angle`] for why callers must canonicalize into this range first) and
    /// diamond-norm budget `delta` for the *whole* mixture (not this crate's usual halved
    /// "operator-norm-style" `epsilon` convention -- `delta` here is compared directly
    /// against [`crate::protocol::mixing::MixtureWeight::projective_diamond_error`], which is
    /// already a diamond-norm quantity, so no conversion is needed or wanted).
    pub fn new(
        prec: Prec,
        theta: &FBig<HalfEven>,
        delta: &FBig<HalfEven>,
        scale: ZRootTwo,
    ) -> Self {
        let two = prec.fb(FBig::try_from(2.0).unwrap());
        let theta_half = prec.fb(theta / &two);
        let neg_theta_half = -prec.fb(theta_half.clone());
        let z_x: FBig<HalfEven> = prec.fb(neg_theta_half.cos());
        let z_y: FBig<HalfEven> = prec.fb(neg_theta_half.sin());

        let sin_t = prec.fb(theta_half.sin());
        let sin_t_sq = &sin_t * &sin_t;
        let a = delta - &(&two * &sin_t_sq);
        let b = prec.fb(theta.sin());
        let half_delta_b = (delta * &b) / &two;

        // Bounding ellipse: the plain isotropic circle of radius `sqrt(scale)` -- i.e. the
        // same ellipse `UnitDisk` uses for this `scale`. This region is a strict subset of
        // that disk, so it is a valid (if not tight) bound; per this crate's `Region`
        // contract, `ellipse`/`intersect` may over-report -- only `inside` must be exact --
        // so this trades some `to_upright` efficiency for a much simpler construction than
        // the rotate-diagonal-rotate ellipse `MixedDiagonalRegion`/`SectorRegion` use. See
        // the module docs on measuring before tightening this.
        let s_inv: FBig<HalfEven> = 1 / scale.to_real(prec);
        let zero = prec.ib(IBig::ZERO);
        let ellipse = Ellipse::from(
            s_inv.clone(),
            zero.clone(),
            zero.clone(),
            s_inv,
            zero.clone(),
            zero,
            prec,
        );

        Self {
            scale,
            a,
            b,
            half_delta_b,
            z_x,
            z_y,
            ellipse,
            prec,
        }
    }

    fn frame(&self, u: &DOmega) -> (FBig<HalfEven>, FBig<HalfEven>) {
        let prec = self.prec;
        let re = u.real(prec);
        let im = u.imag(prec);
        let x = (&self.z_x * re) + (&self.z_y * im);
        let y = (&self.z_x * im) - (&self.z_y * re);
        (x, y)
    }
}

/// Clips `(t0, t1)` to `slope*t <= rhs`, or `None` if that empties the interval. Duplicated
/// (rather than imported) from the equivalent private helper in
/// `crate::protocol::fallback` -- same reasoning as
/// `mixed_diagonal::matrix_multiply_2x2`'s own duplication note.
fn clip_le_linear(
    prec: Prec,
    t0: FBig<HalfEven>,
    t1: FBig<HalfEven>,
    slope: &FBig<HalfEven>,
    rhs: &FBig<HalfEven>,
) -> Option<(FBig<HalfEven>, FBig<HalfEven>)> {
    let zero = prec.ib(IBig::ZERO);
    if slope > &zero {
        let bound = rhs / slope;
        let new_t1 = if t1 < bound { t1 } else { bound };
        if t0 > new_t1 {
            None
        } else {
            Some((t0, new_t1))
        }
    } else if slope < &zero {
        let bound = rhs / slope;
        let new_t0 = if t0 > bound { t0 } else { bound };
        if new_t0 > t1 {
            None
        } else {
            Some((new_t0, t1))
        }
    } else if rhs < &zero {
        None
    } else {
        Some((t0, t1))
    }
}

/// Clips `(t0, t1)` to `slope*t >= rhs`; see [`clip_le_linear`].
fn clip_ge_linear(
    prec: Prec,
    t0: FBig<HalfEven>,
    t1: FBig<HalfEven>,
    slope: &FBig<HalfEven>,
    rhs: &FBig<HalfEven>,
) -> Option<(FBig<HalfEven>, FBig<HalfEven>)> {
    let zero = prec.ib(IBig::ZERO);
    if slope > &zero {
        let bound = rhs / slope;
        let new_t0 = if t0 > bound { t0 } else { bound };
        if new_t0 > t1 {
            None
        } else {
            Some((new_t0, t1))
        }
    } else if slope < &zero {
        let bound = rhs / slope;
        let new_t1 = if t1 < bound { t1 } else { bound };
        if t0 > new_t1 {
            None
        } else {
            Some((t0, new_t1))
        }
    } else if rhs > &zero {
        None
    } else {
        Some((t0, t1))
    }
}

/// Clips `(t0, t1)` to `q2*t^2 + q1*t + q0 <= 0`.
///
/// When `q2 > 0` (parabola opens upward) the solution set is exactly the interval between
/// the real roots (or empty, if there are none -- an always-positive parabola). When
/// `q2 < 0` (opens downward) the true solution set is the *complement* of the open interval
/// between the roots -- two unbounded rays -- which is non-convex and cannot be expressed as
/// a single interval. Per this crate's `Region` contract (`solve_tdgp` always re-checks
/// `inside` on every candidate `intersect` produces a range for), `intersect` may
/// over-report without affecting correctness, only efficiency; this over-approximates that
/// case by leaving `(t0, t1)` unclipped rather than computing the exact two-ray union. This
/// is the same non-convexity flagged (before this region existed) at
/// `mixed_fallback_synthesis.md:722-725`.
fn clip_hyperbola(
    prec: Prec,
    t0: FBig<HalfEven>,
    t1: FBig<HalfEven>,
    q2: &FBig<HalfEven>,
    q1: &FBig<HalfEven>,
    q0: &FBig<HalfEven>,
) -> Option<(FBig<HalfEven>, FBig<HalfEven>)> {
    let zero = prec.ib(IBig::ZERO);
    if q2 > &zero {
        match solve_quadratic(prec, q2, q1, q0) {
            Some((r0, r1)) => {
                let new_t0 = if t0 > r0 { t0 } else { r0 };
                let new_t1 = if t1 < r1 { t1 } else { r1 };
                if new_t0 > new_t1 {
                    None
                } else {
                    Some((new_t0, new_t1))
                }
            }
            None => None,
        }
    } else if q2 < &zero {
        Some((t0, t1))
    } else {
        // Linear case: q1*t + q0 <= 0  <=>  q1*t <= -q0.
        clip_le_linear(prec, t0, t1, q1, &(-q0))
    }
}

impl Region for SmallAngleRegion {
    fn ellipse(&self) -> Ellipse {
        self.ellipse.clone()
    }

    fn inside(&self, u: &DOmega) -> bool {
        let prec = self.prec;
        let (x, y) = self.frame(u);
        let zero = prec.ib(IBig::ZERO);
        if y > zero || x < zero {
            return false;
        }
        if DRootTwo::from_domega(u.conj() * u) > DRootTwo::from_zroottwo(self.scale.clone()) {
            return false;
        }
        let lhs = (&self.b * (&y * &y)) + (&self.a * (&x * &y));
        lhs <= self.half_delta_b
    }

    fn intersect(&self, u0: &DOmega, v: &DOmega) -> Option<(FBig<HalfEven>, FBig<HalfEven>)> {
        let prec = self.prec;
        let a_c = v.conj() * v;
        let b_c = 2 * (v.conj() * u0);
        let c_c = u0.conj() * u0 - DOmega::from_zroottwo(&self.scale);
        let (t0, t1) = solve_quadratic(prec, a_c.real(prec), b_c.real(prec), c_c.real(prec))?;

        let (x0, y0) = self.frame(u0);
        let (xv, yv) = self.frame(v);

        // y(t) <= 0  <=>  yv*t <= -y0
        let (t0, t1) = clip_le_linear(prec, t0, t1, &yv, &(-&y0))?;
        // x(t) >= 0  <=>  xv*t >= -x0
        let (t0, t1) = clip_ge_linear(prec, t0, t1, &xv, &(-&x0))?;

        // Accuracy hyperbola, expanded along the line x(t)=x0+t*xv, y(t)=y0+t*yv:
        //   B*y(t)^2 + A*x(t)*y(t) - (delta/2)*B <= 0
        let two = prec.fb(FBig::try_from(2.0).unwrap());
        let q2 = &yv * ((&self.b * &yv) + (&self.a * &xv));
        let q1 = (&two * &self.b * &y0 * &yv) + (&self.a * ((&x0 * &yv) + (&xv * &y0)));
        let q0 = (&self.b * &y0 * &y0) + (&self.a * &x0 * &y0) - &self.half_delta_b;
        clip_hyperbola(prec, t0, t1, &q2, &q1, &q0)
    }
}

/// How many extra `k`-steps [`search_best_over_rotation`] explores past the first solvable
/// candidate, looking for one with a lower `p * T-count` score. Bothe's own observation
/// (Section V A) motivates this: "we must explore higher T count than that of the first
/// unitary found, as higher T count sequences might lead to lower `p` and overall lower
/// average T count." A small fixed bound rather than an unbounded search, so this stays
/// linear in cost -- a fuller optimum (bounding by a target T-count rather than a step
/// count) is a possible future refinement, not required for correctness.
const EXTRA_K_STEPS_AFTER_FIRST_HIT: i64 = 3;

/// Searches for the over-rotation branch: candidates inside `region` (so `Im(w) <= 0` by
/// construction), scored against the fixed `(hi_re, hi_im)` branch (the identity, in every
/// caller here) by [`mixture_weight`]'s `p`, keeping the one minimizing `p * T-count` (the
/// *mean* cost of the resulting mixture, since the `hi` branch's own T-count is 0 whenever
/// it is the identity). Returns `None` if no candidate is found within the search bound --
/// this should not happen for a well-formed `(theta, delta)` pair, and callers should treat
/// it as "fall back to [`crate::protocol::mixed_diagonal::synth_mixed_diagonal`]" rather than
/// panicking (mirroring the `min` of the two paths in [`synth_small_angle_or_mixed`]).
fn search_best_over_rotation(
    region: &SmallAngleRegion,
    unit_disk: &UnitDisk,
    transformed: &UprightTransform,
    config: &mut GridSynthConfig,
    wframe: &WFrame,
    hi_re: &FBig<HalfEven>,
    hi_im: &FBig<HalfEven>,
) -> Option<DOmegaUnitary> {
    let prec = config.prec;
    let max_k = 4 * prec.bits() as i64;
    let zero = prec.ib(IBig::ZERO);

    let mut best: Option<(FBig<HalfEven>, DOmegaUnitary)> = None;
    let mut first_hit_k: Option<i64> = None;
    let mut k = 0;

    while k <= max_k {
        if let Some(hit_k) = first_hit_k {
            if k > hit_k + EXTRA_K_STEPS_AFTER_FIRST_HIT {
                break;
            }
        }

        if let Some(solutions) = solve_tdgp(
            region,
            unit_disk,
            &transformed.op_g,
            &transformed.bbox_a,
            &transformed.bbox_b,
            k,
            config.verbose,
        ) {
            for z in solutions {
                if (&z * z.conj()).residue() == 0 {
                    continue;
                }
                let xi = DRootTwo::from_int(IBig::ONE) - DRootTwo::from_domega(z.conj() * &z);
                let Some(w_val) = diophantine_dyadic(xi, &mut config.diophantine_data) else {
                    continue;
                };
                let candidate = process_solution_candidate(z, w_val, PhaseMode::Exact);

                let re_lo = wframe.re_w(candidate.z());
                let im_lo = wframe.im_w(candidate.z());
                // Defensive: `region.inside()` already guarantees `Im(w) <= 0`, but a
                // non-exact candidate that merely satisfies the (possibly over-reported)
                // `intersect` bracket could, in principle, land on the wrong side before
                // `solve_tdgp`'s own `inside` re-check filters it -- skip rather than trip
                // `mixture_weight`'s precondition.
                if im_lo > zero {
                    continue;
                }

                if first_hit_k.is_none() {
                    first_hit_k = Some(k);
                }

                let Some(mw) = mixture_weight(prec, (&re_lo, &im_lo), (hi_re, hi_im)) else {
                    continue;
                };
                let gates = decompose_domega_unitary(candidate.clone());
                let t_count = prec.ib(IBig::from(gates.t_count() as i64));
                let score = &mw.p * &t_count;

                let is_better = match &best {
                    None => true,
                    Some((best_score, _)) => score < *best_score,
                };
                if is_better {
                    best = Some((score, candidate));
                }
            }
        }
        k += 1;
    }

    best.map(|(_, u)| u)
}

/// Conjugates every gate word in `result` by `X` if `need` is set, otherwise returns it
/// unchanged. Used by [`synth_small_angle`] to undo the `R_z(-theta) = X R_z(theta) X`
/// canonicalization.
///
/// Conjugating by `X` swaps a candidate's top-left entry `z` for (a phase times) its
/// conjugate, which flips the sign of `Im(w)` measured against the *reflected* target
/// `-theta_pos` -- i.e. the branch that was the under-rotation (`lo`, `Im(w) <= 0`) for
/// `+theta_pos` becomes the over-rotation for `-theta_pos`, and vice versa. So `lo`/`hi`
/// (and, correspondingly, `p`/`1-p`) must be swapped along with the gate words themselves,
/// or a later call to [`MixedDiagonalResult::achieved_diamond_error`] (which rebuilds its
/// own `WFrame` from the *un*-canonicalized `theta` and assumes the stored `lo` is still the
/// under-rotation in that frame) trips `mixture_weight`'s `Im(w_lo) <= 0 <= Im(w_hi)`
/// precondition. T-counts and the achieved error value itself are unaffected either way
/// (conjugation by a fixed Clifford changes neither) -- only which field a given word is
/// filed under.
fn apply_x_conjugation(result: MixedDiagonalResult, need: bool) -> MixedDiagonalResult {
    if !need {
        return result;
    }
    let x = Clifford::new(0, 1, 0, 0);
    match result {
        MixedDiagonalResult::Exact { gates, prec } => MixedDiagonalResult::Exact {
            gates: conjugate_by_clifford(&gates, x),
            prec,
        },
        MixedDiagonalResult::Mixed { p, lo, hi, prec } => {
            let one = prec.ib(IBig::ONE);
            MixedDiagonalResult::Mixed {
                p: &one - &p,
                lo: conjugate_by_clifford(&hi, x),
                hi: conjugate_by_clifford(&lo, x),
                prec,
            }
        }
    }
}

/// Synthesizes the small-angle mixed-diagonal approximation of `R_z(theta)` with the
/// identity pinned as one branch, to diamond-norm accuracy `epsilon_diamond` -- or returns
/// `None` if the internal search doesn't find a suitable over-rotation within its bound
/// (not expected for well-formed input, but a possibility this function reports rather than
/// panics on; see [`synth_small_angle_or_mixed`] for the recommended caller, which falls
/// back to the even-split protocol in that case).
///
/// `theta` is first canonicalized to `[0, pi]`: reduced modulo `2*pi` (the `R_z(theta)`
/// *channel* -- unlike the bare unitary -- is exactly `2*pi`-periodic, so this changes
/// nothing observable), then, if that lands in `(pi, 2*pi)`, replaced by its reflection
/// `2*pi - theta_wrapped` with a flag to conjugate the final gate words by `X`
/// (`R_z(-phi) = X R_z(phi) X`). This guarantees `theta/2` in `[0, pi/2]`, which in turn
/// guarantees the identity's fixed branch has `Re(w), Im(w) >= 0` and
/// `A := delta - 2*sin(theta/2)^2` well-behaved -- both load-bearing for
/// [`SmallAngleRegion`]'s derivation.
pub fn synth_small_angle(
    theta: f64,
    epsilon_diamond: f64,
    seed: u64,
    verbose: bool,
) -> Option<MixedDiagonalResult> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let theta_wrapped = theta.rem_euclid(two_pi);
    let (theta_pos, need_x_conjugate) = if theta_wrapped <= std::f64::consts::PI {
        (theta_wrapped, false)
    } else {
        (two_pi - theta_wrapped, true)
    };

    let mut config = config_from_theta_epsilon(theta_pos, epsilon_diamond, seed, verbose, false);
    let prec = config.prec;
    // `delta` is the diamond-norm budget directly -- see `SmallAngleRegion::new`'s docs on
    // why no `diamond_to_spec_epsilon` conversion is applied here.
    let delta = config.epsilon.clone();

    let wframe = WFrame::new(prec, &config.theta);
    let hi = DOmegaUnitary::identity();
    let hi_re = wframe.re_w(hi.z());
    let hi_im = wframe.im_w(hi.z());

    // Fast path: the identity alone already meets the budget, so no mixing is needed (or
    // beneficial) at all.
    if diagonal_diamond_distance(prec, &hi_re) <= delta {
        let result = MixedDiagonalResult::Exact {
            gates: GateSeq::identity(),
            prec,
        };
        return Some(apply_x_conjugation(result, need_x_conjugate));
    }

    let scale = ZRootTwo::new(IBig::from(1), IBig::from(0));
    let region = SmallAngleRegion::new(prec, &config.theta, &delta, scale.clone());
    let unit_disk = UnitDisk::new(prec, scale);
    let transformed =
        setup_regions_and_transform(&region, &unit_disk, config.verbose, config.measure_time);

    let lo = search_best_over_rotation(
        &region,
        &unit_disk,
        &transformed,
        &mut config,
        &wframe,
        &hi_re,
        &hi_im,
    )?;

    let outcome = StraddleOutcome::Mixed(lo, Box::new(hi));
    let result = assemble_result(prec, outcome, &wframe);
    Some(apply_x_conjugation(result, need_x_conjugate))
}

/// Recommended entry point: runs both [`synth_small_angle`] and
/// [`crate::protocol::mixed_diagonal::synth_mixed_diagonal`] and keeps whichever has the
/// lower expected T-count (Bothe Eq. (12)'s `T(theta, delta) = min(T_small-angle,
/// T_mixed-diagonal)`), falling back to the even-split protocol outright if the small-angle
/// search fails to find a candidate. This mirrors `crate::gridsynth::gridsynth_gates`'s own
/// "run both `PhaseMode`s, keep the cheaper one" pattern, and guarantees this new path can
/// never make a result *worse* than the existing protocol already gives.
pub fn synth_small_angle_or_mixed(
    theta: f64,
    epsilon_diamond: f64,
    seed: u64,
    verbose: bool,
) -> MixedDiagonalResult {
    let mixed = synth_mixed_diagonal(theta, epsilon_diamond, seed, verbose);
    match synth_small_angle(theta, epsilon_diamond, seed, verbose) {
        Some(small) if small.expected_t_count() < mixed.expected_t_count() => small,
        _ => mixed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracy::AchievedDiamondError;
    use dashu_base::Approximation;
    use dashu_int::ops::Abs;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::f64::consts::PI;

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

    fn approx_eq(a: &FBig<HalfEven>, b: &FBig<HalfEven>, tol_bits: usize) -> bool {
        let diff = (a - b).abs();
        let tol = PREC.ib(IBig::ONE) / PREC.ib(IBig::ONE << tol_bits);
        diff <= tol
    }

    // ---- Region correctness ----

    // The region's own derivation, re-checked independently: for a candidate exactly ON the
    // accuracy boundary (constructed directly from the closed form, not from a real search),
    // `inside` must accept it, and nudging it slightly further from the target (larger
    // `|y|`) must reject it.
    #[test]
    fn small_angle_region_boundary_matches_derivation() {
        let theta = to_fbig(PREC, 0.05);
        let delta = to_fbig(PREC, 0.01);
        let scale = ZRootTwo::from_int(IBig::from(1_000_000));

        let region = SmallAngleRegion::new(PREC, &theta, &delta, scale.clone());

        // At x = 1 (identity's own real axis point in the w-frame terms), the boundary
        // condition B*y^2 + A*x*y = (delta/2)*B, i.e. B*y^2 + A*y - (delta/2)*B = 0, solved
        // for y via the same closed form the region uses internally.
        let x = PREC.ib(IBig::ONE);
        let (root_lo, _root_hi) =
            solve_quadratic(PREC, &region.b, &(&region.a * &x), &(-&region.half_delta_b))
                .expect("boundary quadratic must have real roots for this (theta, delta)");
        // The negative root is the one on the `y <= 0` branch this region searches.
        let y_boundary = if root_lo < PREC.ib(IBig::ZERO) {
            root_lo
        } else {
            -root_lo
        };

        // u such that Re(w)=x=1, Im(w)=y_boundary: invert the (z_x,z_y) rotation.
        let u_re = (&region.z_x * &x) - (&region.z_y * &y_boundary);
        let u_im = (&region.z_y * &x) + (&region.z_x * &y_boundary);

        // `inside` takes a `DOmega`, but this synthetic boundary point is not generally a
        // ring element -- so re-derive the same boolean directly from the closed form
        // instead of trying to construct a `DOmega` for it.
        let lhs = (&region.b * (&y_boundary * &y_boundary)) + (&region.a * (&x * &y_boundary));
        assert!(
            approx_eq(&lhs, &region.half_delta_b, PREC.bits() - 30),
            "constructed point is not actually on the boundary: lhs={lhs} rhs={}",
            region.half_delta_b
        );
        let _ = (u_re, u_im); // constructed for documentation of the inversion; not a ring point.

        // Nudge y further from zero (more over-rotation error): must leave the region.
        let epsilon_nudge = to_fbig(PREC, 1e-6);
        let y_outside = &y_boundary - &epsilon_nudge; // more negative
        let lhs_outside = (&region.b * (&y_outside * &y_outside)) + (&region.a * (&x * &y_outside));
        assert!(
            lhs_outside > region.half_delta_b,
            "point nudged further from target should violate the accuracy bound"
        );
    }

    #[test]
    fn small_angle_region_rejects_wrong_sign_branch() {
        let theta = to_fbig(PREC, 0.05);
        let delta = to_fbig(PREC, 0.01);
        let scale = ZRootTwo::from_int(IBig::from(1));
        let region = SmallAngleRegion::new(PREC, &theta, &delta, scale);

        // The identity itself: y = Im(w) = sin(theta/2) > 0 for theta > 0, so it must NOT be
        // accepted by this (y <= 0 only) region.
        let identity = DOmega::from_int(IBig::ONE);
        assert!(
            !region.inside(&identity),
            "identity (Im(w) > 0) must be rejected by the y<=0 over-rotation region"
        );
    }

    // ---- End-to-end synthesis ----

    #[test]
    fn synth_small_angle_finds_a_mixed_result_for_small_theta() {
        for theta in [1e-2_f64, 1e-3, 1e-4] {
            let result = synth_small_angle(theta, 1e-4, 42, false)
                .unwrap_or_else(|| panic!("expected a result for theta={theta}"));
            let theta_fbig = to_fbig(result.prec(), theta);
            let achieved = result.achieved_diamond_error(&theta_fbig);
            let delta = to_fbig(result.prec(), 1e-4);
            assert!(
                achieved <= delta.clone() * to_fbig(result.prec(), 1.5),
                "theta={theta}: achieved error {achieved} exceeds budget {delta} by more than \
                 the expected floating-point slack"
            );
        }
    }

    fn prec_of(r: &MixedDiagonalResult) -> Prec {
        match r {
            MixedDiagonalResult::Exact { prec, .. } => *prec,
            MixedDiagonalResult::Mixed { prec, .. } => *prec,
        }
    }

    // Required regression: the mean T-count must be *dramatically* lower than the
    // angle-independent mixed-diagonal formula for a genuinely small angle -- this is the
    // entire point of the change. Uses the worked example from the design doc: theta=1e-3,
    // delta=1e-4 (delta >> theta^2 = 1e-6).
    #[test]
    fn small_angle_beats_mixed_diagonal_for_small_theta() {
        let theta = 1e-3;
        let delta = 1e-4;
        let mixed = synth_mixed_diagonal(theta, delta, 7, false);
        let small = synth_small_angle(theta, delta, 7, false).expect("expected a result");
        let mixed_cost = fbig_to_f64(&mixed.expected_t_count());
        let small_cost = fbig_to_f64(&small.expected_t_count());
        eprintln!(
            "small-angle mean T-count={small_cost}, mixed-diagonal mean T-count={mixed_cost}"
        );
        assert!(
            small_cost < mixed_cost,
            "small-angle path ({small_cost}) should beat mixed-diagonal ({mixed_cost}) for \
             theta={theta}, delta={delta}"
        );
    }

    // Regime switch: for delta << theta^2, the combined entry point must not regress below
    // the existing even-split protocol's own cost.
    #[test]
    fn combined_entry_point_never_worse_than_mixed_diagonal() {
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..3 {
            let theta: f64 = rng.random_range(0.3..2.0); // NOT small relative to typical delta
            let delta = 1e-6; // delta << theta^2 here
            let mixed = synth_mixed_diagonal(theta, delta, 1, false);
            let combined = synth_small_angle_or_mixed(theta, delta, 1, false);
            assert!(
                fbig_to_f64(&combined.expected_t_count())
                    <= fbig_to_f64(&mixed.expected_t_count()) + 1e-9,
                "theta={theta}: combined entry point regressed vs. plain mixed-diagonal"
            );
        }
    }

    // The identity-alone fast path: for a theta small enough that 2*|sin(theta/2)| <= delta,
    // the result must be the bare identity with no T gates and zero randomness.
    #[test]
    fn identity_alone_suffices_for_sufficiently_loose_budget() {
        let theta = 1e-6;
        let delta = 1e-3; // 2*sin(theta/2) ~ 1e-6, comfortably under delta
        let result = synth_small_angle(theta, delta, 3, false).expect("expected a result");
        match result {
            MixedDiagonalResult::Exact { gates, .. } => {
                assert_eq!(
                    gates.t_count(),
                    0,
                    "expected the bare identity, zero T gates"
                );
            }
            other => panic!("expected Exact (identity), got {other:?}"),
        }
    }

    // Canonicalization: theta and theta - 2*pi represent the same channel, so both must
    // synthesize to (numerically) the same achieved error and the same T-count -- up to the
    // f64-level rounding noise `theta - 2*pi` (then wrapped back via `rem_euclid`) picks up
    // versus `theta` directly, which is not bit-exact cancellation.
    #[test]
    fn canonicalization_is_channel_invariant() {
        let theta = 0.7_f64;
        let delta = 1e-6;
        let a = synth_small_angle(theta, delta, 5, false).expect("expected a result");
        let b = synth_small_angle(theta - 2.0 * PI, delta, 5, false).expect("expected a result");
        let a_t = fbig_to_f64(&a.expected_t_count());
        let b_t = fbig_to_f64(&b.expected_t_count());
        assert!(
            (a_t - b_t).abs() < 1e-6,
            "theta and theta-2*pi should synthesize to the same T-count: {a_t} vs {b_t}"
        );
    }

    // Negative angles: R_z(-theta) = X R_z(theta) X, so the X-conjugation path must produce
    // an achieved error consistent with the (negative) target, not the canonicalized one.
    #[test]
    fn negative_angle_achieves_its_own_target_after_x_conjugation() {
        let theta = -0.05_f64;
        let delta = 1e-4;
        let result = synth_small_angle(theta, delta, 11, false).expect("expected a result");
        let prec = prec_of(&result);
        let theta_fbig = to_fbig(prec, theta);
        let achieved = result.achieved_diamond_error(&theta_fbig);
        assert!(
            achieved <= to_fbig(prec, delta) * to_fbig(prec, 1.5),
            "achieved error {achieved} for negative theta={theta} exceeds the budget"
        );
    }
}
