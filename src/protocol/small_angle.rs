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
//! for `theta` not small relative to the requested accuracy,
//! [`crate::protocol::mixed_diagonal::synth_mixed_diagonal`] (which dispatches here via
//! [`small_angle_could_help`]'s O(1) pre-check) falls back to the existing even-split
//! protocol at zero added cost, so there is no regression either way.
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
use crate::protocol::mixed_diagonal::{assemble_result, StraddleOutcome};
use crate::protocol::mixing::mixture_weight;
use crate::protocol::small_angle_table::{row_gates, OVER_ROTATION_TABLE};
use crate::protocol::MixedDiagonalResult;
use crate::region::Ellipse;
use crate::ring::{DOmega, DRootTwo, ZRootTwo};
use crate::synthesis_of_clifford_t::decompose_domega_unitary;
use crate::tdgp::{solve_tdgp, Region};
use crate::unitary::DOmegaUnitary;

use dashu_base::{Abs, Approximation};
use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;
use dashu_int::IBig;
use log::warn;
use nalgebra::{Matrix2, Vector2};

/// Same 2x2 matrix product helper as `gridsynth::matrix_multiply_2x2` /
/// `mixed_diagonal::matrix_multiply_2x2` (both private to their own modules) -- duplicated
/// verbatim rather than plumbed through a new `pub(crate)` export, matching the pattern
/// `mixed_diagonal.rs` itself already uses for the same helper.
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

fn fbig_to_f64(x: &FBig<HalfEven>) -> f64 {
    match x.to_f64() {
        Approximation::Inexact(v, _) => v,
        Approximation::Exact(v) => v,
    }
}

/// Builds the circular-cap bounding ellipse `x >= d` (in the `(z_x, z_y)`-aligned `w`-frame)
/// for [`SmallAngleRegion`], via the same rotate-diagonal-rotate (`d1*d2*d3`) construction
/// [`crate::protocol::mixed_diagonal::MixedDiagonalRegion`] uses for its own (differently
/// offset) cap: `h := sqrt(s) - d` is the radial semi-axis, `c := sqrt(s - d^2)` the
/// tangential one (the exact circle-chord half-length at `x = d`).
fn cap_ellipse(
    prec: Prec,
    scale_to_real: &FBig<HalfEven>,
    d: &FBig<HalfEven>,
    z_x: &FBig<HalfEven>,
    z_y: &FBig<HalfEven>,
) -> Ellipse {
    let one = prec.ib(IBig::ONE);
    let zero = prec.ib(IBig::ZERO);
    let sqrt_s = scale_to_real.sqrt();
    let h = &sqrt_s - d;
    let c_sq = scale_to_real - (d * d);
    let c = c_sq.max(zero.clone()).sqrt();

    let h_sq = &h * &h;
    let c_sq = &c * &c;
    let inv_h_sq = &one / &h_sq;
    let inv_c_sq = &one / &c_sq;

    let neg_z_y: FBig<HalfEven> = -(z_y.clone());
    let d1: Matrix2<FBig<HalfEven>> =
        Matrix2::new(z_x.clone(), neg_z_y.clone(), z_y.clone(), z_x.clone());
    let d2: Matrix2<FBig<HalfEven>> = Matrix2::new(inv_h_sq, zero.clone(), zero.clone(), inv_c_sq);
    let d3: Matrix2<FBig<HalfEven>> = Matrix2::new(z_x.clone(), z_y.clone(), neg_z_y, z_x.clone());

    let px = d * z_x;
    let py = d * z_y;
    let p = Vector2::new(px, py);
    let m1: Matrix2<FBig<HalfEven>> = matrix_multiply_2x2(prec, &d1, &d2);
    let m: Matrix2<FBig<HalfEven>> = matrix_multiply_2x2(prec, &m1, &d3);
    Ellipse::new(m, p, prec)
}

/// Computes the tight (or, failing that, safely-fall-back-to-loose) bounding ellipse for
/// [`SmallAngleRegion`] -- see its own doc comment for the derivation of `x_min`. Falls back
/// to the full-disk ellipse (`x_min = 0`, i.e. exactly what this region used before this
/// tightening existed) whenever the exact computation is inconclusive; that fallback is
/// always a valid (if loose) superset, so this function can never make [`SmallAngleRegion`]
/// incorrect, only sometimes less efficient than it could be.
fn tight_cap_ellipse(
    prec: Prec,
    a: &FBig<HalfEven>,
    b: &FBig<HalfEven>,
    c: &FBig<HalfEven>,
    scale_to_real: &FBig<HalfEven>,
    z_x: &FBig<HalfEven>,
    z_y: &FBig<HalfEven>,
) -> Ellipse {
    let zero = prec.ib(IBig::ZERO);
    let full_disk_ellipse = || {
        let s_inv: FBig<HalfEven> = 1 / scale_to_real.clone();
        Ellipse::from(
            s_inv.clone(),
            zero.clone(),
            zero.clone(),
            s_inv,
            zero.clone(),
            zero.clone(),
            prec,
        )
    };

    if a <= &zero || b <= &zero {
        return full_disk_ellipse();
    }

    let x_c_sq = c / b;
    let x_c = x_c_sq.sqrt();

    let two = prec.fb(FBig::try_from(2.0).unwrap());
    let qa = (a * a) + (b * b);
    let qb = -((&two * (b * c)) + ((a * a) * scale_to_real));
    let qc = c * c;
    let Some((u1, u2)) = solve_quadratic(prec, &qa, &qb, &qc) else {
        return full_disk_ellipse();
    };

    let mut best_x_min: Option<FBig<HalfEven>> = None;
    for u in [u1, u2] {
        if u < zero || u > *scale_to_real {
            continue;
        }
        let x = u.sqrt();
        if x > x_c {
            continue;
        }
        // Re-check the ORIGINAL (unsquared, correctly-signed) equation directly -- squaring
        // to get a polynomial in `u` introduces a spurious root on the `y >= 0` branch, and
        // this is what filters it out rather than trusting the `x <= x_c` heuristic alone.
        let disk_y = -((scale_to_real - &x * &x).max(zero.clone()).sqrt());
        let lhs = a * &x * &disk_y;
        let rhs = (b * &x * &x) - c;
        let residual = (&lhs - &rhs).abs();
        let scale_for_tol = rhs.clone().abs().max(lhs.abs()).max(c.clone());
        let tol = &scale_for_tol / prec.ib(IBig::ONE << (prec.bits() / 4).max(16));
        if residual > tol {
            continue;
        }
        best_x_min = Some(match best_x_min {
            None => x,
            Some(prev) => prev.min(x),
        });
    }

    let Some(x_min) = best_x_min else {
        return full_disk_ellipse();
    };

    // Conservative shrink: inflate the cap's radial depth `sqrt(s) - x_min` by 1% (plus a
    // tiny absolute margin) so ordinary rounding in the computation above can never make the
    // returned ellipse accidentally exclude a true boundary point -- `Region::ellipse` may
    // over-report without affecting correctness, so this costs a little tightness for a
    // large margin of safety.
    let sqrt_s = scale_to_real.sqrt();
    let depth = &sqrt_s - &x_min;
    let hundred = prec.fb(FBig::try_from(100.0).unwrap());
    let inflated_depth = (&depth * prec.fb(FBig::try_from(1.01).unwrap())) + (&sqrt_s / &hundred);
    let d = (&sqrt_s - &inflated_depth).max(zero);

    cap_ellipse(prec, scale_to_real, &d, z_x, z_y)
}

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
/// With `w_hi := (cos(t), sin(t))` fixed (the identity's frame coordinates, `t = theta/2`;
/// `Re(w_hi)^2 + Im(w_hi)^2 = 1` exactly, since the identity's own `z = 1` is exactly unit
/// modulus) and a candidate `w_lo := (x, y)` with `y <= 0`, [`mixture_weight`]'s *general*
/// closed form (valid for any `|w_lo| <= 1`, not just `|w_lo| = 1` -- **this is load-bearing
/// here**: unlike every other region in this crate, this region's `inside` test does not
/// force candidates to hug the boundary of the unit disk, so a real candidate can and does
/// have `|w_lo|` far below `1`) gives
/// ```text
/// p     = H / (H - x*y),                    H := cos(t)*sin(t) = sin(theta)/2
/// error = 2*(1 - p*x^2 - (1-p)*cos(t)^2)
/// ```
/// Requiring `error <= delta` and clearing the denominator (positive whenever `x >= 0`,
/// since then `H - x*y = H + x*|y| >= H > 0`) gives, after simplification,
/// ```text
/// A*x*y - B*x^2 + C <= 0                (INSIDE-ACCURACY)
/// A := delta - 2*sin(t)^2,  B := sin(theta) = sin(2t),  C := (1 - delta/2)*B
/// ```
/// which is the curve this region's `inside`/`intersect` implement, together with the
/// sign constraints `x >= 0`, `y <= 0` and the usual exact-ring norm test `|u|^2 <= scale`.
///
/// This supersedes an earlier, unsound version of this derivation (`B*y^2 + A*x*y <=
/// (delta/2)*B`) that instead used the `r = 1`-only simplification of `mixture_weight`'s
/// formula (`error = 2*(p*y^2 + (1-p)*sin(t)^2)`, which additionally substitutes `Re(w)^2 =
/// 1 - Im(w)^2` -- sound only when `|w| = 1` exactly). That version spuriously admitted
/// candidates with `x` near `0` (and hence `|w_lo|` far below `1`) as satisfying the
/// accuracy budget when they did not -- the achieved error for such a candidate is close to
/// `2` (the worst possible value), not the small number the old formula reported. See git
/// history for the concrete case this was caught on.
///
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
///
/// # Bounding ellipse
///
/// The true region (`INSIDE-ACCURACY` above, intersected with the disk `x^2+y^2 <= s` and
/// `x >= 0, y <= 0`) is entirely contained in the *circular cap* `x >= x_min` of that same
/// disk, where `x_min` is the disk/hyperbola intersection on the `y <= 0` branch -- exactly
/// the same shape [`crate::protocol::mixed_diagonal::MixedDiagonalRegion`] already bounds
/// (a chord cutting the disk), just with a different offset. This is because, writing
/// `A*x*y - B*x^2 + C <= 0` as `A*x*y >= B*x^2 - C`: for `x >= x_c := sqrt(C/B)` the RHS is
/// already `<= 0`, so *every* `y <= 0` in the disk satisfies the condition (the region is a
/// full circular segment there); for `x < x_c` the condition further requires
/// `y <= (B*x^2-C)/(A*x)` (dividing by `A*x > 0`), and the region is empty once that upper
/// bound falls below the disk's own lower edge `-sqrt(s-x^2)` -- which happens exactly at
/// `x = x_min <= x_c`. `x_min` is found by squaring `-A*x*sqrt(s-x^2) = B*x^2-C` (valid where
/// the RHS is `<= 0`, i.e. `x <= x_c`) into a quadratic in `u = x^2`:
/// `(A^2+B^2)*u^2 - (2*B*C + A^2*s)*u + C^2 = 0`, solved via [`solve_quadratic`] and filtered
/// by `x <= x_c` (squaring introduces a spurious root on the `y >= 0` branch) and by
/// re-checking the original, unsquared equation directly (guards against a wrong root choice
/// under any numerical noise). If that computation is inconclusive for any reason (e.g. `A`
/// or `B` non-positive, no real root, or the re-check fails), this falls back to `x_min = 0`
/// -- the previous, always-correct-but-loose full-disk bound -- rather than risk excluding
/// part of the true region. A conservative shrink (`x_min` pulled a bit further from `sqrt(s)`
/// than the exact computation gives) absorbs any remaining rounding slack, since making the
/// bounding ellipse larger can only cost efficiency, never correctness (`Region::ellipse` may
/// over-report; only `Region::inside` must be exact).
#[derive(Debug)]
pub struct SmallAngleRegion {
    scale: ZRootTwo,
    /// `A := delta - 2*sin(theta/2)^2`.
    a: FBig<HalfEven>,
    /// `B := sin(theta)`.
    b: FBig<HalfEven>,
    /// `C := (1 - delta/2)*B`, precomputed.
    c: FBig<HalfEven>,
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
        let one = prec.ib(IBig::ONE);
        let c = (&one - (delta / &two)) * &b;

        let scale_to_real = scale.to_real(prec);
        let ellipse = tight_cap_ellipse(prec, &a, &b, &c, &scale_to_real, &z_x, &z_y);

        Self {
            scale,
            a,
            b,
            c,
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
        // A*x*y - B*x^2 + C <= 0 -- see the struct docs for the derivation.
        let lhs = (&self.a * (&x * &y)) - (&self.b * (&x * &x)) + &self.c;
        lhs <= prec.ib(IBig::ZERO)
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
        //   A*x(t)*y(t) - B*x(t)^2 + C <= 0
        let q2 = &xv * ((&self.a * &yv) - (&self.b * &xv));
        let q1 = (&self.a * ((&x0 * &yv) + (&xv * &y0))) - (2 * &self.b * &x0 * &xv);
        let q0 = (&self.a * &x0 * &y0) - (&self.b * &x0 * &x0) + &self.c;
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

/// Hard ceiling on how many `k`-steps [`search_best_over_rotation`] will examine before
/// giving up, regardless of working precision. This is a deliberate, *practical* latency
/// bound, not a correctness one: Bothe's own asymptotic win comes from finding a genuinely
/// *rare* over-rotation candidate (low probability `p`, cheap only when it happens to be
/// sampled) -- and rare-in-the-lattice can mean the search must reach a high `k` before any
/// lattice point happens to land inside the (correctly tight, but still sparse) region.
/// Measured directly (`SmallAngleRegion`'s own tight ellipse): per-`k` cost in this search
/// grows roughly 4x per step once it starts climbing (`k=10`: ~140ms, `k=11`: ~590ms, `k=12`:
/// ~2.1s, `k=13`: ~8.4s, in an unoptimized build) -- and the *degenerate* (`A <= 0`, full-disk
/// fallback ellipse) case is measurably worse again, since a bigger box means more work per
/// `k` even before any of it passes the exact containment check. This growth is inherent to
/// searching for a rare candidate at depth `k` -- confirmed by forcing `even_split_search` (a
/// different, already-fast protocol) to a comparably deep `k` via an artificially tiny
/// `epsilon`, where it stays fast (finding *something* quickly, since its region's area only
/// depends on `epsilon`, not on rarity). `k=9` is chosen with real margin below where the
/// tight-ellipse case starts climbing steeply, so the degenerate case's worse per-step cost
/// still lands in a bounded, sub-second-to-low-single-digit-second regime rather than the
/// tight case's own margin being immediately eaten by it.
/// Tightening [`SmallAngleRegion`]'s bounding ellipse (see its own docs) fixes the cases that
/// only needed a *tighter* search at low `k`; it cannot fix a case that genuinely needs a
/// *deeper* one -- no ellipse improves cost for the vast number of lattice points examined at
/// high `k` before any one of them happens to fall inside a small region, since "candidates
/// examined" here is dominated by the underlying 1-D grid search's own per-`k` growth, not by
/// how many pass the final containment check (confirmed directly: cost still grows this way
/// even while zero candidates are found at any given `k`). Bothe's own 262-CPU-hour cost to
/// exhaustively build their static table (`small_angle_table`) is independent evidence this
/// kind of live search is intrinsically expensive whenever it must go deep.
///
/// Capping `k` here trades away the T-count win on cases that need a deeper search than this
/// (falling back to [`crate::protocol::mixed_diagonal::even_split_search`] instead, via
/// [`synth_small_angle`] returning `None`) in exchange for a bounded, predictable worst-case
/// latency -- the tradeoff a compiler pipeline needs. This never costs correctness: every
/// caller already treats `None` here as "use the always-correct even-split protocol instead"
/// (see [`crate::protocol::mixed_diagonal::synth_mixed_diagonal`]), not a hard failure.
const MAX_LIVE_SEARCH_K: i64 = 9;

/// Searches for the over-rotation branch: candidates inside `region` (so `Im(w) <= 0` by
/// construction), scored against the fixed `(hi_re, hi_im)` branch (the identity, in every
/// caller here) by [`mixture_weight`]'s `p`, keeping the one minimizing `p * T-count` (the
/// *mean* cost of the resulting mixture, since the `hi` branch's own T-count is 0 whenever
/// it is the identity). Returns `None` if no candidate is found within
/// [`MAX_LIVE_SEARCH_K`] -- an EXPECTED outcome for `(theta, delta)` pairs whose best
/// over-rotation lies deeper than this bound allows, not a bug; callers treat it as "fall
/// back to [`crate::protocol::mixed_diagonal::synth_mixed_diagonal`]'s always-correct
/// even-split path" (see [`synth_small_angle`]).
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
    let max_k = MAX_LIVE_SEARCH_K;
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

    if best.is_none() {
        warn!(
            "search_best_over_rotation: no over-rotation candidate found within \
             MAX_LIVE_SEARCH_K={MAX_LIVE_SEARCH_K} k-steps (first_hit_k={first_hit_k:?}); \
             falling back to the even-split protocol"
        );
    }

    best.map(|(_, u)| u)
}

/// Tries every row of Bothe's static over-rotation table
/// ([`crate::protocol::small_angle_table::OVER_ROTATION_TABLE`]) as a candidate `lo` (over-
/// rotation) branch against the fixed `(hi_re, hi_im)` branch (the identity, in every caller
/// here), and returns the `(gates, p)` of whichever row achieves the lowest mean cost `p *
/// T-count` among those meeting `delta` -- or `None` if no row meets it.
///
/// This is a fast path *in front of* [`search_best_over_rotation`]'s live lattice search, not
/// a replacement for it: the table's ~55 rows are exact, theta-independent gate words, so
/// checking every one costs 55 calls to [`mixture_weight`] on already-known ring elements --
/// no `solve_tdgp` enumeration at all. It resolves the common case (the table's `tan(alpha)`
/// staircase covers most of the accuracy range Bothe's Fig. 1 plots) essentially instantly;
/// callers fall back to the live search only when this returns `None` (delta tighter than
/// every table row achieves, or the search landed outside the table's covered angle range).
/// This is deliberately a heuristic, not a claim of global optimality: a row not in the table
/// (T-count > 35, or one of the 25 rows Table III expresses in an undocumented Clifford
/// labeling this crate cannot decode -- see `small_angle_table`'s module docs) might in
/// principle beat every row that IS checked here, in which case the live search (run only on
/// a table miss) is what finds it.
fn table_best_over_rotation(
    prec: Prec,
    wframe: &WFrame,
    hi_re: &FBig<HalfEven>,
    hi_im: &FBig<HalfEven>,
    delta: &FBig<HalfEven>,
) -> Option<(GateSeq, FBig<HalfEven>)> {
    let zero = prec.ib(IBig::ZERO);
    let one = prec.ib(IBig::ONE);
    let mut best: Option<(FBig<HalfEven>, GateSeq, FBig<HalfEven>)> = None;

    for i in 0..OVER_ROTATION_TABLE.len() {
        let gates = row_gates(i);
        let u = DOmegaUnitary::from_gates(&gates);
        let re_lo = wframe.re_w(u.z());
        let im_lo = wframe.im_w(u.z());
        // Same `x >= 0, y <= 0` sign constraint `SmallAngleRegion::inside` enforces for the
        // live search: without `Re(lo) >= 0`, `mixture_weight`'s `p` is not guaranteed to lie
        // in `[0, 1]` (its derivation assumes `Re(lo)*Im(lo) <= 0`, which needs both this AND
        // `im_lo <= 0`) -- a fixed, theta-independent table row can and does land with the
        // wrong sign of `Re` for a given theta, so this check is load-bearing, not defensive.
        if im_lo > zero || re_lo < zero {
            continue;
        }
        let Some(mw) = mixture_weight(prec, (&re_lo, &im_lo), (hi_re, hi_im)) else {
            continue;
        };
        if !(zero.clone()..=one.clone()).contains(&mw.p) {
            continue;
        }
        if mw.projective_diamond_error > *delta {
            continue;
        }
        let t_count = prec.ib(IBig::from(gates.t_count() as i64));
        let score = &mw.p * &t_count;
        let is_better = match &best {
            None => true,
            Some((best_score, ..)) => score < *best_score,
        };
        if is_better {
            best = Some((score, gates, mw.p));
        }
    }

    best.map(|(_, gates, p)| (gates, p))
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

    let config = config_from_theta_epsilon(theta_pos, epsilon_diamond, seed, verbose, false);
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

    // Fast path: check Bothe's static over-rotation table before running the (comparatively
    // expensive) live lattice search -- see `table_best_over_rotation`'s own docs. Only used
    // when it actually meets `delta` AND its mean cost (`p * T-count`) doesn't exceed a cheap
    // estimate of what the angle-independent even-split protocol would cost
    // (`1.52*log2(1/delta) - 0.01`, this crate's own already-measured mixed-diagonal slope --
    // see `mixed_diagonal::tests::slope_fit_cost_vs_log2_inv_epsilon`). Without that second
    // check, a table candidate that merely *meets budget* (but isn't actually a good choice
    // for this specific theta -- the table's ~55 rows are theta-independent, so for some
    // (theta, delta) none of them is close to optimal) could regress mean cost below what
    // even-split already achieves for free, which is exactly the risk `table_best_over_rotation`'s
    // own docs flag ("a heuristic, not a claim of global optimality"). A table miss, or a
    // table hit that fails this cheap comparison, falls through to the live search below
    // exactly as if this fast path didn't exist -- the live search's own result is never
    // worse than what shipped before this fast path was added.
    if let Some((lo_gates, p)) = table_best_over_rotation(prec, &wframe, &hi_re, &hi_im, &delta) {
        let mean_t_count = fbig_to_f64(&p) * lo_gates.t_count() as f64;
        let even_split_cost_estimate = (1.52 * (1.0 / epsilon_diamond).log2() - 0.01).max(0.0);
        if mean_t_count <= even_split_cost_estimate {
            let one = prec.ib(IBig::ONE);
            let zero = prec.ib(IBig::ZERO);
            let result = if p == zero {
                MixedDiagonalResult::Exact {
                    gates: GateSeq::identity(),
                    prec,
                }
            } else if p == one {
                MixedDiagonalResult::Exact {
                    gates: lo_gates,
                    prec,
                }
            } else {
                MixedDiagonalResult::Mixed {
                    p,
                    lo: lo_gates,
                    hi: GateSeq::identity(),
                    prec,
                }
            };
            return Some(apply_x_conjugation(result, need_x_conjugate));
        }
        // Table's best qualifying candidate isn't good enough to trust outright; let the live
        // search below do its own, theta-tuned job instead -- same as a table miss.
    }

    synth_small_angle_live_search(config, prec, delta, wframe, hi, need_x_conjugate)
}

/// The live lattice-search fallback tail of [`synth_small_angle`]: builds
/// [`SmallAngleRegion`], runs [`search_best_over_rotation`], and assembles the final result.
/// Factored out so it can be called both when [`table_best_over_rotation`] finds nothing
/// usable and (identically) when it never runs at all.
fn synth_small_angle_live_search(
    mut config: GridSynthConfig,
    prec: Prec,
    delta: FBig<HalfEven>,
    wframe: WFrame,
    hi: DOmegaUnitary,
    need_x_conjugate: bool,
) -> Option<MixedDiagonalResult> {
    let scale = ZRootTwo::new(IBig::from(1), IBig::from(0));
    let region = SmallAngleRegion::new(prec, &config.theta, &delta, scale.clone());
    let unit_disk = UnitDisk::new(prec, scale);
    let transformed =
        setup_regions_and_transform(&region, &unit_disk, config.verbose, config.measure_time);

    let hi_re = wframe.re_w(hi.z());
    let hi_im = wframe.im_w(hi.z());
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

/// Cheap, O(1), search-free pre-check: is it even *plausible* that pinning the identity as
/// one branch could beat the even-split protocol for this `(theta, delta)`?
///
/// [`SmallAngleRegion`]'s own accuracy condition is only satisfiable (for the `x >= 0, y <=
/// 0` branch this module searches) when its `A := delta - 2*sin(theta/2)^2` coefficient is
/// positive -- below that, the region is degenerate (see the module docs on the `delta <=
/// theta^2`-ish regime where the identity stops being the best under-rotation). Using the
/// identity `2*sin(x/2)^2 = 1 - cos(x)`, and noting `cos` is even and `2*pi`-periodic (so
/// this needs none of [`synth_small_angle`]'s own `theta` canonicalization to evaluate
/// correctly), that condition is exactly `delta > 1 - theta.cos()`.
///
/// This is a **performance-only** heuristic, not a correctness gate: [`synth_mixed_diagonal`]
/// (the caller) always falls back to the always-correct even-split search if this returns
/// `true` but the small-angle search then fails to find anything, and both search paths
/// independently guarantee meeting `delta` regardless of which one runs. Deliberately does
/// *not* attempt to be more precise than this single closed-form check (e.g. by also running
/// a cheap cost *estimate* for both paths and comparing) -- the whole point is to spend zero
/// extra search cost decided which protocol to use.
///
/// [`synth_mixed_diagonal`]: crate::protocol::mixed_diagonal::synth_mixed_diagonal
pub(crate) fn small_angle_could_help(theta: f64, delta: f64) -> bool {
    delta > 1.0 - theta.cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracy::AchievedDiamondError;
    use dashu_base::Approximation;
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

    // ---- Region correctness ----

    // The region's own (corrected) derivation, re-checked independently against the exact
    // algebraic condition `A*x*y - B*x^2 + C <= 0`, without needing to construct an actual
    // ring element.
    #[test]
    fn small_angle_region_boundary_matches_derivation() {
        let theta = to_fbig(PREC, 0.05);
        let delta = to_fbig(PREC, 0.01);
        let scale = ZRootTwo::from_int(IBig::from(1_000_000));
        let region = SmallAngleRegion::new(PREC, &theta, &delta, scale);

        let condition = |x: &FBig<HalfEven>, y: &FBig<HalfEven>| -> FBig<HalfEven> {
            (&region.a * (x * y)) - (&region.b * (x * x)) + &region.c
        };

        // x=0 must ALWAYS violate the accuracy condition, regardless of y: at x=0 the
        // condition reduces to `C <= 0`, and `C = (1-delta/2)*B` is strictly positive for
        // any delta < 2. This is exactly the bug the corrected derivation fixes -- the
        // earlier, unsound (`r=1`-assuming) formula spuriously admitted x=0 as satisfying
        // the accuracy budget (its achieved error is actually close to 2, the worst
        // possible value, since a candidate with x=0 has |z| far from 1).
        let x0 = PREC.ib(IBig::ZERO);
        let y_arbitrary = to_fbig(PREC, -0.5);
        assert!(
            condition(&x0, &y_arbitrary) > PREC.ib(IBig::ZERO),
            "x=0 must never satisfy the accuracy condition"
        );

        // A candidate genuinely close to the target (x close to 1, y close to 0) must
        // satisfy it.
        let x_close = to_fbig(PREC, 0.9999);
        let y_close = to_fbig(PREC, -1e-4);
        assert!(
            condition(&x_close, &y_close) <= PREC.ib(IBig::ZERO),
            "a candidate close to target should satisfy the accuracy condition"
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
        // theta values comfortably inside delta >> theta^2 (delta=1e-4, so sqrt(delta)=1e-2
        // is the pre-check boundary -- these stay well clear of it, at ratios delta/theta^2
        // of ~100x/1111x/10000x, rather than sitting right at the boundary itself, which
        // makes for a much more expensive (near-degenerate) search without testing anything
        // the other, more targeted boundary tests don't already cover. 3e-3 (originally
        // included here) was dropped: confirmed, while adding `MAX_LIVE_SEARCH_K`, to no
        // longer find a candidate within that depth cap -- a good over-rotation candidate
        // can be rare in the lattice (see that constant's own doc comment), so not every
        // small theta finds one quickly even deep in this regime.
        for theta in [1e-3_f64, 3e-4, 1e-4] {
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
    // delta=1e-4 (delta >> theta^2 = 1e-6). Compares against `even_split_search` directly
    // (the pure even-split protocol), NOT the now-dispatching `synth_mixed_diagonal` -- at
    // this (theta, delta), `small_angle_could_help` is true, so `synth_mixed_diagonal` would
    // just be this same small-angle result, making the comparison vacuous.
    #[test]
    fn small_angle_beats_mixed_diagonal_for_small_theta() {
        let theta = 1e-3;
        let delta = 1e-4;
        assert!(
            small_angle_could_help(theta, delta),
            "test premise: this (theta, delta) should be in the small-angle-eligible regime"
        );
        let mixed = crate::protocol::mixed_diagonal::even_split_search(theta, delta, 7, false);
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

    // Slope fit: Bothe Eq. (66) predicts, in the delta >> theta^2 regime, a power law
    // T ~ 2*(theta^2/delta)*log2(...) -- i.e. roughly LINEAR in 1/delta (up to a slowly
    // varying log factor), sharply different from the angle-independent protocols' T ~
    // log2(1/delta). This fits the log-log slope of mean T-count against 1/delta (averaged
    // over several angles per delta, to smooth the discreteness of small T-counts, mirroring
    // `mixed_diagonal::tests::slope_fit_cost_vs_log2_inv_epsilon`'s own averaging) and checks
    // it lands near 1, not near 0 (which would indicate the region was accidentally behaving
    // like the angle-independent regime instead). It also checks, for every individual
    // (theta, delta) sample point, that the achieved diamond-norm error actually meets its
    // own delta -- the T-count savings are worthless if the accuracy criterion silently
    // slipped.
    #[test]
    fn small_angle_slope_fit_and_per_point_accuracy() {
        use crate::accuracy::AchievedDiamondError as _;

        // theta values small enough that theta^2 (~1e-8) is far below every delta tried
        // below, so every sample point stays in Bothe's delta >> theta^2 regime. The deltas
        // are chosen BELOW theta itself (not just above theta^2): delta >= theta would let
        // the identity alone satisfy the budget trivially (T-count 0 for every angle),
        // degenerating the fit -- see the comment on `synth_small_angle`'s identity fast
        // path. Genuine mixing requires theta > 2*sin(theta/2) ~= theta > delta.
        //
        // Kept intentionally small (1 angle x 2 deltas, not a wider sweep) and using
        // theta=1e-4 (not the originally much-smaller angles this test used before
        // `MAX_LIVE_SEARCH_K` existed): a good "cheap over-rotation" candidate is *rare* in
        // the lattice (see that constant's own doc comment), so the live search needs enough
        // `k`-depth to find one, and that depth doesn't shrink just because theta and delta
        // both shrink together -- empirically, theta=1e-4 at delta in {3e-5, 1e-5} (both
        // strictly below theta, avoiding the identity-trivial boundary noted above) was
        // confirmed, while adding the search-depth cap, to still find a candidate within it,
        // giving a genuine ~3x delta spread without needing an unbounded search.
        let thetas: [f64; 1] = [1e-4];
        let deltas = [3e-5_f64, 1e-5];

        let mut mean_cost = Vec::with_capacity(deltas.len());
        for (delta_idx, &delta) in deltas.iter().enumerate() {
            let mut total_cost = 0.0;
            for (i, &theta) in thetas.iter().enumerate() {
                let seed = 20_000 + (delta_idx * 1000 + i) as u64;
                let result = synth_small_angle(theta, delta, seed, false).unwrap_or_else(|| {
                    panic!("expected a result for theta={theta}, delta={delta}")
                });
                let prec = prec_of(&result);

                // Per-point accuracy criterion: the achieved diamond-norm error, recomputed
                // independently from the result's own public gate word(s) (not from any
                // internal search state), must not exceed the requested delta. A small
                // floating-point slack accounts for the same independently-rounded-trig
                // noise `mixed_diagonal`'s own tests document (`safe_tol_bits`/`approx_eq`
                // patterns above) -- this is a hard accuracy check, not a fit, so the slack
                // is kept tight (1%).
                let theta_fbig = to_fbig(prec, theta);
                let delta_fbig = to_fbig(prec, delta);
                let achieved = result.achieved_diamond_error(&theta_fbig);
                assert!(
                    achieved <= delta_fbig.clone() * to_fbig(prec, 1.01),
                    "theta={theta}, delta={delta}: achieved diamond error {achieved} exceeds \
                     the requested budget {delta_fbig}"
                );

                total_cost += fbig_to_f64(&result.expected_t_count());
            }
            mean_cost.push(total_cost / thetas.len() as f64);
        }

        let log_inv_delta: Vec<f64> = deltas.iter().map(|d| (1.0_f64 / d).ln()).collect();
        let log_cost: Vec<f64> = mean_cost.iter().map(|c| c.max(1e-12).ln()).collect();
        let slope = (log_cost[log_cost.len() - 1] - log_cost[0])
            / (log_inv_delta[log_inv_delta.len() - 1] - log_inv_delta[0]);

        eprintln!(
            "small-angle slope fit: mean_cost={mean_cost:?} at delta={deltas:?} -> log-log \
             slope vs 1/delta = {slope:.4} (Bothe Eq. 66 predicts ~1; angle-independent \
             protocols would show ~0 on this log-log-vs-1/delta axis)"
        );

        assert!(
            slope > 0.5 && slope < 1.5,
            "measured log-log slope {slope:.4} is far from the ~1 the theta^2/delta power \
             law (Bothe Eq. 66) predicts; mean_cost={mean_cost:?}"
        );
    }

    // Regime switch: for delta << theta^2 (outside the small-angle-eligible regime), the
    // dispatching `synth_mixed_diagonal` must behave *identically* to the pure even-split
    // search -- the whole point of `small_angle_could_help`'s pre-check is to add zero cost
    // and zero behavior change outside the regime where it can plausibly help.
    #[test]
    fn dispatch_matches_even_split_outside_small_angle_regime() {
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..3 {
            let theta: f64 = rng.random_range(0.3..2.0); // NOT small relative to typical delta
            let delta = 1e-6; // delta << theta^2 here
            assert!(
                !small_angle_could_help(theta, delta),
                "test premise: this (theta, delta) should be outside the small-angle regime"
            );
            let even_split =
                crate::protocol::mixed_diagonal::even_split_search(theta, delta, 1, false);
            let dispatched =
                crate::protocol::mixed_diagonal::synth_mixed_diagonal(theta, delta, 1, false);
            assert!(
                (fbig_to_f64(&dispatched.expected_t_count())
                    - fbig_to_f64(&even_split.expected_t_count()))
                .abs()
                    < 1e-9,
                "theta={theta}: dispatching synth_mixed_diagonal should exactly match \
                 even_split_search outside the small-angle regime"
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
        // Since `synth_small_angle` is called directly (bypassing the `could_help` pre-check
        // that would otherwise route a theta this large away from the expensive search),
        // `(theta, delta)` here is chosen to be one of the pairs confirmed, while adding
        // `MAX_LIVE_SEARCH_K`, to actually find a candidate within that depth cap -- an
        // arbitrary small theta/delta pair is not guaranteed to (see that constant's own doc
        // comment on why a good over-rotation candidate can be rare in the lattice).
        let theta = 1e-4_f64;
        let delta = 1e-5;
        let a = synth_small_angle(theta, delta, 5, false).expect("expected a result");
        let b = synth_small_angle(theta - 2.0 * PI, delta, 5, false).expect("expected a result");
        let a_t = fbig_to_f64(&a.expected_t_count());
        let b_t = fbig_to_f64(&b.expected_t_count());
        assert!(
            (a_t - b_t).abs() < 1e-6,
            "theta and theta-2*pi should synthesize to the same T-count: {a_t} vs {b_t}"
        );
    }

    // At a REALISTIC diamond-norm accuracy target (delta <= 1e-6, not the delta ~ 0.1-1
    // curiosities the adversarial-theta test above uses), `1 - cos(theta) < delta` reduces to
    // `theta < sqrt(2*delta)` -- confirmed exactly here: at delta=1e-6, the predicted boundary
    // is sqrt(2e-6)~=1.4142e-3, and `theta=1.4001e-3` (0.99x) measures `could_help=true` while
    // `theta=1.4284e-3` (1.01x) measures `could_help=false`. Crucially, `theta=pi/4` and
    // `theta=1.0` -- the "adversarial, moderately large theta" cases the other test exercises
    // -- are `could_help=false` at this realistic delta: that scenario is real math, but it
    // describes a delta regime (~0.1-1) no actual accuracy target lives in. At realistic
    // delta, the win is exactly the "headline" small-theta Bothe/Kliuchnikov result: large,
    // monotonically growing as theta shrinks. (A single, small theta is used below, not a
    // sweep across the boundary, to keep this test's runtime reasonable: each search here
    // costs on the order of a minute once `SmallAngleRegion` was corrected to the general
    // accuracy condition, since the old, buggy region's spurious `x~=0` shortcut is what made
    // this kind of search look artificially cheap before.)
    #[test]
    fn realistic_delta_boundary_matches_sqrt_2_delta_and_wins_throughout() {
        use crate::accuracy::AchievedDiamondError as _;
        let delta = 1e-6_f64;
        let boundary = (2.0 * delta).sqrt();

        assert!(small_angle_could_help(0.99 * boundary, delta));
        assert!(!small_angle_could_help(1.01 * boundary, delta));
        // The "adversarial theta" test's regime is unreachable at this realistic delta.
        assert!(!small_angle_could_help(std::f64::consts::PI / 4.0, delta));
        assert!(!small_angle_could_help(1.0, delta));

        let theta = boundary * 1e-2;
        let small = synth_small_angle(theta, delta, 1, false)
            .unwrap_or_else(|| panic!("expected a result for theta={theta}"));
        let even = crate::protocol::mixed_diagonal::even_split_search(theta, delta, 1, false);
        let prec = prec_of(&small);
        let theta_fbig = to_fbig(prec, theta);

        let small_achieved = fbig_to_f64(&small.achieved_diamond_error(&theta_fbig));
        let even_achieved = fbig_to_f64(&even.achieved_diamond_error(&theta_fbig));
        assert!(
            small_achieved <= delta * 1.5,
            "theta={theta}: small-angle achieved error {small_achieved} exceeds budget"
        );
        assert!(
            even_achieved <= delta * 1.5,
            "theta={theta}: even-split achieved error {even_achieved} exceeds budget"
        );

        let small_cost = fbig_to_f64(&small.expected_t_count());
        let even_cost = fbig_to_f64(&even.expected_t_count());
        assert!(
            small_cost < even_cost,
            "theta={theta}: small-angle cost {small_cost} should beat even-split cost {even_cost}"
        );
    }

    // A sweep across 6 angles (0.05 through 3.1, covering nearly the whole canonicalized
    // `[0, pi]` range) x several delta-factors each found no case where the small-angle path
    // cost more than `even_split_search` when `could_help == true`. Two things had to be
    // checked carefully before trusting that, though, both caught while investigating this
    // test (see the two points below) -- not merely T-count comparisons, since a bare
    // `expected_t_count()` reading can be trivially (and misleadingly) 0 on *both* sides at
    // once at very loose deltas:
    //
    // 1. For theta > ~1.33 rad, `1 - cos(theta)` (this crate's pre-check threshold) exceeds
    //    `2*sin(pi/8) ~= 0.765` -- the worst-case diamond distance from ANY target angle to
    //    the nearest exactly-Clifford-representable point (multiples of pi/2; confirmed by
    //    this crate's own `mixed_diagonal` tests as the only exact 0-T-count directions).
    //    So for theta above that, `could_help == true` only ever admits deltas already loose
    //    enough that a single Clifford point trivially wins on its own, regardless of which
    //    protocol is used -- there is no genuinely large-theta case to test past that point,
    //    not because the protocol is secretly fine there, but because the pre-check's own
    //    threshold structurally never engages a meaningful comparison for such theta.
    // 2. The genuinely largest, most adversarial NON-trivial case is theta = pi/4: the exact
    //    midpoint between two Clifford grid points, maximizing both the identity's own
    //    distance to target AND the width of the non-trivial `could_help == true` window
    //    before that 0.765 floor is reached. There, `even_split_search` needs real work
    //    (verified via its own `achieved_diamond_error`, not just T-count: 4 T-gates at
    //    achieved error 0.0366, dropping to 2 T-gates at achieved error 0.7654 once delta
    //    exceeds that floor) while `synth_small_angle` finds a 0-T-count mixture (identity
    //    with `S`) whose own achieved error (0.2929, independently verified) already beats
    //    every delta in the swept range -- a large, genuine, verified win, not an artifact.
    #[test]
    fn small_angle_never_worse_than_even_split_when_pre_check_says_yes() {
        use crate::accuracy::AchievedDiamondError as _;

        // (theta, delta) pairs spanning small/moderate/"as large as meaningfully testable"
        // theta, each just inside the `could_help == true` side of the boundary.
        // theta=pi/4 is the adversarial case from point 2 above. Kept to 3 cases (not the 6
        // originally used to explore this): each search here costs on the order of a minute
        // once `SmallAngleRegion` was corrected to the general accuracy condition, since the
        // old, buggy region's spurious `x~=0` shortcut is what made this look far cheaper.
        let pi_4 = std::f64::consts::PI / 4.0;
        let cases: [(f64, f64); 3] = [
            (0.05, 1.0001 * (1.0 - 0.05_f64.cos())),
            (1.0, 1.1 * (1.0 - 1.0_f64.cos())),
            (pi_4, 1.001 * (1.0 - pi_4.cos())),
        ];
        for (theta, delta) in cases {
            assert!(
                small_angle_could_help(theta, delta),
                "test premise violated: theta={theta}, delta={delta} should be in the \
                 pre-check's `could_help` regime"
            );
            let small = synth_small_angle(theta, delta, 1, false)
                .unwrap_or_else(|| panic!("expected a result for theta={theta}, delta={delta}"));
            let even = crate::protocol::mixed_diagonal::even_split_search(theta, delta, 1, false);
            let prec = prec_of(&small);
            let theta_fbig = to_fbig(prec, theta);

            // Verify BOTH results actually meet the requested budget (not just compare
            // T-counts) -- a cheap-but-wrong result would otherwise look like a "win".
            let small_achieved = fbig_to_f64(&small.achieved_diamond_error(&theta_fbig));
            let even_achieved = fbig_to_f64(&even.achieved_diamond_error(&theta_fbig));
            assert!(
                small_achieved <= delta * 1.01,
                "theta={theta}, delta={delta}: small-angle achieved error {small_achieved} \
                 exceeds its own budget"
            );
            assert!(
                even_achieved <= delta * 1.01,
                "theta={theta}, delta={delta}: even-split achieved error {even_achieved} \
                 exceeds its own budget"
            );

            let small_cost = fbig_to_f64(&small.expected_t_count());
            let even_cost = fbig_to_f64(&even.expected_t_count());
            assert!(
                small_cost <= even_cost + 1e-9,
                "theta={theta}, delta={delta}: small-angle cost {small_cost} exceeded \
                 even-split cost {even_cost} despite could_help == true"
            );
        }
    }

    // Negative angles: R_z(-theta) = X R_z(theta) X, so the X-conjugation path must produce
    // an achieved error consistent with the (negative) target, not the canonicalized one.
    #[test]
    fn negative_angle_achieves_its_own_target_after_x_conjugation() {
        // -0.05 (this test's original value) canonicalizes to theta_pos=0.05, which has
        // `A <= 0` at delta=1e-4 (outside `SmallAngleRegion`'s useful regime -- confirmed
        // `small_angle_could_help(0.05, 1e-4)` is false) and so, correctly, no longer finds a
        // candidate within `MAX_LIVE_SEARCH_K` (that combination previously only "succeeded"
        // via an effectively unbounded, multi-minute search). Using the negative counterpart
        // of one of the pairs already confirmed to work within the cap instead.
        let theta = -1e-4_f64;
        let delta = 1e-5;
        let result = synth_small_angle(theta, delta, 11, false).expect("expected a result");
        let prec = prec_of(&result);
        let theta_fbig = to_fbig(prec, theta);
        let achieved = result.achieved_diamond_error(&theta_fbig);
        assert!(
            achieved <= to_fbig(prec, delta) * to_fbig(prec, 1.5),
            "achieved error {achieved} for negative theta={theta} exceeds the budget"
        );
    }

    // `MAX_LIVE_SEARCH_K` is a deliberate latency bound (see its own doc comment): a
    // `(theta, delta)` pair whose best over-rotation needs a deeper search than that cap
    // allows must make `synth_small_angle` return `None` quickly (bounded time), NOT search
    // indefinitely -- and the dispatcher (`synth_mixed_diagonal`) must then fall back to the
    // always-correct even-split protocol and still meet the requested budget. `theta=7e-6,
    // delta=1e-8` is squarely inside `small_angle_could_help`'s regime (A > 0, confirmed
    // directly) yet was measured, while diagnosing this bound, to find zero candidates through
    // k=13 with per-k cost already at ~8s and growing ~4x per step -- i.e. it genuinely exceeds
    // `MAX_LIVE_SEARCH_K` rather than merely being outside the pre-check's regime.
    // `even_split_search` handles it comfortably since its region's area depends only on
    // `delta`, not on how rare a good identity-pinned candidate happens to be.
    #[test]
    fn deep_search_falls_back_to_even_split_within_a_bounded_time() {
        use crate::accuracy::AchievedDiamondError as _;
        let theta = 7e-6_f64;
        let delta = 1e-8_f64;

        let start = std::time::Instant::now();
        let result = crate::protocol::mixed_diagonal::synth_mixed_diagonal(theta, delta, 1, false);
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "synth_mixed_diagonal took {elapsed:?} for a case expected to exceed \
             MAX_LIVE_SEARCH_K and fall back to even-split -- the k-cap may not be working"
        );

        let prec = result.prec();
        let theta_fbig = to_fbig(prec, theta);
        let achieved = result.achieved_diamond_error(&theta_fbig);
        assert!(
            achieved <= to_fbig(prec, delta) * to_fbig(prec, 1.01),
            "achieved error {achieved} exceeds the requested budget {delta} even via the \
             even-split fallback"
        );
    }
}
