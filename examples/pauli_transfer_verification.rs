// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Independent numerical verification that the synthesized channels actually meet their
//! requested diamond-norm accuracy budget -- entirely independent of this crate's own
//! closed-form `mixture_weight`/`diagonal_diamond_distance` formulas (this check does not
//! call either; it only reads final gate strings and probabilities and recomputes everything
//! from scratch).
//!
//! ## Two different, non-interchangeable ways to measure "how far" a channel is from target
//!
//! **Single Z-rotation vs. target Z-rotation** (`single_rotation_diamond_distance`): exact,
//! closed form, `2*sqrt(1-Re(w)^2)`. Always valid for comparing one pure rotation to another.
//!
//! **Pauli transfer matrix** (`ptm`/`pauli_diamond_distance_from_branches`): computes
//! `Sum_P |delta_q_P|` from a channel's Pauli-basis decomposition. This closed form is only
//! mathematically valid when the *difference* between the channel and the target is itself
//! diagonal in the Pauli basis -- true for mixed diagonal's output (the {Z,S} twirl exists
//! specifically to enforce this by symmetrizing the off-diagonal X/Y error), but NOT true in
//! general for an untwisted single rotation mixed with something else: a lone Z-rotation by a
//! small angle delta genuinely mixes X and Y in its Pauli transfer matrix (that mixing IS what
//! the twirl is for), so applying the Pauli-diagonal shortcut to a channel that was never
//! twirled computes a meaningless number. An earlier version of this file made exactly that
//! mistake for fallback/mixed fallback and reported achieved errors around 1.0 (nonsense) for
//! everything, subsequently reduced to a state that *looked* plausible but was still wrong,
//! because the shortcut simply does not apply to the untwisted projective branch.
//!
//! ## The correct decomposition for fallback / mixed fallback
//!
//! Per the paper's own circuit description, the "success" outcome of the ancilla+measurement
//! circuit (out of scope for this crate) is an idealized rotation by `Arg(z)` -- not the raw
//! synthesized unitary's full matrix, whose off-diagonal component the projection removes.
//! The "failure" outcome is an idealized rotation by `Arg(v)`, followed by the ordinary
//! (non-projected) correction gate `B`. Since `B` is applied as an ordinary unconditional
//! unitary after a *fixed* rotation, and diamond norm is invariant under composing both sides
//! of a difference with the same fixed unitary channel, `B`'s own accuracy is exactly its
//! distance to the *residual* target angle `theta - Arg(v)` -- which is a fair question to ask
//! via the Pauli-diagonal shortcut, because `B` (a mixed-diagonal result) genuinely IS
//! twirled internally.
//!
//! So the right check is a **triangle-inequality upper bound**, combining an exact formula for
//! the untwisted success branch with a Pauli-diagonal-valid check for the twirled correction:
//!   `‖q*Z_{Arg z} + (1-q)*(B o Z_{Arg v}) - Z_theta‖⋄`
//!   `<= q * ‖Z_{Arg z} - Z_theta‖⋄  +  (1-q) * ‖B - Z_{theta-Arg(v)}‖⋄`
//! Both right-hand terms are independently exact/valid; the sum is a genuine, conservative
//! upper bound on the true (harder to compute exactly without twirling) diamond distance. If
//! this bound is <= the requested budget, that is a rigorous pass; if the bound exceeds the
//! budget, it does not by itself prove a failure (the true distance could still be smaller),
//! but it does mean this check cannot certify a pass and warrants a tighter look.

use dashu_base::Approximation;
use num::Complex;
use rsgridsynth::clear_caches;
use rsgridsynth::protocol::fallback::exact_q;
use rsgridsynth::protocol::{
    synth_fallback, synth_mixed_diagonal, synth_mixed_fallback, FallbackResult,
    MixedDiagonalResult, MixedFallbackResult, MixedFallbackSide,
};
use rsgridsynth::unitary::DOmegaUnitary;

type C = Complex<f64>;
type M2 = [[C; 2]; 2];

fn to_f64(x: &dashu_float::FBig<dashu_float::round::mode::HalfEven>) -> f64 {
    match x.to_f64() {
        Approximation::Exact(v) => v,
        Approximation::Inexact(v, _) => v,
    }
}

fn matrix_from_gates(gates: &str) -> M2 {
    let m = DOmegaUnitary::from_gates(gates).to_complex_matrix();
    [
        [
            Complex::new(to_f64(&m[(0, 0)].re), to_f64(&m[(0, 0)].im)),
            Complex::new(to_f64(&m[(0, 1)].re), to_f64(&m[(0, 1)].im)),
        ],
        [
            Complex::new(to_f64(&m[(1, 0)].re), to_f64(&m[(1, 0)].im)),
            Complex::new(to_f64(&m[(1, 1)].re), to_f64(&m[(1, 1)].im)),
        ],
    ]
}

fn mat_mul(a: &M2, b: &M2) -> M2 {
    let mut r = [[Complex::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut s = Complex::new(0.0, 0.0);
            for k in 0..2 {
                s += a[i][k] * b[k][j];
            }
            r[i][j] = s;
        }
    }
    r
}

fn mat_dagger(a: &M2) -> M2 {
    [
        [a[0][0].conj(), a[1][0].conj()],
        [a[0][1].conj(), a[1][1].conj()],
    ]
}

fn pauli_basis() -> [M2; 4] {
    let zero = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    let i = Complex::new(0.0, 1.0);
    let id = [[one, zero], [zero, one]];
    let x = [[zero, one], [one, zero]];
    let y = [[zero, -i], [i, zero]];
    let z = [[one, zero], [zero, -one]];
    [id, x, y, z]
}

/// Pauli transfer matrix `R[P][Q] = (1/2) Re(Tr[sigma_P * Lambda(sigma_Q)])` for
/// `Lambda(rho) = sum_k weight_k * U_k * rho * U_k^dagger`.
fn ptm(branches: &[(f64, M2)]) -> [[f64; 4]; 4] {
    let sigma = pauli_basis();
    let mut r = [[0.0; 4]; 4];
    for (q_idx, sq) in sigma.iter().enumerate() {
        let mut lambda_sq = [[Complex::new(0.0, 0.0); 2]; 2];
        for &(weight, ref u) in branches {
            let u_dagger = mat_dagger(u);
            let term = mat_mul(&mat_mul(u, sq), &u_dagger);
            for i in 0..2 {
                for j in 0..2 {
                    lambda_sq[i][j] += term[i][j] * weight;
                }
            }
        }
        for (p_idx, sp) in sigma.iter().enumerate() {
            let prod = mat_mul(sp, &lambda_sq);
            let trace = prod[0][0] + prod[1][1];
            r[p_idx][q_idx] = 0.5 * trace.re;
        }
    }
    r
}

/// Target channel `Z_theta(rho) = R_z(theta) rho R_z(theta)^dagger`,
/// `R_z(theta) = diag(e^{-i*theta/2}, e^{i*theta/2})` (this crate's convention).
fn target_matrix(theta: f64) -> M2 {
    let half = theta / 2.0;
    let zero = Complex::new(0.0, 0.0);
    [
        [Complex::new(half.cos(), -half.sin()), zero],
        [zero, Complex::new(half.cos(), half.sin())],
    ]
}

/// `delta_q_P` from the diagonal of a *difference* of two channels' PTMs (each channel's own
/// `q_I+q_X+q_Y+q_Z=1` normalization cancels exactly in a genuine difference, so only the
/// linear part applies -- verified in `debug_tests` below by checking a channel against
/// itself gives exactly 0, not the `1` a naive "reapply the affine +1 formula" mistake gives).
fn delta_q_from_ptm_diagonal_difference(diff_diag: [f64; 4]) -> [f64; 4] {
    let (xx, yy, zz) = (diff_diag[1], diff_diag[2], diff_diag[3]);
    [
        (xx + yy + zz) / 4.0,
        (xx - yy - zz) / 4.0,
        (-xx + yy - zz) / 4.0,
        (-xx - yy + zz) / 4.0,
    ]
}

/// Diamond distance of a channel to a target Z-rotation, valid ONLY when the channel is
/// genuinely (or was constructed to be, e.g. via a {Z,S} twirl) Pauli-diagonal relative to
/// that target -- see the module docs. Also returns the max absolute off-diagonal entry of
/// the difference PTM as a diagnostic: large values here mean the Pauli-diagonal precondition
/// does not hold and the returned "distance" should not be trusted.
fn pauli_diamond_distance_from_branches(branches: &[(f64, M2)], theta: f64) -> (f64, f64) {
    let actual = ptm(branches);
    let target = ptm(&[(1.0, target_matrix(theta))]);

    let mut diff = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            diff[i][j] = actual[i][j] - target[i][j];
        }
    }

    let mut max_off_diag = 0.0_f64;
    for (i, row) in diff.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            if i != j {
                max_off_diag = max_off_diag.max(value.abs());
            }
        }
    }

    let diag = [diff[0][0], diff[1][1], diff[2][2], diff[3][3]];
    let dq = delta_q_from_ptm_diagonal_difference(diag);
    let diamond_distance: f64 = dq.iter().map(|v| v.abs()).sum();
    (diamond_distance, max_off_diag)
}

/// `Re(z * e^{i*theta/2})`, matching `WFrame::re_w`'s convention exactly.
fn re_w_direct(z: C, theta: f64) -> f64 {
    let half = theta / 2.0;
    z.re * half.cos() - z.im * half.sin()
}

/// Exact diamond distance between a single IDEALIZED rotation by `Arg(z)` (i.e. the unit-
/// modulus phase `z/|z|`, NOT `z` itself -- for fallback/mixed fallback `|z|^2 = q < 1`, and
/// the idealized success action is a pure phase rotation, with the magnitude deficit `1-q`
/// entirely accounted for by the separate failure-branch weight, not by this term) and the
/// target rotation by `theta`: `2*sqrt(1-Re(w)^2)` with `w = (z/|z|) * e^{i*theta/2}`. Always
/// valid (no Pauli-diagonal precondition -- this is a direct unitary-vs-unitary comparison,
/// not a mixture). Feeding this the raw (non-unit-modulus) `z` instead would conflate the
/// magnitude deficit with angular error and give a nonsensical, epsilon-independent ~`2*sqrt(1-q)`
/// answer -- exactly the bug an earlier version of this file had, caught by comparing against
/// `plain_diagonal_epsilon_convention_calibration`'s epsilon-scaling expectation.
fn single_rotation_diamond_distance(z: C, theta: f64) -> f64 {
    let norm = (z.re * z.re + z.im * z.im).sqrt();
    let phase = z / norm;
    let re_w = re_w_direct(phase, theta).clamp(-1.0, 1.0);
    2.0 * (1.0 - re_w * re_w).max(0.0).sqrt()
}

fn fbig_to_f64_pub(x: &dashu_float::FBig<dashu_float::round::mode::HalfEven>) -> f64 {
    to_f64(x)
}

/// Exact diamond distance of a weighted combination of same-axis (Z-)rotations minus a
/// target Z-rotation, `‖Sum_k w_k*Z_{psi_k} - Z_theta‖⋄`. Derivation: any Z-rotation acts on
/// a density matrix's diagonal entries as the identity and on its off-diagonal entry `rho_01`
/// by multiplying by a single complex phase `e^{-2i*psi}` -- so the *difference* channel acts
/// as zero on the diagonal (regardless of weights, since each term individually fixes it) and
/// as multiplication by a single complex number
/// `c = Sum_k w_k*(e^{-2i*psi_k} - e^{-i*theta})` on the off-diagonal. For a qubit map of
/// exactly this form (diagonal-preserving, off-diagonal scaled by one fixed complex number),
/// the diamond norm equals `|c|` exactly, achieved by the maximally-coherent probe state --
/// verified against the ALREADY-independently-checked single-rotation formula
/// `2*|sin(delta)|` as the `w=[1.0]` special case in `debug_tests` below (both formulas must
/// agree there, and they do, in closed form: `|e^{-2i*delta}-1| = 2*|sin(delta)|` exactly).
///
/// This is what lets mixed fallback's projective-mixing cancellation (the whole point of the
/// protocol) be verified independently of the crate's own `mixture_weight` closed form: a
/// naive triangle-inequality bound on the two branches SEPARATELY (an earlier version of this
/// check) is too loose to see the cancellation at all, and wrongly looks like a failure.
fn rotation_mixture_diamond_distance(weighted_psis: &[(f64, f64)], theta: f64) -> f64 {
    let mut c = Complex::new(0.0, 0.0);
    let target_phase = Complex::new(theta.cos(), -theta.sin()); // e^{-i*theta}
    for &(w, psi) in weighted_psis {
        let phase = Complex::new((2.0 * psi).cos(), -(2.0 * psi).sin()); // e^{-2i*psi}
        c += w * (phase - target_phase);
    }
    (c.re * c.re + c.im * c.im).sqrt()
}

/// `-Arg(z)`, i.e. the `psi` such that the idealized rotation `diag(z/|z|, conj(z)/|z|)`
/// equals `diag(e^{-i*psi}, e^{i*psi})` -- matching the target's own `diag(e^{-i*theta/2},
/// e^{i*theta/2})` convention.
fn psi_from_z(z: C) -> f64 {
    -z.im.atan2(z.re)
}

fn mixed_diagonal_pauli_branches(result: &MixedDiagonalResult) -> Vec<(f64, M2)> {
    result
        .branches
        .iter()
        .map(|b| (fbig_to_f64_pub(&b.weight), matrix_from_gates(&b.gates)))
        .collect()
}

/// Triangle-inequality upper bound on the total diamond distance of a plain fallback result:
/// `q * single_rotation_distance(z, theta) + (1-q) * pauli_distance(B, theta - Arg(v))`.
/// Returns `(bound, correction_off_diagonal_diagnostic)`.
fn fallback_upper_bound(result: &FallbackResult, z: C, v: C, theta: f64) -> (f64, f64) {
    let q = fbig_to_f64_pub(&result.success_probability);
    let success_term = single_rotation_diamond_distance(z, theta);

    let residual_theta = theta - v.im.atan2(v.re);
    let b_matrix = matrix_from_gates(&result.correction_gates);
    let (correction_term, off_diag) =
        pauli_diamond_distance_from_branches(&[(1.0, b_matrix)], residual_theta);

    (q * success_term + (1.0 - q) * correction_term, off_diag)
}

/// Total diamond-distance bound for a mixed fallback result, correctly exploiting the
/// projective-mixing cancellation (see `rotation_mixture_diamond_distance`'s docs) instead of
/// a naive triangle-inequality split of the two sides' success terms, which is too loose to
/// see the cancellation at all and looks like a failure even for a correct implementation.
///
/// Groups the four underlying pieces (lo success, lo correction, hi success, hi correction)
/// as: `[p*q_lo*Z_{psi_lo} + (1-p)*q_hi*Z_{psi_hi} - Z_theta] + [p*(1-q_lo)*(B_lo-related)] +
/// [(1-p)*(1-q_hi)*(B_hi-related)]`. The first bracket has EXACTLY the same
/// "diagonal-preserving, single-complex-off-diagonal-scaling" structure
/// `rotation_mixture_diamond_distance` handles (the weights `p*q_lo`/`(1-p)*q_hi` need not
/// themselves sum to 1 -- the derivation there only used that each individual rotation fixes
/// the diagonal, which holds regardless of the weight). The other two brackets are each a
/// twirled mixed-diagonal correction (Pauli-diagonal-valid) composed with a fixed rotation
/// (removable by diamond-norm's unitary invariance), exactly as in `fallback_upper_bound`.
/// Combining the three brackets via the triangle inequality is a valid (if not perfectly
/// tight, since the theorem's own formula could in principle also correlate the correction
/// terms with the success terms) upper bound.
/// One side of a mixed fallback result, bundled with the `z`/`v` complex entries extracted
/// from its re-multiplied projective gate matrix (purely to keep
/// `mixed_fallback_total_bound`'s argument count clippy-clean).
struct MixedFallbackSideInput<'a> {
    side: &'a MixedFallbackSide,
    z: C,
    v: C,
}

fn mixed_fallback_total_bound(
    lo: &MixedFallbackSideInput,
    hi: &MixedFallbackSideInput,
    p: f64,
    theta: f64,
) -> (f64, f64) {
    let q_lo = fbig_to_f64_pub(&lo.side.success_probability);
    let q_hi = fbig_to_f64_pub(&hi.side.success_probability);

    let psi_lo = psi_from_z(lo.z);
    let psi_hi = psi_from_z(hi.z);
    let projective_term =
        rotation_mixture_diamond_distance(&[(p * q_lo, psi_lo), ((1.0 - p) * q_hi, psi_hi)], theta);

    let residual_theta_lo = theta - lo.v.im.atan2(lo.v.re);
    let (correction_lo, off_diag_lo) = pauli_diamond_distance_from_branches(
        &mixed_diagonal_pauli_branches(&lo.side.correction),
        residual_theta_lo,
    );

    let residual_theta_hi = theta - hi.v.im.atan2(hi.v.re);
    let (correction_hi, off_diag_hi) = pauli_diamond_distance_from_branches(
        &mixed_diagonal_pauli_branches(&hi.side.correction),
        residual_theta_hi,
    );

    let bound = projective_term
        + p * (1.0 - q_lo) * correction_lo
        + (1.0 - p) * (1.0 - q_hi) * correction_hi;
    (bound, off_diag_lo.max(off_diag_hi))
}

fn main() {
    let thetas = [
        0.1,
        3.0 * std::f64::consts::PI / 32.0,
        5.0 * std::f64::consts::PI / 32.0,
        7.0 * std::f64::consts::PI / 32.0,
        std::f64::consts::PI / 3.0,
        std::f64::consts::PI / 6.0,
        1.9,
        4.2,
    ];
    let epsilons = [1e-4, 1e-6, 1e-8];
    let q = exact_q(7);

    println!("protocol,theta,epsilon,epsilon_diamond_requested,achieved_diamond_distance_upper_bound,off_diagonal_diagnostic,meets_budget");

    for &eps in &epsilons {
        for (i, &theta) in thetas.iter().enumerate() {
            let seed = 2000 + i as u64;

            clear_caches();
            let md = synth_mixed_diagonal(theta, eps, seed, false);
            let branches = mixed_diagonal_pauli_branches(&md);
            let (dd, max_od) = pauli_diamond_distance_from_branches(&branches, theta);
            println!(
                "mixed_diagonal,{theta},{eps},{eps},{dd},{max_od},{}",
                dd <= eps
            );

            clear_caches();
            let sin_alpha = eps / 4.0;
            match synth_fallback(theta, eps, q.clone(), sin_alpha, seed, false) {
                Some(result) => {
                    let v_mat = matrix_from_gates(&result.projective_gates);
                    let z = v_mat[0][0];
                    let v = v_mat[1][0];
                    let (bound, off_diag) = fallback_upper_bound(&result, z, v, theta);
                    println!(
                        "fallback,{theta},{eps},{eps},{bound},{off_diag},{}",
                        bound <= eps
                    );
                }
                None => println!("fallback,{theta},{eps},{eps},ERROR,NotFound,false"),
            }

            clear_caches();
            match synth_mixed_fallback(theta, eps, q.clone(), seed, false) {
                Some(MixedFallbackResult::Exact { gates }) => {
                    let m = matrix_from_gates(&gates);
                    let (dd, off_diag) = pauli_diamond_distance_from_branches(&[(1.0, m)], theta);
                    println!(
                        "mixed_fallback,{theta},{eps},{eps},{dd},{off_diag},{}",
                        dd <= eps
                    );
                }
                Some(MixedFallbackResult::Mixed { lo, hi, p, .. }) => {
                    let p_f64 = fbig_to_f64_pub(&p);
                    let lo_v = matrix_from_gates(&lo.projective_gates);
                    let hi_v = matrix_from_gates(&hi.projective_gates);
                    let lo_input = MixedFallbackSideInput {
                        side: &lo,
                        z: lo_v[0][0],
                        v: lo_v[1][0],
                    };
                    let hi_input = MixedFallbackSideInput {
                        side: &hi,
                        z: hi_v[0][0],
                        v: hi_v[1][0],
                    };
                    let (bound, off_diag) =
                        mixed_fallback_total_bound(&lo_input, &hi_input, p_f64, theta);
                    println!(
                        "mixed_fallback,{theta},{eps},{eps},{bound},{off_diag},{}",
                        bound <= eps
                    );
                }
                None => println!("mixed_fallback,{theta},{eps},{eps},ERROR,NotFound,false"),
            }
        }
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn self_consistency_target_vs_itself() {
        let theta = 0.7_f64;
        let m = target_matrix(theta);
        let (dd, max_od) = pauli_diamond_distance_from_branches(&[(1.0, m)], theta);
        eprintln!("dd={dd}, max_od={max_od}");
        assert!(dd < 1e-9, "self-comparison should give ~0, got {dd}");
    }

    /// Locks in the derivation `delta_q_from_ptm_diagonal_difference` is built from: a pure
    /// identity channel's PTM diagonal is `[1,1,1,1]` (I,X,Y,Z all fixed) and decomposes to
    /// `q_I=1`, others 0; a pure-Z channel's PTM diagonal is `[1,-1,-1,1]` (Z fixes I and Z,
    /// negates X and Y) and decomposes to `q_Z=1`, others 0.
    #[test]
    fn q_from_ptm_diagonal_matches_known_pauli_channels() {
        fn q_from_ptm_diagonal(diag: [f64; 4]) -> [f64; 4] {
            let (xx, yy, zz) = (diag[1], diag[2], diag[3]);
            [
                (1.0 + xx + yy + zz) / 4.0,
                (1.0 + xx - yy - zz) / 4.0,
                (1.0 - xx + yy - zz) / 4.0,
                (1.0 - xx - yy + zz) / 4.0,
            ]
        }
        let identity = q_from_ptm_diagonal([1.0, 1.0, 1.0, 1.0]);
        assert!((identity[0] - 1.0).abs() < 1e-12);
        assert!(
            identity[1].abs() < 1e-12 && identity[2].abs() < 1e-12 && identity[3].abs() < 1e-12
        );

        let pure_z = q_from_ptm_diagonal([1.0, -1.0, -1.0, 1.0]);
        assert!((pure_z[3] - 1.0).abs() < 1e-12);
        assert!(pure_z[0].abs() < 1e-12 && pure_z[1].abs() < 1e-12 && pure_z[2].abs() < 1e-12);
    }

    #[test]
    fn identity_channel_vs_target_theta_zero() {
        let theta = 0.0_f64;
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        let id = [[one, zero], [zero, one]];
        let (dd, max_od) = pauli_diamond_distance_from_branches(&[(1.0, id)], theta);
        eprintln!("dd={dd}, max_od={max_od}");
        assert!(
            dd < 1e-9,
            "identity vs target(theta=0) should give ~0, got {dd}"
        );
    }

    /// `single_rotation_diamond_distance` cross-checked against the Pauli-diagonal formula
    /// for the special case where they SHOULD agree (a lone rotation IS trivially
    /// Pauli-diagonal relative to another rotation about the same axis when compared this
    /// way is not generally true off-diagonally, but the diamond DISTANCE value itself, for a
    /// single unitary vs a single target unitary, has a well-known closed form regardless --
    /// this test checks `single_rotation_diamond_distance` against the textbook value
    /// `2*|sin(delta/2)|`... actually the exact value for two Z-rotations by phi1, phi2 is
    /// `2*|sin((phi1-phi2)/2)|`; verify against a direct small example instead of re-deriving).
    /// `rotation_mixture_diamond_distance`'s single-weight special case must agree exactly
    /// with `single_rotation_diamond_distance` (both independently derived, one via the
    /// Pauli-transfer/witness argument, one via the direct `2*sqrt(1-Re(w)^2)` formula).
    #[test]
    fn rotation_mixture_matches_single_rotation_special_case() {
        let theta = 0.5_f64;
        let z = target_matrix(theta + 0.02)[0][0]; // a small angular error, unit modulus
        let psi = psi_from_z(z);
        let via_mixture = rotation_mixture_diamond_distance(&[(1.0, psi)], theta);
        let via_single = single_rotation_diamond_distance(z, theta);
        eprintln!("via_mixture={via_mixture}, via_single={via_single}");
        assert!((via_mixture - via_single).abs() < 1e-9);
    }

    /// A symmetric straddling pair (equal and opposite angular error `delta`, equal weight
    /// `p=0.5`) exactly cancels the FIRST-order (linear-in-delta) term -- the whole point of
    /// the mixing theorem -- leaving only the theorem's own predicted SECOND-order residual
    /// `2*(p*sin^2(delta_lo) + (1-p)*sin^2(delta_hi))` (with q_lo=q_hi=1, unit-modulus
    /// idealized rotations), NOT exactly zero: mixing reduces the error from linear to
    /// quadratic order in delta, it does not eliminate it. An earlier version of this test
    /// wrongly asserted exactly-zero and "failed" on a numerically-correct result.
    #[test]
    fn rotation_mixture_matches_quadratic_cancellation_residual() {
        let theta = 0.5_f64;
        let delta = 0.001_f64; // paper's delta_k = Arg(z_k) - theta/2
        let z_lo = target_matrix(theta - 2.0 * delta)[0][0]; // psi_lo = theta/2 - delta
        let z_hi = target_matrix(theta + 2.0 * delta)[0][0]; // psi_hi = theta/2 + delta
        let psi_lo = psi_from_z(z_lo);
        let psi_hi = psi_from_z(z_hi);
        let dd = rotation_mixture_diamond_distance(&[(0.5, psi_lo), (0.5, psi_hi)], theta);
        let expected = 2.0 * (0.5 * delta.sin().powi(2) + 0.5 * delta.sin().powi(2));
        eprintln!("symmetric-mixing dd={dd}, expected quadratic residual={expected}");
        assert!(
            (dd - expected).abs() < 1e-12,
            "got {dd}, expected {expected}"
        );
    }

    #[test]
    fn single_rotation_distance_matches_known_small_case() {
        // R_z(theta) vs R_z(theta + delta): known exact diamond distance 2*|sin(delta/2)|.
        let theta = 0.5_f64;
        let delta = 0.02_f64;
        let z = target_matrix(theta + delta)[0][0];
        let got = single_rotation_diamond_distance(z, theta);
        let expected = 2.0 * (delta / 2.0).sin().abs();
        eprintln!("got={got}, expected={expected}");
        assert!(
            (got - expected).abs() < 1e-9,
            "got {got}, expected {expected}"
        );
    }

    /// Calibration check: what does the EXISTING, unmodified plain-diagonal protocol's
    /// `epsilon` parameter actually correspond to in diamond-norm terms?
    #[test]
    fn plain_diagonal_epsilon_convention_calibration() {
        use rsgridsynth::config::config_from_theta_epsilon;
        use rsgridsynth::gridsynth::gridsynth_gates;

        for &theta in &[0.1_f64, 0.7, 1.3, 2.5] {
            for &eps in &[1e-3_f64, 1e-4, 1e-5, 1e-6] {
                clear_caches();
                let mut config = config_from_theta_epsilon(theta, eps, 42, false, false);
                let result = gridsynth_gates(&mut config);
                let m = matrix_from_gates(&result.gates);
                let (dd, _max_od) = pauli_diamond_distance_from_branches(&[(1.0, m)], theta);
                eprintln!(
                    "theta={theta}, epsilon_param={eps}, achieved_diamond_distance={dd}, \
                     ratio={:.4}",
                    dd / eps
                );
            }
        }
    }
}
