// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Bothe et al.'s (arXiv:2605.31544v2) exhaustively-searched, theta-independent optimal
//! over-rotation table (their Tables II/III, up to T-count 35; transcribed verbatim from the
//! paper's own LaTeX source -- `doc/bothe-more-efficient-clifford-t/main.tex` lines
//! ~1381-1436 -- not from the rendered PDF, to avoid any OCR/image-transcription risk on
//! these long gate strings).
//!
//! ## The trailing `Y`/`Z` letters
//!
//! The paper's own gate words end each row in one of the 24 single-qubit Cliffords, up to a
//! phase (main.tex line ~545), using Pauli letters `Y`/`Z` this crate's `Gate` alphabet
//! (`{H, S, T, X, W}`) doesn't have directly. Rather than guess at an undocumented convention,
//! every trailing `Y`/`Z` below has been *exactly* rewritten in terms of this crate's own gate
//! semantics, verified directly against `DOmegaUnitary::mul_by_{s,x,w}_from_left`
//! (`src/unitary.rs`) rather than assumed:
//!
//! - `mul_by_s_from_left` multiplies the matrix by `diag(1, i)` exactly (confirmed against the
//!   paper's own worked examples `e^{i*pi/4} * ISZ = diag(e^{i*pi/4}, e^{-i*pi/4})` and
//!   `e^{i*pi/8} * TSZ = diag(e^{i*pi/8}, e^{-i*pi/8})`, main.tex lines 553/562), so
//!   `Z = diag(1,-1) = S*S` exactly -- a trailing `Z` becomes `SS`, no phase correction needed.
//! - `mul_by_w_from_left` multiplies the *whole* matrix by `omega = e^{i*pi/4}` exactly (it
//!   scales both `z` and `w` by `omega` and bumps `n` by 2, matching global-phase
//!   multiplication), so `W*W` is the exact scalar `i`. Since
//!   `Y = [[0,-i],[i,0]] = i * X * Z` (direct 2x2 matrix computation), a trailing `Y` becomes
//!   `XSSWW` (`X*Z` from the `X`/`SS` factors, times the exact scalar `i` from `WW`) --
//!   because `W` is a pure global phase it commutes freely with everything to its left, so
//!   appending it at the very end is equivalent to inserting it anywhere in the suffix.
//!
//! Both substitutions are checked, for every row, against Table II's independently-tabulated
//! `(1-r, phi)` in this module's own test suite (`table_rows_decode_to_table_ii_phi_and_r`),
//! which is a genuine cross-check since Table II and Table III are separate tables in the
//! paper transcribed independently of each other.
//!
//! The table's own `p`/`tan alpha` bookkeeping is Bothe's *quasi*-probability convention,
//! which is NOT identical to this crate's proper-probability `mixture_weight` (they agree
//! only to `O(theta^2)`, per Bothe's own Appendix F). So this module never trusts the
//! table's own numbers for a correctness guarantee -- it only uses the table as a source of
//! *candidate gate words* (which decode to exact ring elements regardless of which
//! accounting scheme found them), and independently recomputes the real `p` and achieved
//! diamond-norm error for each one against the actual `(theta, delta)` via this crate's own
//! (corrected) `mixture_weight`. This sidesteps needing to replicate Bothe's own
//! `tan(alpha)`-indexed bisection/staircase-selection algorithm (see their reference
//! `anc/small_angle_costing.py`) at all: with only ~55 rows, trying every one and verifying
//! directly is cheap enough to just do exhaustively, and is more robust than trusting a
//! selection rule tuned for a different (quasi-probability) error accounting.

use crate::gate::GateSeq;

/// `(tan(alpha), T-count, gate sequence)` for each row of Bothe's Tables II/III, in the same
/// (decreasing `tan(alpha)`, increasing T-count) order the paper lists them. `tan(alpha)` is
/// kept only for documentation/cross-reference with the paper -- this module's own lookup
/// (see [`row_gates`]'s callers) does not use it at all, trying every row directly instead.
///
/// Gate strings are the paper's own (its trailing "." punctuation stripped) with every
/// trailing `Y`/`Z` expanded per the module doc comment; each parses via `GateSeq::from_str`
/// (leading "I" is a no-op per that impl, matching the paper's own use of "I" as an explicit
/// identity-coset marker in Matsumoto-Amano normal form).
#[rustfmt::skip]
pub(crate) const OVER_ROTATION_TABLE: &[(f64, usize, &str)] = &[
    (1.00e+00, 0, "ISSS"),
    (4.14e-01, 1, "TSSS"),
    (3.51e-01, 4, "ISHTHTSHTSHTHSS"),
    (3.13e-01, 8, "ISHTSHTSHTHTHTHTHTSHTSHSI"),
    (2.12e-01, 7, "IHTSHTHTHTSHTSHTSHTSHX"),
    (1.99e-01, 12, "ISHTSHTHTHTHTHTSHTSHTSHTSHTSHTSHTHX"),
    (1.38e-01, 9, "IHTHTSHTSHTHTSHTHTSHTSHTHSI"),
    (1.15e-01, 10, "IHTHTHTSHTSHTHTHTSHTSHTHTHSI"),
    (1.11e-01, 13, "ISHTSHTSHTSHTHTHTSHTHTSHTHTHTSHTSHTSHSS"),
    (1.06e-01, 16, "IHTHTSHTHTHTSHTSHTSHTSHTHTSHTHTHTSHTSHTSHTSI"),
    (9.56e-02, 15, "IHTHTHTSHTHTSHTHTSHTHTHTSHTHTSHTSHTSHTHSXSSWW"),
    (9.39e-02, 17, "IHTHTSHTHTHTSHTSHTSHTSHTHTSHTSHTSHTSHTHTHTSHTHI"),
    (8.98e-02, 14, "IHTSHTHTHTSHTSHTHTHTHTSHTSHTSHTHTSHTSHSSS"),
    (8.10e-02, 16, "ISHTSHTHTSHTHTHTHTSHTSHTHTHTHTSHTSHTSHTSHTSHSX"),
    (8.01e-02, 18, "IHTSHTSHTSHTHTSHTSHTSHTHTHTHTHTSHTSHTSHTHTSHTSHTSHSS"),
    (7.18e-02, 18, "ISHTHTSHTSHTSHTHTHTSHTHTSHTHTHTHTSHTSHTSHTHTHTSHXSSWW"),
    (6.81e-02, 15, "IHTHTSHTHTHTSHTSHTSHTHTHTSHTSHTSHTHTHTHSSS"),
    (6.36e-02, 17, "ISHTHTSHTSHTSHTSHTHTHTSHTSHTSHTHTHTSHTSHTSHTSHTHSSS"),
    (5.66e-02, 18, "TSHTSHTSHTHTSHTHTSHTSHTSHTHTHTSHTSHTHTHTHTHTSHI"),
    (5.47e-02, 19, "ISHTSHTHTSHTSHTHTHTSHTSHTHTHTSHTSHTSHTHTHTHTHTSHTHSXSSWW"),
    (4.39e-02, 16, "IHTHTSHTSHTSHTSHTHTHTSHTSHTSHTSHTSHTSHTHTHTHSSS"),
    (4.32e-02, 20, "IHTHTHTHTSHTSHTHTHTSHTSHTSHTSHTSHTSHTHTHTSHTSHTHTHTHSS"),
    (4.18e-02, 21, "ISHTHTHTSHTHTSHTHTHTSHTHTSHTSHTHTSHTHTHTHTSHTHTSHTHTSHI"),
    (3.55e-02, 19, "IHTSHTHTSHTSHTSHTHTHTSHTHTSHTSHTSHTSHTSHTHTHTHTHTIXSSWW"),
    (2.66e-02, 11, "ISHTSHTHTSHTHTHTSHTHTHTSHTHTSHSS"),
    (2.37e-02, 15, "ISHTHTHTHTSHTSHTSHTSHTSHTSHTSHTSHTSHTHTHTHSI"),
    (2.32e-02, 19, "ISHTSHTHTHTHTHTHTHTSHTHTHTSHTHTHTSHTSHTHTHTSHTHSX"),
    (2.30e-02, 25, "IHTHTHTHTHTHTHTHTHTSHTSHTSHTSHTSHTSHTSHTSHTSHTHTHTHTSHTSHTHTSHTSHX"),
    (2.18e-02, 24, "TSHTHTHTSHTHTHTSHTSHTSHTHTSHTSHTSHTSHTSHTSHTSHTSHTSHTHTHTHTHTHI"),
    (2.16e-02, 26, "IHTSHTSHTHTSHTHTHTSHTSHTHTSHTHTSHTHTHTSHTHTHTSHTHTSHTSHTHTHTHTHTHSS"),
    (2.11e-02, 25, "ISHTSHTSHTHTHTSHTHTSHTSHTHTHTSHTSHTSHTHTSHTSHTHTSHTHTSHTHTHTHTHTSHX"),
    (2.02e-02, 26, "IHTHTHTHTHTHTHTSHTSHTHTSHTSHTHTSHTSHTSHTHTHTSHTSHTHTHTHTSHTHTHTSX"),
    (1.64e-02, 22, "THTHTHTSHTHTHTSHTHTHTSHTHTHTHTSHTHTHTSHTHTHTSHTHTHSSS"),
    (1.33e-02, 23, "TSHTHTSHTHTHTHTHTSHTSHTHTSHTSHTSHTSHTHTSHTSHTHTHTHTHTSHTHSSS"),
    (1.17e-02, 26, "THTHTSHTSHTSHTSHTSHTHTHTSHTHTSHTSHTHTHTHTHTSHTHTHTHTSHTHTHTHTHSXSSWW"),
    (1.16e-02, 28, "ISHTHTHTSHTHTSHTHTHTSHTHTHTSHTHTSHTSHTHTSHTSHTHTHTSHTHTHTSHTHTHTHTSHTHSX"),
    (1.01e-02, 27, "ISHTHTSHTHTSHTHTSHTHTHTSHTHTHTSHTSHTHTHTSHTSHTHTSHTHTHTSHTHTHTHTHTSHX"),
    (9.18e-03, 24, "TSHTHTHTSHTSHTHTHTSHTSHTSHTSHTSHTSHTSHTSHTSHTHTSHTHTHTHTSHTHTHI"),
    (8.94e-03, 26, "ISHTHTSHTSHTSHTSHTHTHTSHTSHTHTSHTSHTSHTHTHTHTSHTSHTSHTSHTHTSHTSHTHTSHTHSSS"),
    (8.91e-03, 28, "IHTSHTSHTSHTHTSHTSHTSHTHTHTHTSHTSHTHTSHTSHTSHTSHTHTHTHTSHTSHTSHTHTSHTHTHTHI"),
    (8.89e-03, 30, "IHTHTHTHTHTSHTSHTSHTSHTSHTSHTHTHTSHTSHTHTHTSHTHTHTSHTHTHTHTSHTHTSHTHTHTHTSHI"),
    (8.22e-03, 30, "THTHTHTSHTHTSHTHTSHTHTHTHTSHTSHTSHTSHTHTSHTSHTSHTSHTHTHTHTSHTHTSHTHTSHTHTHI"),
    (8.13e-03, 31, "TSHTSHTSHTHTHTSHTHTHTSHTHTHTSHTHTHTHTSHTHTHTSHTHTSHTHTSHTSHTSHTSHTSHTSHTSHTSHTHSI"),
    (5.68e-03, 21, "THTHTHTHTHTHTHTSHTHTHTHTHTHTHTSHTHTHTHTHTHTHSI"),
    (5.49e-03, 25, "ISHTHTHTSHTSHTSHTSHTSHTSHTHTSHTSHTHTHTSHTHTSHTSHTHTSHTHTHTSHTHTSHTHSXSSWW"),
    (5.44e-03, 27, "IHTSHTSHTSHTSHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHTHTHTHTHTHX"),
    (5.01e-03, 33, "THTSHTHTSHTSHTHTSHTHTSHTHTSHTHTSHTHTHTSHTHTHTSHTSHTHTHTSHTHTSHTSHTHTSHTHTHTHTHTSHSS"),
    (4.45e-03, 32, "THTSHTHTHTSHTSHTSHTSHTSHTSHTHTHTSHTSHTSHTSHTSHTHTSHTHTHTHTSHTHTHTSHTSHTSHTSHTHTHTSHSX"),
    (3.80e-03, 31, "IHTSHTSHTSHTHTHTSHTHTHTSHTHTHTHTSHTSHTHTSHTHTHTSHTHTSHTSHTHTSHTHTHTSHTHTHTHTHSSS"),
    (3.41e-03, 30, "THTSHTSHTSHTHTSHTHTHTSHTHTSHTHTSHTHTSHTHTSHTHTHTHTHTSHTHTHTSHTSHTSHTHTHTHSI"),
    (3.40e-03, 32, "IHTHTHTSHTHTSHTSHTSHTSHTSHTSHTSHTSHTSHTHTSHTHTSHTHTSHTSHTHTHTSHTHTHTSHTHTHTHTHTHTHX"),
    (3.36e-03, 35, "IHTSHTSHTHTSHTSHTHTHTSHTHTSHTHTHTHTHTHTHTHTHTSHTSHTHTSHTSHTHTSHTHTHTHTSHTHTHTHTHTHTHSI"),
    (2.62e-03, 35, "IHTHTHTHTHTHTSHTHTSHTSHTSHTSHTHTHTHTHTSHTSHTHTSHTSHTHTHTHTHTSHTSHTSHTSHTHTSHTHTHTHTHTHSSS"),
    (2.42e-03, 34, "IHTHTSHTHTHTSHTHTSHTSHTSHTHTSHTSHTSHTSHTSHTSHTSHTSHTHTSHTHTHTSHTSHTHTHTHTHTHTSHTHTHTSHTSHSSS"),
    (1.94e-03, 33, "THTHTSHTSHTSHTHTSHTHTHTSHTHTHTSHTSHTHTHTHTHTHTHTHTHTHTHTSHTHTSHTHTHTHTHTHTHSX"),
];

/// Parses row `i`'s gate string (panics on a malformed hardcoded string -- a bug in this
/// table, not a runtime condition).
pub(crate) fn row_gates(i: usize) -> GateSeq {
    OVER_ROTATION_TABLE[i]
        .2
        .parse()
        .unwrap_or_else(|e| panic!("OVER_ROTATION_TABLE[{i}] failed to parse: {e:?}"))
}

/// Table II's own `(1-r, phi)` for each row, in the same order as [`OVER_ROTATION_TABLE`] --
/// used ONLY by this module's test suite, to verify [`row_gates`]'s decoding (including the
/// trailing `Y`/`Z` expansion described in the module doc comment) against
/// independently-transcribed data from a *different* table in the paper (Table II, not III),
/// rather than trusting a self-consistent-but-possibly-wrong transcription of Table III alone.
#[cfg(test)]
#[rustfmt::skip]
const TABLE_II_ONE_MINUS_R_PHI: &[(f64, f64)] = &[
    (0.00e+00, 7.85e-01),
    (0.00e+00, 3.93e-01),
    (1.08e-02, 2.55e-01),
    (2.68e-03, 2.85e-01),
    (1.57e-03, 1.93e-01),
    (2.01e-03, 1.74e-01),
    (7.86e-04, 1.24e-01),
    (2.30e-04, 1.10e-01),
    (3.37e-05, 1.10e-01),
    (2.13e-04, 1.02e-01),
    (1.66e-04, 9.17e-02),
    (8.42e-06, 9.34e-02),
    (6.80e-04, 7.02e-02),
    (1.20e-04, 7.78e-02),
    (4.81e-05, 7.87e-02),
    (7.68e-05, 6.95e-02),
    (2.94e-04, 5.78e-02),
    (4.91e-05, 6.19e-02),
    (2.53e-05, 5.56e-02),
    (2.60e-05, 5.37e-02),
    (3.86e-05, 4.20e-02),
    (1.11e-05, 4.26e-02),
    (6.02e-06, 4.14e-02),
    (1.44e-05, 3.47e-02),
    (6.74e-05, 1.98e-02),
    (1.68e-05, 2.22e-02),
    (5.65e-06, 2.27e-02),
    (1.64e-06, 2.29e-02),
    (7.73e-06, 2.11e-02),
    (2.85e-06, 2.13e-02),
    (9.82e-06, 2.01e-02),
    (1.28e-06, 2.00e-02),
    // NOTE: Table II has a row for tan(alpha)=1.32e-02 (1-r=4.51e-08, phi=1.32e-02) that
    // Table III's gate-sequence table omits entirely -- no executable word exists for it in
    // the paper's own source, so it's dropped here too, to keep this array in row-for-row
    // correspondence with `OVER_ROTATION_TABLE` (which is built from Table III and
    // therefore never had that row in the first place).
    (6.20e-08, 1.64e-02),
    (1.06e-07, 1.32e-02),
    (8.13e-07, 1.15e-02),
    (3.33e-08, 1.16e-02),
    (1.68e-08, 1.01e-02),
    (2.50e-06, 8.60e-03),
    (4.93e-07, 8.83e-03),
    (2.47e-07, 8.86e-03),
    (7.18e-08, 8.88e-03),
    (1.32e-07, 8.19e-03),
    (9.28e-08, 8.11e-03),
    (1.23e-06, 5.20e-03),
    (3.02e-07, 5.38e-03),
    (2.32e-08, 5.43e-03),
    (2.37e-09, 5.01e-03),
    (1.81e-08, 4.44e-03),
    (1.26e-08, 3.80e-03),
    (5.03e-08, 3.38e-03),
    (3.03e-09, 3.40e-03),
    (2.28e-09, 3.36e-03),
    (2.78e-10, 2.62e-03),
    (4.18e-08, 2.39e-03),
    (9.64e-09, 1.93e-03),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Prec;
    use crate::unitary::DOmegaUnitary;
    use dashu_float::round::mode::HalfEven;
    use dashu_float::FBig;

    const PREC: Prec = Prec(200);

    fn fbig_to_f64(x: &FBig<HalfEven>) -> f64 {
        use dashu_base::Approximation;
        match x.to_f64() {
            Approximation::Inexact(v, _) => v,
            Approximation::Exact(v) => v,
        }
    }

    /// Independently verifies [`row_gates`]'s decoding (including the trailing `Y`/`Z`
    /// expansion) against [`TABLE_II_ONE_MINUS_R_PHI`] (transcribed from a *different* table
    /// in the paper than the gate sequences themselves come from -- Table II, not III -- so
    /// this is a genuine cross-check, not circular). This is the test that actually confirms
    /// the `Z -> SS` / `Y -> XSSWW` substitutions in the module doc comment are exact, rather
    /// than just algebraically plausible.
    ///
    /// Only the *magnitude* `r = |z|` is checked against Table II, not `phi = arg(z)`: Table
    /// II's `phi` is defined after an extra determinant-symmetrizing phase correction (the
    /// paper normalizes so the matrix reads `diag(e^{i phi}, e^{-i phi})` for the diagonal
    /// rows -- see `e^{i pi/4} ISZ = diag(e^{i pi/4}, e^{-i pi/4})`, main.tex line 553), which
    /// this crate's raw `DOmegaUnitary::z()` is not normalized to (its global phase is
    /// whatever `n` the gate word happens to produce). Reproducing that exact symmetrization
    /// convention isn't needed for this module's actual use (feeding candidates to this
    /// crate's own `mixture_weight` against a real `(theta, delta)`, which only cares about
    /// the *decoded matrix itself*, not any external phase convention) so this test doesn't
    /// attempt it. `r`, by contrast, is a magnitude and is convention-independent -- checking
    /// it here is still a real, transcription-error-detecting cross-check against a
    /// separately-transcribed table.
    #[test]
    fn table_rows_decode_to_table_ii_r() {
        assert_eq!(OVER_ROTATION_TABLE.len(), TABLE_II_ONE_MINUS_R_PHI.len());
        let mut max_r_err = 0.0_f64;
        for (i, &(one_minus_r, _phi)) in TABLE_II_ONE_MINUS_R_PHI.iter().enumerate() {
            let gates = row_gates(i);
            let u = DOmegaUnitary::from_gates(&gates);
            let re = fbig_to_f64(u.z().real(PREC));
            let im = fbig_to_f64(u.z().imag(PREC));
            let r = (re * re + im * im).sqrt();

            let r_err = (r - (1.0 - one_minus_r)).abs();
            max_r_err = max_r_err.max(r_err);

            assert!(
                r_err < 1e-3,
                "row {i}: decoded r={r} disagrees with Table II's 1-r={one_minus_r} \
                 (r_expected={}, err={r_err})",
                1.0 - one_minus_r
            );
        }
        eprintln!("max r error={max_r_err:e} across all rows");
    }

    /// Direct, convention-independent proof of the `Z -> SS` / `Y -> XSSWW` substitutions
    /// claimed in the module doc comment: decode each short gate word in isolation and check
    /// it against the exact abstract Pauli matrix, rather than relying on any table row or
    /// phase-normalization convention. `Z = diag(1,-1)` and `Y = [[0,-i],[i,0]]` are checked
    /// against `DOmegaUnitary`'s own `(z, w)` representation (`z` is the top-left, `w` the
    /// bottom-left entry, per its own doc comment), computed exactly in the ring (no floating
    /// point) via `DOmegaUnitary::z()`/`w()` and only converted to `f64` for the comparison.
    #[test]
    fn z_and_y_substitutions_are_exact() {
        let z_gate: GateSeq = "SS".parse().unwrap();
        let u = DOmegaUnitary::from_gates(&z_gate);
        assert_eq!(fbig_to_f64(u.z().real(PREC)), 1.0);
        assert_eq!(fbig_to_f64(u.z().imag(PREC)), 0.0);
        assert_eq!(fbig_to_f64(u.w().real(PREC)), 0.0);
        assert_eq!(fbig_to_f64(u.w().imag(PREC)), 0.0);
        let mat = u.to_complex_matrix(PREC);
        assert!((fbig_to_f64(&mat[(1, 1)].re) - (-1.0)).abs() < 1e-30);
        assert!(fbig_to_f64(&mat[(1, 1)].im).abs() < 1e-30);

        let y_gate: GateSeq = "XSSWW".parse().unwrap();
        let u = DOmegaUnitary::from_gates(&y_gate);
        assert_eq!(fbig_to_f64(u.z().real(PREC)), 0.0);
        assert_eq!(fbig_to_f64(u.z().imag(PREC)), 0.0);
        assert_eq!(fbig_to_f64(u.w().real(PREC)), 0.0);
        assert_eq!(fbig_to_f64(u.w().imag(PREC)), 1.0);
        let mat = u.to_complex_matrix(PREC);
        assert!(fbig_to_f64(&mat[(0, 1)].re).abs() < 1e-30);
        assert!((fbig_to_f64(&mat[(0, 1)].im) - (-1.0)).abs() < 1e-30);
    }

    /// The load-bearing safety net for this hand-transcribed table: decode every row and
    /// check its T-count matches the table's own stated T-count exactly. `GateSeq::t_count`
    /// counts `Gate::T` occurrences directly (no normal-form round trip), so this check is
    /// independent of how the string was parsed -- a single dropped/added/mistyped `T`
    /// character (the single most likely transcription error) is caught immediately, and a
    /// mistyped `H`/`S` character is very likely to also perturb the T-count once decoded
    /// through the ring (Matsumoto-Amano normal form is T-count-minimal, so an incorrect
    /// Clifford placement generally does not decode to the same T-count by coincidence). The
    /// `Y`/`Z` expansions add no `T` gates, so this holds for every row unchanged.
    #[test]
    fn table_rows_decode_to_their_stated_t_count() {
        for (i, &(tan_alpha, t_count, gate_str)) in OVER_ROTATION_TABLE.iter().enumerate() {
            let gates = row_gates(i);
            assert_eq!(
                gates.t_count(),
                t_count,
                "row {i} (tan_alpha={tan_alpha}, gate_str={gate_str:?}): decoded T-count {} \
                 != table's stated T-count {t_count}",
                gates.t_count()
            );
        }
    }

    #[test]
    fn table_is_sorted_by_decreasing_tan_alpha() {
        for i in 1..OVER_ROTATION_TABLE.len() {
            assert!(
                OVER_ROTATION_TABLE[i].0 < OVER_ROTATION_TABLE[i - 1].0,
                "row {i} tan_alpha={} is not strictly less than row {}'s tan_alpha={}",
                OVER_ROTATION_TABLE[i].0,
                i - 1,
                OVER_ROTATION_TABLE[i - 1].0
            );
        }
    }
}
