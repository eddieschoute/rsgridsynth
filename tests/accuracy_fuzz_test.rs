//! Fuzz-style accuracy tests.
//!
//! These generate many random target angles across a range of epsilons -- down to 1e-15 -- and
//! check that the synthesized gate string is actually within the requested tolerance of the
//! ideal rotation. `GridSynthConfig::epsilon` is a diamond-norm budget (see `CLAUDE.md`'s
//! "Accuracy convention" section) -- `EpsilonRegion`'s cap makes `achieved_diamond_error <=
//! epsilon` exact, with equality on the boundary, so there is no factor-of-2 slack to spend.
//! Accuracy is computed on demand via `GridSynthResult::achieved_diamond_error`
//! (`AchievedDiamondError`), not cached eagerly during synthesis, and cross-checked against a
//! genuinely different derivation (`independent_operator_error` below): it rebuilds the exact
//! unitary represented by the returned gate string (via `DOmegaUnitary::from_gates`) and the
//! ideal target rotation (via `Prec::cos`/`Prec::sin` at the same working precision) and computes
//! the *operator*-norm distance between them from the full matrix eigenvalue formula -- a
//! different code path from `achieved_diamond_error`'s `WFrame`-based shortcut, related by the
//! well-known `diamond = 2 * operator_norm` identity for this special (SU(2)-with-phase) matrix
//! form. That way a bug in either derivation would show up as a disagreement, not just as both
//! being wrong in the same way.

use dashu_base::Approximation;
use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;
use num::Complex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rsgridsynth::accuracy::AchievedDiamondError;
use rsgridsynth::common::Prec;
use rsgridsynth::config::config_from_theta_epsilon;
use rsgridsynth::gate::Gate;
use rsgridsynth::gridsynth::gridsynth_gates;
use rsgridsynth::unitary::DOmegaUnitary;
use serial_test::serial;

fn to_fbig(prec: Prec, x: f64) -> FBig<HalfEven> {
    FBig::<HalfEven>::try_from(x)
        .unwrap()
        .with_precision(prec.bits())
        .value()
}

fn fbig_to_f64(x: &FBig<HalfEven>) -> f64 {
    match x.to_f64() {
        Approximation::Exact(v) => v,
        Approximation::Inexact(v, _) => v,
    }
}

/// Relative slack for an `achieved <= epsilon`-style diamond-norm assertion. The *bound* is
/// exact -- `EpsilonRegion`'s cap `Re(w) >= sqrt(1 - eps^2/4)` gives
/// `diagonal_diamond_distance = 2*sqrt(1 - Re(w)^2) <= eps`, with equality on the boundary.
/// The *evaluation* is not: working precision is only `12 * log10(1/epsilon)` bits
/// (`config::prec_bits_for_epsilon`), and since `1 - Re(w)^2 ~= eps^2/4`, an absolute error
/// `2^-prec` in `Re(w)` becomes a *relative* error `~4 * 2^-prec / eps^2` in the reported
/// distance: ~2.4e-3 at eps=1e-2 (only 24 working bits), ~6e-5 at 1e-3, ~1.4e-6 at 1e-4, below
/// 1e-9 from 1e-6 down. A flat 1e-9 margin is therefore NOT safe at coarse epsilon.
fn measurement_slack(epsilon: f64) -> f64 {
    let prec_bits = (12.0 * (1.0 / epsilon).log10()).max(16.0);
    (16.0 * 2f64.powf(-prec_bits) / (epsilon * epsilon)).max(1e-9)
}

/// The exact operator-norm counterpart of `EpsilonRegion`'s diamond-norm cap: a candidate at
/// the boundary has `Re(w) = sqrt(1 - eps^2/4)`, so its operator-norm distance to the target is
/// `sqrt(2 - 2*Re(w))`. NOT `epsilon / 2` -- that is only the leading order; the exact value
/// exceeds it by a relative `eps^2/32` (~3e-6 at eps=1e-2), which a naive `epsilon / 2.0` bound
/// would fail to cover.
///
/// Computed as `sqrt(2*x / (1 + sqrt(1-x)))` for `x = eps^2/4`, the algebraic rationalization
/// of `sqrt(2 - 2*sqrt(1-x))` (multiply/divide by `1 + sqrt(1-x)`), rather than the textbook
/// form directly: at small epsilon, `x` is tiny and `1.0 - x` rounds to exactly `1.0` in `f64`,
/// so `2.0 - 2.0*(1.0-x).sqrt()` catastrophically cancels to exactly `0.0` (observed at
/// epsilon=1e-8 during development). The rationalized form never subtracts two nearly-equal
/// floats, so it stays accurate down to the smallest epsilon this crate supports.
fn operator_budget(epsilon: f64) -> f64 {
    let x = epsilon * epsilon / 4.0;
    (2.0 * x / (1.0 + (1.0 - x).sqrt())).sqrt()
}

/// Recomputes the *operator*-norm distance between the ideal z-rotation by `theta` and the
/// unitary represented by `gates`, entirely from public API, via the full matrix eigenvalue
/// formula -- a different derivation from `achieved_diamond_error`'s `WFrame`-based shortcut,
/// not a copy of it. `shifted` selects whether the synthesized unitary should be compared up to
/// the extra global phase `e^{i pi/8}` (this crate's `PhaseMode`), matching
/// `GridSynthResult::global_phase`.
fn independent_operator_error(
    prec: Prec,
    gates: &[Gate],
    theta: &FBig<HalfEven>,
    shifted: bool,
) -> f64 {
    let two = prec.fb(FBig::try_from(2.0).unwrap());
    let neg_theta_half = -prec.fb(theta / &two);
    let z_x = prec.fb(neg_theta_half.cos());
    let z_y = prec.fb(neg_theta_half.sin());

    let synthesized = DOmegaUnitary::from_gates(gates).to_complex_matrix(prec);
    let mut u = synthesized[(0, 0)].clone();
    if shifted {
        let p = to_fbig(prec, std::f64::consts::PI / 8.);
        let phase = Complex::new(p.cos(), p.sin());
        u = &u * &phase;
    }

    // Squared operator norm of (expected - synthesized), via the shared eigenvalue formula:
    // ||A^* A|| for A = expected - synthesized, both being 2x2 unitaries with the same
    // (0,0)-entry phase relationship.
    let eig: FBig<HalfEven> = 2 - 2 * (&z_x * &u.re + &z_y * &u.im);
    let eig = eig.max(FBig::from(0));
    let norm = eig.sqrt();
    match norm.to_f64() {
        Approximation::Inexact(v, _) => v,
        Approximation::Exact(v) => v,
    }
}

/// Runs the fuzzer for a given `up_to_phase` setting across a spread of epsilons -- from coarse
/// (1e-2) down to 1e-15 -- and many random target angles per epsilon, checking that:
///  - the on-demand `achieved_diamond_error` is within the requested diamond-norm budget
///    (`epsilon`, exactly -- see the module doc),
///  - an independently derived operator-norm error is *also* within the exact operator-norm
///    counterpart of that same budget (`operator_budget(epsilon)`, ~`epsilon/2`),
///  - the two error computations -- different derivations, related by `diamond = 2*operator` --
///    agree with each other.
fn run_accuracy_fuzz(up_to_phase: bool, thetas_per_epsilon: usize, seeds: &[u64]) {
    // From coarse tolerances down to 1e-15, spanning the precision regimes the algorithm has to
    // handle differently (see `config_from_theta_epsilon`'s `calculated_prec_bits`).
    let epsilons = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-15];

    let mut rng = StdRng::seed_from_u64(0xACC0_FA22);

    for &epsilon in &epsilons {
        for _ in 0..thetas_per_epsilon {
            let theta = rng.random_range(0.0..std::f64::consts::TAU);
            for &seed in seeds {
                let mut config =
                    config_from_theta_epsilon(theta, epsilon, seed, false, up_to_phase);
                let res = gridsynth_gates(&mut config);

                let slack = measurement_slack(epsilon);
                let diamond_error = fbig_to_f64(&res.achieved_diamond_error(&config.theta));
                assert!(
                    diamond_error <= epsilon * (1.0 + slack),
                    "achieved diamond error {diamond_error:e} exceeds requested budget \
                     epsilon={epsilon:e} for theta={theta}, seed={seed}, \
                     up_to_phase={up_to_phase}, gates={}",
                    res.gates
                );

                let independent_error = independent_operator_error(
                    config.prec,
                    &res.gates,
                    &config.theta,
                    res.global_phase,
                );
                let op_budget = operator_budget(epsilon);
                assert!(
                    independent_error <= op_budget * (1.0 + slack),
                    "independently computed operator error {independent_error:e} exceeds the \
                     exact operator-norm budget {op_budget:e} for theta={theta}, \
                     epsilon={epsilon:e}, seed={seed}, up_to_phase={up_to_phase}, gates={}",
                    res.gates
                );

                // Two different derivations (WFrame-shortcut diamond error vs. full-matrix
                // eigenvalue operator norm), related by `diamond = 2*operator_norm` for this
                // special matrix form -- should agree up to the last couple of bits of f64
                // rounding.
                let independent_diamond_error = 2.0 * independent_error;
                let diff = (diamond_error - independent_diamond_error).abs();
                let scale = diamond_error.max(independent_diamond_error).max(1e-300);
                assert!(
                    diff <= scale * 1e-6,
                    "achieved diamond error {diamond_error:e} and 2x independently computed \
                     operator error {independent_diamond_error:e} disagree for theta={theta}, \
                     epsilon={epsilon:e}, seed={seed}, up_to_phase={up_to_phase}"
                );
            }
        }
    }
}

#[test]
#[serial]
fn fuzz_accuracy_exact_phase() {
    run_accuracy_fuzz(false, 6, &[0, 1234, 987654321]);
}

#[test]
#[serial]
fn fuzz_accuracy_up_to_phase() {
    run_accuracy_fuzz(true, 6, &[0, 1234, 987654321]);
}

/// Dedicated, larger sweep specifically at the 1e-15 tolerance boundary this crate's README
/// flags as not fully supported for the CLI's f64-based epsilon parsing -- exercising many more
/// random angles at exactly that precision to build confidence the library entry point
/// (`config_from_theta_epsilon`/`gridsynth_gates`) still produces accurate results there.
#[test]
#[serial]
fn fuzz_accuracy_at_1e_minus_15() {
    let epsilon = 1e-15;
    let mut rng = StdRng::seed_from_u64(0x1E_15FA22);

    for _ in 0..40 {
        let theta = rng.random_range(0.0..std::f64::consts::TAU);
        let mut config = config_from_theta_epsilon(theta, epsilon, 42, false, false);
        let res = gridsynth_gates(&mut config);

        let slack = measurement_slack(epsilon);
        let diamond_error = fbig_to_f64(&res.achieved_diamond_error(&config.theta));
        assert!(
            diamond_error <= epsilon * (1.0 + slack),
            "achieved diamond error {diamond_error:e} exceeds requested budget epsilon={epsilon:e} \
             for theta={theta}"
        );

        let independent_error =
            independent_operator_error(config.prec, &res.gates, &config.theta, res.global_phase);
        let op_budget = operator_budget(epsilon);
        assert!(
            independent_error <= op_budget * (1.0 + slack),
            "independently computed operator error {independent_error:e} exceeds the exact \
             operator-norm budget {op_budget:e} for theta={theta}, gates={}",
            res.gates
        );
    }
}
