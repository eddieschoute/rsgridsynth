//! Regression tests proving `rsgridsynth` is safe to call concurrently from multiple threads.
//!
//! This crate used to keep its working precision in a single process-global `AtomicUsize`
//! (`PREC_BITS`, `src/common.rs`), mutated on every `config_from_theta_epsilon` call. Two
//! threads synthesizing at different epsilons could each overwrite it mid-search, so a
//! thread's region predicates and its `max_k` search bound (`4 * get_prec_bits()`) could
//! silently be evaluated at another thread's precision -- the confirmed cause of reported
//! 60-second-to-20-minute hangs under parallel test execution: a corrupted precision drove
//! the exponential-in-`k` candidate search past any reasonable bound instead of finding a
//! solution and terminating.
//!
//! Precision is now explicit everywhere (`common::Prec`, carried on `GridSynthConfig` and
//! every per-synthesis struct) -- there is no ambient or global precision left in the crate,
//! so there is no shared state left for concurrent callers to race on. None of the tests
//! below are `#[serial]`; that is itself part of what they are checking.
//!
//! This file is a separate integration-test binary so it never interacts with the
//! `#[serial]`-guarded tests in `integration_test.rs` (which are serialized for a different,
//! unrelated reason: they assert exact golden gate strings after `clear_caches()`, and the
//! diophantine caches -- genuinely safe to share, keyed on exact integers -- would otherwise
//! let a concurrently-running test's cache population change another test's RNG consumption
//! and hence its golden string).

use rsgridsynth::accuracy::AchievedDiamondError;
use rsgridsynth::config::config_from_theta_epsilon;
use rsgridsynth::gridsynth::gridsynth_gates;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;
use std::thread;

fn fbig_to_f64(x: &dashu_float::FBig<dashu_float::round::mode::HalfEven>) -> f64 {
    match x.to_f64() {
        dashu_base::Approximation::Exact(v) => v,
        dashu_base::Approximation::Inexact(v, _) => v,
    }
}

/// The decisive test: many threads synthesize concurrently at very different epsilons (so a
/// precision mix-up would be numerically obvious, not a rounding-noise-sized discrepancy),
/// and each must independently meet its own requested accuracy. Before this crate's precision
/// was made explicit, this reliably reproduced the hang/incorrect-result failure mode within
/// a handful of rounds -- confirmed by running this exact test against the pre-refactor
/// `PREC_BITS`-based code (stashed locally), where it either times out or fails the accuracy
/// assertion.
#[test]
fn concurrent_synthesis_meets_each_own_epsilon() {
    let n = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .clamp(4, 8);
    let epsilons = [1e-2, 1e-12];
    let barrier = std::sync::Arc::new(Barrier::new(n));

    let handles: Vec<_> = (0..n)
        .map(|i| {
            let barrier = std::sync::Arc::clone(&barrier);
            thread::spawn(move || {
                let epsilon = epsilons[i % epsilons.len()];
                barrier.wait(); // maximize the chance every thread is mid-search at once
                for round in 0..3 {
                    let theta = (0.37 + 0.91 * (i * 3 + round) as f64) % std::f64::consts::TAU;
                    let mut config =
                        config_from_theta_epsilon(theta, epsilon, 7 + i as u64, false, false);
                    let expected_bits = config.prec.bits();

                    let res = gridsynth_gates(&mut config);

                    assert_eq!(
                        config.prec.bits(),
                        expected_bits,
                        "thread {i} round {round}: config.prec changed across the call \
                         (nothing should mutate it -- it is a plain field, not ambient state)"
                    );

                    let err = fbig_to_f64(&res.achieved_diamond_error(&config.theta));
                    assert!(
                        err <= 2.0 * epsilon,
                        "thread {i} round {round}: theta={theta} epsilon={epsilon:e} \
                         bits={expected_bits} achieved diamond error {err:e} exceeds budget \
                         {:e}",
                        2.0 * epsilon
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

/// Same input, run once alone and once concurrently under heavy unrelated load -- the two
/// must produce byte-identical gate strings. This is a stronger, more direct statement of
/// "no shared state" than the accuracy check above: any cross-thread leakage (precision or
/// otherwise) that didn't happen to violate the accuracy budget would still show up here as a
/// diverging gate string.
///
/// Deliberately uses only one thread doing real synthesis while others hammer
/// `config_from_theta_epsilon` at a very different (loose) epsilon -- reproducing the exact
/// shape of the originally reported bug (one tight-tolerance search corrupted by concurrent
/// unrelated low-precision activity).
#[test]
fn synthesis_is_immune_to_concurrent_unrelated_activity() {
    let theta = std::f64::consts::PI / 8.0;
    let epsilon = 1e-10;
    let seed = 1234;

    let mut solo_config = config_from_theta_epsilon(theta, epsilon, seed, false, false);
    let solo_gates = gridsynth_gates(&mut solo_config).gates.to_string();

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let gate = std::sync::Arc::new(Barrier::new(2));

    let clobberer = {
        let stop = std::sync::Arc::clone(&stop);
        let gate = std::sync::Arc::clone(&gate);
        thread::spawn(move || {
            gate.wait();
            let mut i = 0u64;
            while !stop.load(Ordering::Relaxed) {
                // 24-bit precision (per `prec_bits_for_epsilon`) -- if this crate still had
                // ambient precision, this would very likely stomp on the concurrent tight
                // search below mid-computation.
                let _ = config_from_theta_epsilon(0.5, 1e-2, i, false, false);
                i += 1;
            }
        })
    };

    let mut concurrent_config = config_from_theta_epsilon(theta, epsilon, seed, false, false);
    gate.wait();
    let concurrent_gates = gridsynth_gates(&mut concurrent_config).gates.to_string();
    stop.store(true, Ordering::Relaxed);
    clobberer.join().expect("clobberer thread panicked");

    assert_eq!(
        solo_gates, concurrent_gates,
        "gate string synthesized while a concurrent thread hammered an unrelated, \
         loose-precision config differs from the solo run -- indicates shared mutable state"
    );
}
