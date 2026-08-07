// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Ad-hoc differential check: prints the same (protocol, theta, epsilon, t_count,
//! diamond_distance, q_or_p) CSV shape as `tools/differential_check/check.py`, for the same
//! inputs, so the two can be compared side by side. Exploratory verification tool, not part
//! of the committed test suite.

use rsgridsynth::clear_caches;
use rsgridsynth::protocol::fallback::exact_q;
use rsgridsynth::protocol::{
    synth_fallback, synth_mixed_diagonal, synth_mixed_fallback, FallbackResult, MixedFallbackResult,
};

fn fbig_to_f64(x: &dashu_float::FBig<dashu_float::round::mode::HalfEven>) -> f64 {
    match x.to_f64() {
        dashu_base::Approximation::Exact(v) => v,
        dashu_base::Approximation::Inexact(v, _) => v,
    }
}

fn t_count(gates: &str) -> f64 {
    gates.chars().filter(|&c| c == 'T').count() as f64
}

fn main() {
    let thetas = [
        0.1,
        3.0 * std::f64::consts::PI / 32.0,
        5.0 * std::f64::consts::PI / 32.0,
        7.0 * std::f64::consts::PI / 32.0,
        std::f64::consts::PI / 3.0,
        std::f64::consts::PI / 6.0,
    ];
    let epsilons = [1e-4, 1e-6, 1e-8];
    let q = exact_q(7);

    println!("protocol,theta,epsilon,t_count,diamond_distance,q_or_p");

    for &eps in &epsilons {
        for (i, &theta) in thetas.iter().enumerate() {
            let seed = 1000 + i as u64;

            clear_caches();
            let md = synth_mixed_diagonal(theta, eps, seed, false);
            let mut cost = 0.0;
            for b in &md.branches {
                cost += fbig_to_f64(&b.weight) * t_count(&b.gates);
            }
            println!(
                "mixed_diagonal,{theta},{eps},{cost},{},",
                fbig_to_f64(&md.projective_diamond_error)
            );

            clear_caches();
            let sin_alpha = eps / 4.0; // matches fallback.rs's own slope-fit test convention
            match synth_fallback(theta, eps, q.clone(), sin_alpha, seed, false) {
                Some(FallbackResult {
                    projective_gates,
                    correction_gates,
                    success_probability,
                    ..
                }) => {
                    let p = fbig_to_f64(&success_probability);
                    let cost = t_count(&projective_gates) + (1.0 - p) * t_count(&correction_gates);
                    println!("fallback,{theta},{eps},{cost},,{p}");
                }
                None => println!("fallback,{theta},{eps},ERROR,NotFound,"),
            }

            clear_caches();
            match synth_mixed_fallback(theta, eps, q.clone(), seed, false) {
                Some(MixedFallbackResult::Exact { gates }) => {
                    println!("mixed_fallback,{theta},{eps},{},0,1.0", t_count(&gates));
                }
                Some(MixedFallbackResult::Mixed { lo, hi, p, .. }) => {
                    let p_f64 = fbig_to_f64(&p);
                    let side_cost = |s: &rsgridsynth::protocol::MixedFallbackSide| {
                        let p_success = fbig_to_f64(&s.success_probability);
                        let mut corr_cost = 0.0;
                        for b in &s.correction.branches {
                            corr_cost += fbig_to_f64(&b.weight) * t_count(&b.gates);
                        }
                        t_count(&s.projective_gates) + (1.0 - p_success) * corr_cost
                    };
                    let cost = p_f64 * side_cost(&lo) + (1.0 - p_f64) * side_cost(&hi);
                    println!("mixed_fallback,{theta},{eps},{cost},,{p_f64}");
                }
                None => println!("mixed_fallback,{theta},{eps},ERROR,NotFound,"),
            }
        }
    }
}
