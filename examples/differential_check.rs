// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Ad-hoc differential check: prints the same (protocol, theta, epsilon, t_count,
//! diamond_distance, q_or_p) CSV shape as `tools/differential_check/check.py`, for the same
//! inputs, so the two can be compared side by side. Exploratory verification tool, not part
//! of the committed test suite.

use rsgridsynth::clear_caches;
use rsgridsynth::config::config_from_theta_epsilon;
use rsgridsynth::protocol::fallback::exact_q;
use rsgridsynth::protocol::{
    synth_fallback, synth_mixed_diagonal, synth_mixed_fallback, AchievedDiamondError,
    MixedFallbackResult,
};

fn fbig_to_f64(x: &dashu_float::FBig<dashu_float::round::mode::HalfEven>) -> f64 {
    match x.to_f64() {
        dashu_base::Approximation::Exact(v) => v,
        dashu_base::Approximation::Inexact(v, _) => v,
    }
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
            let theta_fbig = config_from_theta_epsilon(theta, eps, seed, false, false).theta;

            clear_caches();
            let md = synth_mixed_diagonal(theta, eps, seed, false);
            let mut cost = 0.0;
            for b in &md.branches {
                cost += fbig_to_f64(&b.weight) * b.gates.t_count() as f64;
            }
            println!(
                "mixed_diagonal,{theta},{eps},{cost},{},",
                fbig_to_f64(&md.achieved_diamond_error(&theta_fbig))
            );

            clear_caches();
            let sin_alpha = eps / 4.0; // matches fallback.rs's own slope-fit test convention
            match synth_fallback(theta, eps, q.clone(), sin_alpha, seed, false) {
                Some(result) => {
                    let p = fbig_to_f64(&result.achieved_success_probability());
                    let cost = result.projective_gates.t_count() as f64
                        + (1.0 - p) * result.correction_gates.t_count() as f64;
                    println!("fallback,{theta},{eps},{cost},,{p}");
                }
                None => println!("fallback,{theta},{eps},ERROR,NotFound,"),
            }

            clear_caches();
            match synth_mixed_fallback(theta, eps, q.clone(), seed, false) {
                Some(MixedFallbackResult::Exact { gates }) => {
                    println!("mixed_fallback,{theta},{eps},{},0,1.0", gates.t_count());
                }
                Some(MixedFallbackResult::Mixed { lo, hi, p, .. }) => {
                    let p_f64 = fbig_to_f64(&p);
                    let side_cost = |s: &rsgridsynth::protocol::MixedFallbackSide| {
                        let p_success = fbig_to_f64(&s.achieved_success_probability());
                        let mut corr_cost = 0.0;
                        for b in &s.correction.branches {
                            corr_cost += fbig_to_f64(&b.weight) * b.gates.t_count() as f64;
                        }
                        s.projective_gates.t_count() as f64 + (1.0 - p_success) * corr_cost
                    };
                    let cost = p_f64 * side_cost(&lo) + (1.0 - p_f64) * side_cost(&hi);
                    println!("mixed_fallback,{theta},{eps},{cost},,{p_f64}");
                }
                None => println!("mixed_fallback,{theta},{eps},ERROR,NotFound,"),
            }
        }
    }
}
