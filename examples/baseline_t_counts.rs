// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Captures a baseline of plain-diagonal (Ross-Selinger) T-counts across a
//! sweep of angles, at a fixed epsilon. This mirrors the exact angle sweep
//! and parameters used by `tests/integration_test.rs::test_correct_decomposition_exact`
//! so that future "mixed diagonal" / "fallback" / "mixed fallback" protocol
//! work (Stage 1+) can compare their T-counts against this plain-diagonal
//! baseline on the identical angle set without re-running anything.
//!
//! Run with:
//!   cargo run --example baseline_t_counts
//!
//! To (re)capture the committed fixture at `tests/fixtures/baseline_diagonal_t_counts.csv`:
//!   cargo run --example baseline_t_counts > tests/fixtures/baseline_diagonal_t_counts.csv

use rsgridsynth::clear_caches;
use rsgridsynth::config::config_from_theta_epsilon;
use rsgridsynth::gridsynth::gridsynth_gates;

fn main() {
    let epsilon = 1e-10;
    let verbose = false;
    let seed = 0u64;
    let up_to_phase = false;

    let thetas: Vec<f64> = (0..64)
        .map(|k| k as f64 * std::f64::consts::PI / 32.0)
        .collect();

    println!("theta,t_count");

    let mut total_t_count: u64 = 0;
    for &theta in &thetas {
        clear_caches();
        let mut gridsynth_config =
            config_from_theta_epsilon(theta, epsilon, seed, verbose, up_to_phase);

        let res = gridsynth_gates(&mut gridsynth_config);
        let t_count = res.gates.t_count();
        total_t_count += t_count as u64;

        println!("{theta},{t_count}");
    }

    let mean_t_count = total_t_count as f64 / thetas.len() as f64;
    println!("# mean_t_count,{mean_t_count}");
}
