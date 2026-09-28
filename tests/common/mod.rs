// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Shared helpers for this crate's integration tests. Living under `tests/common/` (rather
//! than a bare `tests/common.rs`) keeps Cargo from compiling this as its own test binary --
//! only files directly in `tests/` become separate test targets, so each test file instead
//! declares `mod common;` and pulls in what it needs from here.
//!
//! `examples/pauli_transfer_verification.rs` is a separate compilation unit again (an example,
//! not a test binary), but includes this same source file via `#[path = "../tests/common/mod.rs"]
//! mod common;` rather than keeping a second hand-duplicated copy.

/// Relative slack for an `achieved <= epsilon`-style diamond-norm assertion. The *bound* is
/// exact -- `EpsilonRegion`'s cap `Re(w) >= sqrt(1 - eps^2/4)` gives
/// `diagonal_diamond_distance = 2*sqrt(1 - Re(w)^2) <= eps`, with equality on the boundary.
/// The *evaluation* is not: working precision is only `12 * log10(1/epsilon)` bits
/// (`config::prec_bits_for_epsilon`), and since `1 - Re(w)^2 ~= eps^2/4`, an absolute error
/// `2^-prec` in `Re(w)` becomes a *relative* error `~4 * 2^-prec / eps^2` in the reported
/// distance: ~2.4e-3 at eps=1e-2 (only 24 working bits), ~6e-5 at 1e-3, ~1.4e-6 at 1e-4, below
/// 1e-9 from 1e-6 down. A flat 1e-9 margin is therefore NOT safe at coarse epsilon.
///
/// The `16.0` factor below (not the `4.0` the paragraph above derives) is a deliberate 4x
/// safety margin on top of that linearized estimate, to absorb the imprecision of the
/// estimate itself (it is not a tight error bound, just an order-of-magnitude one) -- it is
/// still at least 2 orders of magnitude tighter than the `2.0 *` slack this replaced at the
/// worst-case epsilon, and at least 8 orders tighter from 1e-6 down.
pub fn measurement_slack(epsilon: f64) -> f64 {
    let prec_bits = (12.0 * (1.0 / epsilon).log10()).max(16.0);
    (16.0 * 2f64.powf(-prec_bits) / (epsilon * epsilon)).max(1e-9)
}
