# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`rsgridsynth` is a Rust reimplementation of [pygridsynth], decomposing single-qubit Z-axis rotations
into exact Clifford+T gate sequences using the number-theoretic / geometry-of-numbers algorithm from
Ross & Selinger, "Optimal ancilla-free Clifford+T approximation of z-rotations" (arXiv:1403.2975).
It ships as both a library (`rsgridsynth::*`) and an optional CLI binary.

[pygridsynth]: https://github.com/quantum-programming/pygridsynth

## Commands

```bash
# Build the library (no CLI, default feature set)
cargo build

# Build/run the CLI binary — requires the `cli` feature (clap is optional behind it)
cargo build --bin rsgridsynth -F cli --release
./target/release/rsgridsynth <theta> <epsilon> [-v] [-t] [-p] [--error]

# Run all tests (unit tests in src/, integration tests in tests/)
cargo test

# Run a single integration test
cargo test --test integration_test simple_test -- --exact

# Run a single unit test (module-scoped, e.g. in src/common.rs)
cargo test --lib common::tests::test_sin_fbig_random -- --exact

# Lint (both are enforced in CI — clippy warnings are errors)
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings

# Run the example binary showing library usage
cargo run --example interface
```

CI (`.github/workflows/main.yml`) runs the above build/fmt/clippy/test sequence on stable and the pinned
MSRV (1.87).

## Architecture

The crate implements one pipeline: angle+tolerance in → exact Clifford+T gate string out. Each stage
lives in its own module and is called, in order, from `gridsynth::gridsynth_gates`:

1. **`config.rs`** — `GridSynthConfig`/`GridSynthConfig::with_compute_error` hold theta, epsilon, RNG,
   and timeouts. `config_from_theta_epsilon` is the library entry point for embedding; it also resets
   the global precision (see below) so repeated calls in one process don't inherit stale precision.
2. **`gridsynth.rs`** — orchestrates the search. Builds an `EpsilonRegion` (the target-angle wedge) and
   a `UnitDisk`, calls `to_upright::to_upright_set_pair` once to get a `GridOp` transform, then loops
   over increasing denominator exponent `k`, calling `tdgp::solve_tdgp` for lattice-point candidates and
   `diophantine::diophantine_dyadic` to try to complete each candidate into a full solution.
3. **`to_upright.rs`** — implements the "upright set pair" reduction (Ross–Selinger §5–6): a state
   machine (`step_lemma`) that repeatedly applies named lattice operators (Z, X, S, R, K, A, B, σ —
   named to match the paper) to bring an ellipse pair into a numerically tractable "upright" shape.
   Don't rename these branches away from the paper's names; they're referenced by section/lemma in
   comments.
4. **`tdgp.rs`** (two-dimensional grid problem) — given the upright transform, searches for `DOmega`
   lattice points inside the (transformed) target regions at a given `k`, delegating each axis to the
   one-dimensional solver in `odgp.rs`.
5. **`odgp.rs`** (one-dimensional grid problem) — `solve_odgp`/`solve_scaled_odgp`/`*_with_parity`
   enumerate `ZRootTwo`/`DRootTwo` points inside real intervals, using the `LAMBDA` unit of `Z[√2]` to
   rescale recursively (this is the performance-critical inner loop).
6. **`diophantine.rs`** — for a candidate `z`, solves the norm equation for a companion `w` such that
   `(z, w)` forms a unitary (via factoring + primality testing, each with its own timeout from
   `GridSynthConfig`). Caches (`PRIMALITY_CACHE`, `SQRT_CACHE`) are process-global; `clear_caches()`
   (re-exported at the crate root) resets them and is called between test cases.
7. **`unitary.rs`** — `DOmegaUnitary { z, w, n }` represents the exact synthesized matrix
   `[[z, -w̄ωⁿ], [w, z̄ωⁿ]]` up to the denominator exponent shared by `z`/`w`. Gate application is done
   algebraically via `mul_by_{t,s,h,x,w}_from_left`, not by materializing matrices.
8. **`synthesis_of_clifford_t.rs`** — `decompose_domega_unitary` is the inverse of step 7: walks the
   denominator exponent down to 0, peeling off `H`/`T`/`S`/`W` gates one at a time based on residues mod
   2, then hands the final Clifford correction to `normal_form.rs`.
9. **`normal_form.rs`** — canonical (coset × syllable) normal form for the Clifford group mod global
   phase; used to simplify the trailing Clifford part of the synthesized gate string.

### Number rings (`src/ring/`)

- `ZOmega`: `Z[ω]`, `ω = e^{iπ/4}`, quartic ring — the exact lattice coordinates.
- `ZRootTwo`: `Z[√2]`, the real subring (used by `odgp.rs`'s 1-D search).
- `DOmega` / `DRootTwo`: the corresponding rings extended with `1/√2` ("dyadic"), i.e. a `ZOmega`/
  `ZRootTwo` value paired with a denominator exponent `k` (scale factor `(√2)^-k`).
- All four cache derived values (`conj`, `conj_sq2`, `norm`, `residue`, …) in `OnceCell` fields — these
  types are meant to be cloned cheaply and recomputation-free, not mutated in place.
- `conj_sq2` is the `√2 ↦ -√2` ring automorphism (distinct from complex conjugation `conj`); both show
  up constantly across regions/grid ops because the algorithm tracks a value and its √2-conjugate in
  parallel (an "ellipse pair").

### Precision (`common.rs`)

`PREC_BITS` is a single process-global `AtomicUsize` controlling the working precision of every
`dashu_float::FBig<HalfEven>` computation in the crate (via `ib_to_bf_prec`/`fb_with_prec`). It is not
thread-local. `main.rs` and `config_from_theta_epsilon` set/reset it per run; calling lower-level
functions directly without going through one of those entry points will silently reuse whatever
precision the previous call left behind. This is also why `tests/integration_test.rs` marks every test
`#[serial]` — parallel test threads would race on `PREC_BITS`.

### CLI vs library

The binary (`src/main.rs`) is gated behind the `cli` feature (`required-features = ["cli"]`), which
pulls in `clap`; plain `cargo build`/`cargo check` only builds the library. Library callers should go
through `config::config_from_theta_epsilon(theta, epsilon, seed, verbose, up_to_phase)` →
`gridsynth::gridsynth_gates(&mut config)` → `GridSynthResult { gates, global_phase }`, as in
`examples/interface.rs`. Accuracy is not cached eagerly on the result; call
`achieved_diamond_error(theta)` (the `accuracy::AchievedDiamondError` trait, implemented for
`GridSynthResult` and the `protocol::*` result types) to recompute it on demand from the gate
string. `theta`/`epsilon` on the CLI are parsed with a custom decimal+exponent parser
(`config::parse_decimal_with_exponent`) rather than through `f64`, since `f64` cannot represent the
arbitrary decimal precision (`--dps`) the algorithm can target; the library's `f64`-based
`config_from_theta_epsilon` entry point has been fuzz-tested for accuracy down to `epsilon = 1e-15`
(see `tests/accuracy_fuzz_test.rs`).

### Phase modes

`gridsynth.rs::PhaseMode` distinguishes exact synthesis (`Exact`) from synthesis up to the fixed phase
`e^{iπ/8}` (`Shifted`), which uses differently-scaled epsilon-region/unit-disk pairs. When
`up_to_phase` is set, `gridsynth_gates` runs both and keeps whichever produced fewer `T` gates;
`GridSynthResult::global_phase` records which branch won.
