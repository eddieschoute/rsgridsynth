# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`rsgridsynth` is a Rust reimplementation of [pygridsynth], decomposing single-qubit Z-axis rotations
into exact Clifford+T gate sequences using the number-theoretic / geometry-of-numbers algorithm from
Ross & Selinger, "Optimal ancilla-free Clifford+T approximation of z-rotations" (arXiv:1403.2975).
It ships as both a library (`rsgridsynth::*`) and an optional CLI binary.

[pygridsynth]: https://github.com/quantum-programming/pygridsynth

## Commands

A `Makefile` at the repo root wraps the commands below (`make help` lists all targets). It has no
dependencies beyond `cargo`/`rustup` — this is a plain GNU Makefile, not `cargo-make`.

```bash
# Show all available targets
make help

# Build the library (no CLI, default feature set)
make build           # == cargo build

# Build the release CLI binary — requires the `cli` feature (clap is optional behind it)
make build-cli        # == cargo build --bin rsgridsynth -F cli --release
./target/release/rsgridsynth <theta> <epsilon> [-v] [-t] [-p] [--error]

# Lint and test — same flags CI runs
make fmt-check         # == cargo fmt --all -- --check
make clippy            # == cargo clippy --all-features --all-targets -- -D warnings
make test              # == cargo test --all-features --all-targets && cargo test --all-features --doc

# The full CI gate, in one command
make ci                # == fmt-check + clippy + test

# Reformat in place (writes files; not run in CI)
make fmt                # == cargo fmt --all

# Run a single integration test
cargo test --test integration_test simple_test -- --exact

# Run a single unit test (module-scoped, e.g. in src/common.rs)
cargo test --lib common::tests::test_sin_fbig_random -- --exact

# Run the example binary showing library usage
cargo run --example interface
```

CI (`.github/workflows/main.yml`) runs `make ci` on stable and the pinned MSRV (1.87).
`make clippy`/`make test` use `--all-features --all-targets`, which is broader than a bare
`cargo clippy`/`cargo test`: it also lints `src/main.rs` (only reachable via the `cli` feature)
and runs the `#[test]` cases embedded in `examples/pauli_transfer_verification.rs`.

## Tooling

Prefer LSP (rust-analyzer) over grep for navigating this crate: `documentSymbol` gives instant,
signature-complete outlines of a file (works immediately, no indexing wait) and `goToDefinition`/
`findReferences` jump across the ring/region/gate types accurately. Note `findReferences` and
`workspaceSymbol` need rust-analyzer to finish indexing the workspace first — they return "no
results" rather than erroring while that's in progress (noticeable right after opening a fresh
worktree), so fall back to `grep` if they come back empty and you're not sure indexing is done.

## Architecture

The crate implements one pipeline: angle+tolerance in → exact Clifford+T gate sequence
(`gate::GateSeq`) out. Each stage lives in its own module and is called, in order, from
`gridsynth::gridsynth_gates`:

1. **`config.rs`** — `GridSynthConfig` holds theta, epsilon, RNG, working precision (`prec: Prec`,
   see below), timeouts, and the per-call `diophantine::Caches`. `config_from_theta_epsilon` is the
   library entry point for embedding; it builds a fresh `Prec` and `Caches` for that call, so
   concurrent calls never share state and output depends only on `(theta, epsilon, seed)`.
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
   `GridSynthConfig`). The primality/sqrt/factor/diophantine memo caches (`diophantine::Caches`) are
   caller-owned, living on `GridSynthConfig::diophantine_data`, not process-global — each call to
   `config_from_theta_epsilon` gets its own, so concurrent syntheses never share them.
7. **`unitary.rs`** — `DOmegaUnitary { z, w, n }` represents the exact synthesized matrix
   `[[z, -w̄ωⁿ], [w, z̄ωⁿ]]` up to the denominator exponent shared by `z`/`w`. Gate application is done
   algebraically via `mul_by_{t,s,h,x,w}_from_left`, not by materializing matrices.
8. **`synthesis_of_clifford_t.rs`** — `decompose_domega_unitary` is the inverse of step 7: walks the
   denominator exponent down to 0, peeling off `H`/`T`/`S`/`W` gates one at a time based on residues mod
   2, then hands the final Clifford correction to `normal_form.rs`.
9. **`normal_form.rs`** — canonical (coset × syllable) normal form for the Clifford group mod global
   phase; used to simplify the trailing Clifford part of the synthesized gate sequence.

### Gate sequences (`gate.rs`)

`Gate` (`H`/`S`/`T`/`X`/`W`) and `GateSeq` (a `Vec<Gate>` newtype) are the crate's one gate
representation; every public result type (`GridSynthResult` and the `protocol::*` results) carries a
`GateSeq`, not a `String`. There is no `Gate::I` variant — identity is the empty sequence, and
`GateSeq`'s `Display` renders `"I"` only for that empty case, never mid-sequence (fixing a
historical bug where `Clifford::to_gates`'s old empty-string "I" sentinel could leak into the middle
of a longer word, e.g. `"HTSHTI"`, that `NormalForm::from_gates` couldn't parse back). `GateSeq`
derefs to `&[Gate]`, so most call sites that used to hold a `String` need no change beyond the field
type; `Display`/`FromStr` are the serialization pair (`FromStr` treats `'I'` as a no-op anywhere in
the string, for backward compatibility with pre-`GateSeq` output).

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

Working precision is **explicit, not ambient** — there is no global or thread-local precision
anywhere in the crate. `Prec(pub usize)` is a `Copy` newtype threaded through every computation:
either as a field on the per-synthesis structs that already exist (`GridSynthConfig`, `Ellipse`,
`Interval`, `EpsilonRegion`/`UnitDisk`, the `protocol::*` regions/results, `WFrame`), or as a
parameter on the leaf helpers (`math.rs`, `ring::*` float projections, `odgp.rs`/`tdgp.rs`).
`Prec::ib` (`IBig` → `FBig`) is the load-bearing coercion — `FBig::from(IBig)` is precision-0,
which `dashu_float` defines as unlimited — while `Prec::fb` (re-pinning an existing `FBig`) is
needed only at genuine precision boundaries, since dashu propagates precision as `max(lhs, rhs)`
through arithmetic. Because nothing is shared across calls or threads, concurrent syntheses are
independent by construction and output is a deterministic function of `(theta, epsilon, seed)`;
`tests/concurrency_test.rs` asserts this directly, and no test in the crate needs `#[serial]` for
precision reasons anymore.

### CLI vs library

The binary (`src/main.rs`) is gated behind the `cli` feature (`required-features = ["cli"]`), which
pulls in `clap`; plain `cargo build`/`cargo check` only builds the library. Library callers should go
through `config::config_from_theta_epsilon(theta, epsilon, seed, verbose, up_to_phase)` →
`gridsynth::gridsynth_gates(&mut config)` → `GridSynthResult { gates: GateSeq, global_phase }`, as in
`examples/interface.rs`. Accuracy is not cached eagerly on the result; call
`achieved_diamond_error(theta)` (the `accuracy::AchievedDiamondError` trait, implemented for
`GridSynthResult` and the `protocol::*` result types) to recompute it on demand from the gate
sequence. `theta`/`epsilon` on the CLI are parsed with a custom decimal+exponent parser
(`config::parse_decimal_with_exponent`) rather than through `f64`, since `f64` cannot represent the
arbitrary decimal precision (`--dps`) the algorithm can target; the library's `f64`-based
`config_from_theta_epsilon` entry point has been fuzz-tested for accuracy down to `epsilon = 1e-15`
(see `tests/accuracy_fuzz_test.rs`).

### Phase modes

`gridsynth.rs::PhaseMode` distinguishes exact synthesis (`Exact`) from synthesis up to the fixed phase
`e^{iπ/8}` (`Shifted`), which uses differently-scaled epsilon-region/unit-disk pairs. When
`up_to_phase` is set, `gridsynth_gates` runs both and keeps whichever produced fewer `T` gates;
`GridSynthResult::global_phase` records which branch won.
