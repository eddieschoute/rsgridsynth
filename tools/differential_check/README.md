# Differential check: rsgridsynth vs. Qualtran

Ad-hoc, exploratory verification tool used while implementing the mixed-diagonal/fallback/
mixed-fallback protocols (Stages 1-3). Not part of the committed CI fixture pipeline -- see
`../generate_qualtran_fixtures` for that (which produces the version-controlled
`tests/fixtures/qualtran_reference.json` that Rust tests can read without needing Python).

This directory instead pairs `check.py` (Qualtran side) with `../../examples/differential_check.rs`
(rsgridsynth side): both print the same `protocol,theta,epsilon,t_count,diamond_distance,q_or_p`
CSV shape for the same `(theta, epsilon, q)` inputs, so the two can be diffed by hand.

## Usage

```bash
# Qualtran side (needs network the first time, to resolve/install qualtran; cached afterward)
uv run tools/differential_check/check.py

# rsgridsynth side
cargo run --example differential_check --quiet
```

## What this found

Comparing the two side by side across 6 angles x {1e-4, 1e-6, 1e-8} (a small, noisy sample --
not a rigorous statistical claim), rsgridsynth's measured T-count slopes (mixed diagonal ~1.59,
fallback ~1.11, mixed fallback ~0.50) track Qualtran's own observed slopes on the same inputs
(~1.53, ~1.02, ~0.53) reasonably closely, with a modest, roughly *constant* (not growing with
precision) excess T-count -- consistent with known, accepted implementation-detail differences
(rsgridsynth's mixed-diagonal/fallback regions cover only one of the two antipodal lobes the
paper's own area formulas count, see `mixed_diagonal.rs`'s module docs; the specific straddling
candidate pair chosen when several valid ones exist can also differ) rather than a fundamental
algorithmic error. Re-run this after any change to the region geometry or search logic in
`src/protocol/` to re-confirm.
