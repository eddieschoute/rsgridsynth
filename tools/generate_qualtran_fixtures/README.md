# generate_qualtran_fixtures

A `uv`-managed Python project that generates a differential-testing fixture
against [Qualtran](https://github.com/quantumlib/Qualtran)'s
`qualtran.rotation_synthesis` module -- the reference/oracle implementation
of the diagonal, fallback, mixed-diagonal, and mixed-fallback rotation
synthesis protocols from Kliuchnikov, Lauter, Minko, Paetznick, Petit,
["Shorter quantum circuits via single-qubit gate approximation"](https://arxiv.org/abs/2203.10064),
Quantum 7, 1208 (2023).

The output lets Rust tests in `rsgridsynth` differential-test the
(forthcoming) mixed-diagonal/fallback/mixed-fallback protocol
implementations against Qualtran's numbers **without requiring Python at
test/CI time** -- CI only ever reads the committed
`tests/fixtures/qualtran_reference.json`.

## Running it

From anywhere in the repo:

```sh
uv run --project tools/generate_qualtran_fixtures tools/generate_qualtran_fixtures/generate.py
```

(equivalently, `cd tools/generate_qualtran_fixtures && uv run generate.py`)

`uv` will create/reuse this directory's own `.venv` (see `pyproject.toml`,
which pins `qualtran>=0.7.0`) and install dependencies from `uv.lock` if
needed, then run the script. This requires network access to PyPI the
first time (or whenever the lockfile changes / the venv is missing); once
dependencies are cached by `uv`, no network is needed to re-run it.

The script writes `tests/fixtures/qualtran_reference.json` (repo-relative;
it locates the repo root from its own `__file__` path, not a hardcoded
absolute path), overwriting any previous contents.

## What it generates

For a grid of angles (`theta = 0.1` as an anchor point, plus
`k * pi / 32` for `k in 0..8`) crossed with epsilons
`{1e-4, 1e-6, 1e-8, 1e-10}`, it calls all four of:

- `qualtran.rotation_synthesis.diagonal_unitary_approx`
- `qualtran.rotation_synthesis.fallback_protocol`
- `qualtran.rotation_synthesis.mixed_diagonal_protocol`
- `qualtran.rotation_synthesis.mixed_fallback_protocol`

(the latter two protocols with `success_probability = 0.99`), and records
`t_count` (`channel.expected_num_ts(config)`), `diamond_distance`
(`channel.diamond_norm_distance_to_rz(theta, config)`), and a
protocol-dependent `q` field (success/mixing probability, where
applicable -- see the module docstring in `generate.py` for the exact
semantics per protocol). Any combination that raises inside Qualtran
(observed here only at the degenerate `theta = 0.0` point, and a handful of
other exact-zero-crossing points where Qualtran's own internal geometry
code hits a `ZeroDivisionError`/similar) is recorded with `"error"` set and
numeric fields set to `null`, rather than aborting the whole run.

Each row is tagged with a `"source"` field:

- `"qualtran_generated"`: produced by an actual run of this script against
  a real Qualtran install.
- `"paper_table_transcribed"`: copied by hand from the anchor table
  (`theta = 0.1`, `epsilon = 1e-8`) in the Stage 0 design document, which in
  turn cites the paper's own reported numbers.

See the top-level `"_provenance_note"` field inside the generated JSON
itself for the full, current story on how the transcribed rows relate to
the generated ones (including whether they've been cross-checked against a
live run in whatever environment last regenerated the file).

## Status of the currently-committed fixture

As of the last time this was run in the agent sandbox used for Stage 0
Track C, `uv run` **did** successfully reach PyPI and install
`qualtran==0.7.0` (this was initially expected to fail -- direct `curl` to
`pypi.org` times out in that sandbox, but `uv`'s own HTTP client evidently
does not go through the same restriction/TLS interception). The script ran
end-to-end and produced 144 freshly-generated rows (121 successful, 23
recorded errors, all at degenerate/exact-zero-crossing angles) plus the 4
hand-transcribed anchor rows. The anchor rows were checked against the
freshly-generated output for the same `(theta, epsilon, protocol)` and
matched to the displayed precision.

**This should not be assumed to hold in every environment.** If you run
this script somewhere without PyPI access and it fails during
`uv run`/dependency resolution, that is expected -- the committed
`tests/fixtures/qualtran_reference.json` already has a usable fixture from
the last successful run, including the `paper_table_transcribed` rows
which do not require Qualtran at all. When you *do* have an environment
with network access, please regenerate the file and re-verify the
transcribed anchor rows against the fresh output, and update this section
and the JSON's `_provenance_note` accordingly.
