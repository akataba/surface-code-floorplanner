# Local compilation benchmark

This protocol measures how well **simulated annealing** can reduce rectangular
grid area multiplied by compiled duration. Area is measured in abstract tiles,
duration in completed compiler slices, and their product in tile-slices. See
[concepts and terminology](concepts.md) for definitions and
[the results](local-results.md) for every circuit and native baseline.

All evaluations use `-P wave --local`: wave scheduling plus explicit local
instruction decomposition. Selecting the `edpc` layout generator does not enable
the separate `-P edpc` scheduler. Magic-state supplies are abstract `M` cells with
fixed `distillation_time=10` compiler timesteps, not physical factory simulations.

## Inputs and comparisons

The input set comprises 20 Clifford+T circuits across eight families, identified
by `examples/corrected-v3/manifest.json`. The name records a synthesis revision
that avoids unnecessary approximation of exactly representable rotations;
[data provenance](history-status.md) explains the change.

Three native layout generators—`edpc`, `compact` and `compact_no_clogging`—are
measured on the same inputs with local compilation enabled. Exported native grids
are replayed using their corresponding preprocessing mode. Failed compilations
receive no numeric score. Each custom result is compared with the lowest-volume
**successful** native result for that circuit.

## Starting layouts and inherited search cost

A **warm start** is a layout found by an earlier optimization and reused as a
starting candidate. This benchmark includes 140 such layouts from an earlier
search that omitted `--local`: one for each circuit and tested supply count.
They were all re-evaluated with local compilation enabled; only 22 succeeded.
Earlier nonlocal scores are not used in the result tables.

Each run compares that prior layout with a native EDPC grid adjusted to the same
supply count. Counts above 16 begin with spaced outer supply rows. This is a
starting-layout heuristic, not a restriction on the dimensions explored later.
The earlier optimization cost remains additional to the budget below. Consequently,
the experiment reports best observed designs, not an equal-total-budget contest
between optimization and the native generators.

## Search and verification budget

- Seven supply counts: 1, 2, 4, 8, 16, 24 and 32; two random seeds: 61 and 62.
  Across 20 circuits, this gives 280 simulated-annealing runs.
- Each run has 32 charged budget slots. Two evaluate starting-layout choices;
  the remaining 30 requests include the baseline and eight temperature-calibration
  probes. Invalid construction still consumes its initialization slot. A run
  without a successfully compiled initialization stops and is reported.
- Seven equally selected proposal families cover supply relocation, logical-qubit
  assignment and geometry changes, compound moves, and row/column insertion or
  deletion. Supply count stays fixed within a run; separate runs compare counts.
  There is no hard area cap.
- Every successful per-count winner receives a fresh compilation of its exported
  grid and matched circuit. The per-count selection pools the two seeds, giving
  140 winner replays. Per-seed outcomes are also reported.
- Prior-layout evaluations, native/export checks and winner replays are additional
  to the search slots. A request answered from cache avoids a compiler invocation.
  Requests, cache hits, compiler calls and replays are recorded separately.

The short searches and two seeds do not establish a global optimum or a typical
performance distribution. No **hyperparameter optimization** (search over algorithm
settings) was performed; no Optuna tuning study is claimed. MLflow stores the
experiment settings and results under `local-wave-corrected-layouts`.

## Reproduction

From an environment with this package, NumPy and MLflow installed:

```sh
python scripts/benchmark_local.py \
  --native /path/to/lsqecc_slicer \
  --adapter /path/to/liblsqecc_adapter \
  --exporter /path/to/export_layout \
  --out validation/new-local-wave
python scripts/report_local.py validation/new-local-wave
python scripts/verify_local.py validation/new-local-wave
```

The runner refuses to reuse an existing protocol directory. Paths are explicit;
there is no dependency on the original workspace location. The preserved
`examples/benchmark-starts/manifest.json` and its 140 layout/identity
bundles are required as declared warm starts. These are included in the source repository, but not in the documentation site. Corrected inputs under `examples/corrected-v3` are also required. Source snapshots and
binary/input hashes record the exact implementation used for the benchmark.

The local flag is stored in `CompilerConfig` and the cache key. The API default
remains `local=False` for compatibility; use `CompilerConfig(local=True)` or
`lattice-floorplan-optimize ... --local` explicitly for new local-mode work.

## Interpretation and review

Gemini provided an AI-assisted protocol critique concerning starting-layout cost,
short budgets and failures under local decomposition. This was an editorial and
methodological review, not independent execution or verification of the experiment.
Measured claims come from the saved compiler records and winner replays.

A change in the best observed supply count does not establish its cause. Routing
congestion and supply starvation are possible explanations, but stall causes were
not measured. Some compact-layout failures report that the route selected by
`BellBasedCNOT` is too short for its implemented local decomposition. Such a failure
is a failure of that compilation attempt, not proof that no valid layout exists.
Partial schedules are discarded rather than reported as successful durations.
