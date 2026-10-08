# Python API and result bundles

A **result bundle** is a layout output folder containing related files, not a
special file format. It includes the grid (`layout.txt`), a picture (`layout.svg`),
logical-qubit positions (`identity.json`), the original and mapped circuit files,
and resource counts (`resources.json`). The public optimizer also saves its
results and compilation evidence there. Keep these files together: the grid alone
does not identify which logical qubit occupies each data location.

For the meaning of tiles, slices, supply cells and logical-qubit mapping, see
[concepts and terminology](concepts.md).

```python
from pathlib import Path
from lattice_floorplan.api import optimize_layout
from lattice_floorplan.compiler import CompilerConfig
from lattice_floorplan.geometry import GridState
from lattice_floorplan.tempering import TemperingConfig

initial = GridState.parse(Path("initial-layout.txt").read_text())
result = optimize_layout(
    Path("circuit.qasm").read_text(),
    initial,
    adapter_path="/path/to/liblsqecc_adapter",
    output_directory="runs/example",
    method="sa",  # simulated annealing; alternatives: restart_sa, parallel_tempering
    budget=256,
    seed=61,
    search_config=TemperingConfig(move_supply=True),
    compiler_config=CompilerConfig(local=True),
)
print(result["best_area"], result["best_slices"], result["best_objective"])
```

`GridState.rows` is an immutable tuple of equal-width ASCII rows. `qubits[i]` is
logical qubit i's `(row, column)` coordinate. When loading an optimized bundle,
pass its `logical_qubit_positions` as the second argument to `GridState.parse`.
Never reconstruct identities from row order unless that is the intended mapping.

`TemperingConfig` controls temperature calibration, replica count, swap frequency,
objective and optional M relocation. Defaults minimize area times slices. For
frontier exploration use `objective="augmented_chebyshev"` and a positive
`area_weight` strictly between zero and one. Normalization uses the initial grid
and its measured latency, fixed throughout that run. No area cap is imposed.

The search budget counts baseline and temperature probes, cache hits, failed
compilations and null proposals. The minimum accepted budget is
`1 + calibration_queries + replicas`. The public API adds one fresh winner replay
outside that budget. It does not tune parameters, synthesize circuits, initialize
supply counts or start MLflow implicitly. The experimental count runner owns those
outer choices. Reinforcement-learning training remains in `geometry_rl.py` and research runners;
policy checkpoints trained on fixed grids require compatibility checks before
use with variable geometry.

## Schema version 1

| Artifact / field | Meaning |
|---|---|
| layout.txt / layout.svg | Actual grid and visual rendering |
| identity.json | Logical positions and compiler-site mapping |
| original-circuit.qasm | Unmapped caller circuit |
| mapped-circuit.qasm | Circuit matched to this grid's row-major compiler sites |
| resources.json | Height, width, area and tile counts |
| result.json | Search settings, history, measured winner and replay |
| best_area / best_slices | Grid tiles and completed wave frames |
| best_objective | Product by default; Chebyshev score when explicitly selected |
| baseline / baseline_area | Initial measured slices and area |
| history | Charged queries, phases, proposals, acceptance and scores |
| accounting | Score requests, new native compilations and cache hits during search |
| replay_native_calls | Additional winner verification call, currently one |
| pareto/ | Observed nondominated layouts, each with its own matched bundle |
| native-evidence/ | Content-addressed compiler records for search queries |
| replay/ | Independent winner compilation evidence |

The return value is a JSON-compatible dictionary, matching result.json. New
optional fields may be added; incompatible field/meaning changes require a schema
version change. Internal research-result dictionaries are not all schema-versioned.
A failed run may leave partial evidence; absence of an exception is required for
successful API completion. A run can have a lower product but more slices.

Low-level interfaces remain available: `Compiler.run` for native/custom compilation,
`GeometryOracle.evaluate` for cached grid evaluation, and `tempering.search` for
in-memory searches against a duck-typed oracle. These do not automatically perform
the public API's export replay.
