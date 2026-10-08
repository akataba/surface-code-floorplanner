# Lattice Floorplan

## Start here: runnable Python notebooks

Follow these notebooks in order. Each includes setup instructions and explains
its inputs, compiler settings, outputs and failure handling.

1. [Measure slices for native liblsqecc grids](notebooks/01_native_layout_slices.ipynb).
2. [Optimize a grid and evaluate its saved output](notebooks/02_optimizer_layout_slices.ipynb).
3. [Connect your own algorithm](notebooks/03_custom_algorithm.ipynb).

Install with `python -m pip install -e '.[notebooks]'`, build the adapter using
the [integration guide](docs/integration.md), then run `jupyter lab notebooks`.
See the [notebook guide](notebooks/README.md) for troubleshooting.


Generate actual lattice-surgery grid layouts and evaluate them with **liblsqecc**.
The current objective is **grid area × completed time slices**, with the observed
area/time frontier retained. Geometry, dimensions and initial logical-qubit
placement can change without a hard area cap.

## Start here

Python 3.10+ and a C++20 toolchain are required. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The optimization package requires NumPy. Circuit synthesis (`.[haar]`) and RL
(`.[rl]`) are optional; running annealing does not import PyTorch, Optuna or MLflow.
See [native build and replay instructions](docs/integration.md) to prepare the
compiler. Given a built adapter and an exported initial grid:

```sh
lattice-floorplan-optimize circuit.qasm --layout initial-layout.txt \
  --adapter /path/to/liblsqecc_adapter --out runs/example \
  --method sa --budget 256 --seed 61 --move-supply --local
```

Input must already be synthesized to the supported Clifford+T OpenQASM 2 gate
set, with one quantum register. When continuing from a previous output, also pass
`--identity previous/identity.json`. Omitting it assigns logical qubits to Q cells
in row-major order. The output directory must be new.

## Outputs and integration

Each result bundles `layout.txt`, `layout.svg`, `identity.json`,
`original-circuit.qasm`, `mapped-circuit.qasm`, `resources.json`, `result.json`,
compiler evidence and observed Pareto layouts. The installed command performs a
fresh compilation of the winning exported grid/QASM pair before returning success.

- [Python API and artifact fields](docs/api.md)
- [liblsqecc build, preprocessing, replay and compatibility](docs/integration.md)
- [Architecture, tests and contribution guidelines](docs/development.md)
- [Data provenance and availability](docs/history-status.md)

## Scientific scope

The current local study uses **EDPC preprocessing with wave scheduling and
explicit local compilation (`--local`)**, not the full EDPC scheduling algorithm.
The library default remains nonlocal for backward compatibility: pass
`CompilerConfig(local=True)` or the command-line `--local` flag explicitly.
Historical frontier/supply reports used nonlocal compilation. Stock `-l layout.txt` does not reproduce this contract;
use the documented adapter. The adapter is pinned to upstream revision
`82a1cbe64717a64a289dd4b0d02e3dc38cae3890`.

Retained search constraints: static single-cell interior Q patches, connected
routing, fixed resource count within a run. Supply relocation is opt-in; supply
count selection is an outer experimental sweep. M cells are abstract sources,
not physical distillation-factory footprints. Replenishment remains deterministic
with latency 10. Tile-slices are not physical qubit-cycles.

Current corrected workloads are in `examples/corrected-v3/manifest.json`: line,
star, matching, dense, QFT, Haar, QSP and QSVT. Method comparisons must include all
eight families, native compact/compact_no_clogging/EDPC references, failures,
resource differences, seed variation and all search/training/tuning costs.
Use Optuna for declared tuning and MLflow for experiments. The single-circuit API
is an integration entry point, not an all-family benchmark or an automatic tracker.

See the [20-circuit local-compilation benchmark](docs/local-results.md), covering all eight
circuit families. The full evidence archive is not included in this repository; the small starting-layout inputs are in `examples/benchmark-starts`.
Nonlocal experiment outputs have been removed; historical code is not a new result.

## Documentation site

The Sphinx site includes typed API reference pages, integration instructions and
local results. See [building and publishing](docs/publishing.md). Read the Docs
configuration and a documentation CI workflow are included; hosting requires
connecting the intended Git repository to Read the Docs.

### Adapter interface

Build `liblsqecc_adapter` with `scripts/build_liblsqecc_adapter.py`. Supply its
location using `--adapter` on the command line or `adapter_path` in Python.
This is a breaking interface rename; older executable names and command-line
options are not provided as aliases. Saved experiment evidence is unchanged.
