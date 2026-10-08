# Installation

Use a source checkout of this project. From its root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
lattice-floorplan-optimize --help
```

The package declares Python 3.10 or newer. Dependencies and optional research
extras can impose narrower interpreter compatibility. The documentation build
is pinned separately to Python 3.14. The base package uses NumPy; annealing does
not import PyTorch, Optuna or MLflow.

## Native compiler

“Native” here means the compiled C++ executable that evaluates a layout, rather
than the Python code that proposes layouts.

Actual optimization requires the pinned liblsqecc compiler and this project's
liblsqecc adapter. Follow [the native build instructions](integration.md) for Git,
CMake, C++20, upstream development libraries and the Unix Makefiles requirement.
The Python package does not download or build the native compiler when imported.

## Optional development and research dependencies

Install only the extras needed by your task:

| Extra | Purpose |
|---|---|
| `pip install -e '.[dev]'` | Formatting and lint checks |
| `pip install -e '.[qft]'` | Rotation synthesis helpers |
| `pip install -e '.[haar]'` | Haar-random unitary circuit generation and synthesis |
| `pip install -e '.[rl]'` | PyTorch reinforcement-learning policies, Optuna tuning and MLflow tracking |
| `pip install -e '.[experiments]'` | Optuna and MLflow tracking |

Historical mapping-table experiments require their own input tables; installing
an extra does not restore removed experiment data. See [data provenance](history-status.md).

## Build these docs only

A documentation build requires no native compiler, Torch, MLflow database or
experiment archive. See [building and publishing](publishing.md).
