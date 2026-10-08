# Lattice Floorplan

**Generate lattice-surgery grids and minimize grid area × compiled time slices.**

Area is the full rectangular grid size in abstract tiles. Time slices count
completed compiler schedule frames. Their product, measured in tile-slices, is
the optimization objective.

Lattice Floorplan searches over grid dimensions, data-patch placement and logical
qubit assignment, evaluates candidates with liblsqecc, and exports a layout with
its matching circuit and identity map. The supported integration API provides
simulated annealing, independent-restart annealing and parallel tempering.

Start with [concepts and terminology](concepts.md), then [installation](installation.md) and the [first optimization](quickstart.md).
The [API reference](reference/index.rst) is generated from the Python docstrings,
including parameter types, return values and failure contracts.

```{important}
Current evaluations use **wave scheduling with local compilation enabled**.
Pass `CompilerConfig(local=True)` or `--local` explicitly; the library retains
its older nonlocal default. Tile-slices measure the declared compiler model,
not physical qubit-cycles or full magic-state factory cost.
```

## What you get

- An actual ASCII grid and SVG illustration, with variable dimensions and no hard area cap.
- Matched original/mapped QASM and an explicit logical-qubit identity file.
- Search history, observed area/time frontier and native compiler evidence.
- A fresh replay of the winning exported grid/circuit pair.

The [20-circuit local-compilation benchmark](local-results.md) compares custom
layouts with built-in liblsqecc layouts across eight circuit families. Its results are best observed under the stated budgets,
initializations and abstract resource model—not a general optimality claim.

```{toctree}
:maxdepth: 2
:caption: User guide

concepts
installation
notebooks
quickstart
integration
api
algorithms
local-compilation
local-results
history-status
```

```{toctree}
:maxdepth: 2
:caption: Reference and development

reference/index
development
publishing
```
