# Your first optimization

For the meaning of tiles, slices, supply cells and logical-qubit mapping, see
[concepts and terminology](concepts.md).

This example uses a small five-qubit circuit already expressed in Clifford+T
gates. It is an integration example, not a benchmark superiority claim.
First [install the package](installation.md) and [build the liblsqecc adapter](integration.md).

## Run the example

The repository includes a [five-qubit circuit](_static/example.qasm) and a
[native EDPC starting grid](_static/edpc5.txt). From the repository root:

```sh
lattice-floorplan-optimize docs/_static/example.qasm \
  --layout docs/_static/edpc5.txt \
  --adapter .deps/adapter/liblsqecc_adapter \
  --out runs/first-example --method sa --budget 32 --seed 61 \
  --move-supply --local
```

`--method sa` selects **simulated annealing**, the randomized search described
in the [algorithm guide](algorithms.md). `--seed 61` initializes its random choices.

Use a new output directory each time. The adapter executable path is explicit;
replace it if you built elsewhere. `--move-supply` allows M-cell relocation while
holding the count fixed within this run. Grid dimensions and qubit assignment
can change. The objective is rectangular grid tiles times completed local slices.

## Inspect the result

Open `runs/first-example/layout.svg` to see the selected grid. Its `result.json`
contains `best_area`, `best_slices`, `best_objective`, search history and replay
status. Lower volume can mean fewer tiles even when slice count increases.

The **layout output folder**, also called a **bundle**, is simply a directory
of related files—not a special file format. Keep these files together because
the grid alone does not identify which logical qubit occupies each data location:

| File | Purpose |
|---|---|
| `layout.txt` | Compiler grid |
| `identity.json` | Logical coordinates and compiler-wire mapping |
| `mapped-circuit.qasm` | Circuit matched to the grid's row-major compiler wires |
| `original-circuit.qasm` | Original logical circuit |
| `resources.json` | Area and resource counts |
| `result.json` | Search result and independent replay evidence |

## Resume from an exported grid

Use the **original** QASM and saved identity mapping. Do not remap an already
mapped circuit a second time.

```sh
lattice-floorplan-optimize runs/first-example/original-circuit.qasm \
  --layout runs/first-example/layout.txt \
  --identity runs/first-example/identity.json \
  --adapter .deps/adapter/liblsqecc_adapter \
  --out runs/second-example --budget 64 --move-supply --local
```

## Understand coordinates without a native build

This executable documentation example validates a single-qubit grid and reports
its tile area. It performs no compilation:

```{doctest}
>>> from lattice_floorplan.geometry import GridState, valid_geometry
>>> grid = GridState.parse("rrr\nrQr\nrrr\n")
>>> grid.area, grid.qubits, grid.logical_to_site
(9, ((1, 1),), (0,))
>>> valid_geometry(grid)
True
```

For configuration and Python integration see [the API guide](api.md).
