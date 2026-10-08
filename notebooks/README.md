# Running the example notebooks

These notebooks are the recommended starting point for using the package.
They use a small five-qubit Clifford+T circuit and actual compiler calls.
They are tutorials, not all-family performance comparisons.

1. `01_native_layout_slices.ipynb`: generate EDPC, compact and compact_no_clogging
   grids, compile each using its matching preprocessing, and inspect time slices.
2. `02_optimizer_layout_slices.ipynb`: run simulated annealing, load its exported
   layout with logical identities, and independently measure the duration.
3. `03_custom_algorithm.ipynb`: implement a Python search function receiving a
   scoring callable; export and recompile the result.

## Setup

From the repository root, activate your Python environment and run:

```sh
python -m pip install -e '.[notebooks]'
python scripts/bootstrap.py --jobs 4
python scripts/build_liblsqecc_adapter.py --source .deps/liblsqecc --build .deps/liblsqecc/build --out .deps/adapter
jupyter lab notebooks
```

See [native prerequisites](../docs/integration.md) before building. Choose the
Python kernel for the environment containing this package. Open a notebook and
choose **Restart Kernel and Run All**. The notebooks are independent; notebook 2
does not depend on the files created by notebook 1. Set ADAPTER and EXPORTER in
the setup cell if your executable paths differ.

Each run creates a new `runs/notebook-*` directory. Compiler invocations store
commands, input files, diagnostics and scores there. Saved example outputs are
illustrative: rerunning creates a different directory. You may clear notebook
outputs without affecting the code. Full frame JSON is saved by notebooks 1 and 3;
notebook 2 demonstrates summary scoring and verified export.

## Common problems

- **ModuleNotFoundError:** install the package in the selected kernel's environment.
- **Missing executable:** build the adapter, or update ADAPTER and EXPORTER paths.
- **Repository not found:** launch Jupyter from the checkout root or notebooks folder.
- **Failed compilation:** inspect the returned status and evidence diagnostics;
  never interpret a missing slice count as zero. Native compact layouts can fail
  with local decomposition even when EDPC succeeds.
- **Incorrect qubit placement:** preserve `identity.json` when loading a custom
  layout. The oracle takes original QASM; direct exported replay takes mapped QASM.
- **Running larger circuits:** supply already synthesized supported QASM, match
  the number of data sites, and choose an appropriate timeout. These examples do
  not automatically synthesize arbitrary rotations.

## Validation

All nine code cells were executed sequentially with separate namespaces for each
notebook against the real pinned adapter. Assertions verify successful EDPC
compilation, optimizer replay, and custom-algorithm replay. Validation used Python
cell execution; a Jupyter browser/kernel launch was not part of that check.
