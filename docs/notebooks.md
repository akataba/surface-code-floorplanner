# Runnable notebook tutorials

Start here to see the package execute a complete workflow. These three Python
notebooks explain setup, input paths, compiler settings and results. They use the
small supplied five-qubit Clifford+T circuit, with local compilation enabled.

| Notebook | What you learn |
|---|---|
| {download}`1. Native layout slices <../notebooks/01_native_layout_slices.ipynb>` | Generate all three native layout types and obtain completed slice counts, reporting failures explicitly |
| {download}`2. Optimizer layout slices <../notebooks/02_optimizer_layout_slices.ipynb>` | Run simulated annealing and re-evaluate the exported grid with its logical identities |
| {download}`3. Your own algorithm <../notebooks/03_custom_algorithm.ipynb>` | Connect a search function through a scoring callable and verify the exported winner |

Run these inside a source checkout; downloading a notebook alone does not include
the package, circuit fixture or native compiler. From the repository root:

```sh
python -m pip install -e '.[notebooks]'
jupyter lab notebooks
```

First build `liblsqecc_adapter` as explained in [the integration guide](integration.md).
Select the matching Python kernel, then **Restart Kernel and Run All**. Each
notebook is independent and creates a new evidence directory on every run.

## Where your algorithm connects

The custom-algorithm notebook defines `my_search(start, evaluate, requests, seed)`.
`start` is a `GridState`; `evaluate` is a callable accepting a candidate grid and
returning area times slices, or `None` on failure. Your algorithm chooses candidates
and decides which to keep. `GeometryOracle` handles relabeling, compilation and
cached measurements. No registration or subclass is required.

`optimize_layout` exposes the package's built-in methods; it is not a callable
plugin registry. Use the lower-level evaluator when supplying your own algorithm.
Export and fresh replay are demonstrated explicitly because the evaluator alone
does not perform the public optimizer's final verification.

These are integration tutorials, not evidence of an algorithm outperforming
another. For the all-family comparison see [benchmark results](local-results.md).
