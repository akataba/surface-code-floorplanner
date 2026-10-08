# Data provenance and availability

The published results describe the **20-circuit local-compilation benchmark**:
custom grids optimized by simulated annealing and compared with liblsqecc's built-in
layouts. Three stages of earlier work matter when interpreting these results.

1. **Fixed-grid mapping experiments** changed qubit assignments on an unchanged
   grid. They did not optimize grid dimensions or resource placement, so they are
   not evidence of improved geometry.
2. **Nonlocal geometry experiments** changed the grid but ran without liblsqecc's
   `--local` decomposition. Their durations are not interchangeable with the
   local-compilation scores published here.
3. **Local-compilation experiments** enabled `--local` with wave scheduling and
   measured the native baselines afresh. Some initial candidates came from the
   earlier nonlocal search and were re-evaluated under the new configuration.

Those 140 previously optimized layout/identity bundles remain reproducibility
inputs under `examples/benchmark-starts`, with identifying hashes.
Only 22 compiled successfully in the local configuration. Their prior search cost
is additional to the current benchmark budget, so this is not an equal-total-cost
comparison between the custom optimizer and native generators.

## Why the circuit directory is named corrected-v3

Circuit synthesis converts operations into the Clifford+T gate set accepted by
the workflow. An earlier synthesis path unnecessarily approximated rotations
that could be represented exactly by multiples of pi/4. The revised path handles
those exact rotations directly, using a declared numerical tolerance and
verification against the target operation. The input manifest records which
circuits changed. Results from different synthesis versions must not be mixed:
extra synthesized gates change the workload being compiled.

## What is available

This site's [results page](local-results.md) and downloadable JSON contain all
eight families, selected per-circuit scores, baseline failures and seed variation.
The full experiment archive is not bundled with the documentation. Reproduction
requires the versioned circuit corpus, starting-layout bundles and native toolchain
specified in the [benchmark protocol](local-compilation.md).

Earlier mapping and nonlocal performance reports are no longer in the working
experiment archive. Historical research notes may remain in a source checkout,
but are excluded from this site's navigation and search. Their absence does not
remove the starting-layout cost described above. Older research runners may require
unavailable input tables, and fixed-grid policy checkpoints cannot be assumed to
work with variable-size layouts.
