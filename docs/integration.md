# Integrating with liblsqecc

This adapter connects the Python optimizer to the compiled C++ liblsqecc
executable. The **liblsqecc_adapter** allows a custom grid to replace a built-in
layout while preserving the selected compiler preprocessing. Read
[concepts and terminology](concepts.md) for layout, mapping and scheduling terms.

Build this executable once, then reuse it with the supplied optimizers or your
own layout-search algorithm. The build below produces `.deps/adapter/liblsqecc_adapter`;
pass that path as `adapter_path` in Python or `--adapter` on the command line.

## Build contract

The supported upstream commit is `82a1cbe64717a64a289dd4b0d02e3dc38cae3890`.
A binary hash identifies the actual executable in every measurement; a commit
string alone does not certify an arbitrary user-supplied binary. Keep build.json
with the binaries. Neither Python nor the adapter modifies the upstream checkout.

Use an existing pinned CMake **Unix Makefiles** build, or run:

```sh
python scripts/bootstrap.py --jobs 4
python scripts/build_liblsqecc_adapter.py \
  --source .deps/liblsqecc --build .deps/liblsqecc/build --out .deps/adapter
.deps/adapter/export_layout edpc 5 > initial-layout.txt
```

Bootstrap needs Git, CMake, a C++20 compiler and upstream development libraries
(including the dependencies named by upstream CMake). On macOS, pass a dependency
prefix with `--prefix /path/to/prefix` if discovery fails. Bootstrap downloads the
pinned checkout and submodules; the adapter build itself works offline against an
existing build. The adapter script currently reads Makefile flags/link metadata;
Ninja/MSVC builds are not supported by this helper. Paths are explicit; there is
no requirement to use the original experiment workspace.

The adapter recompiles the compact and EDPC generator translation units with
renamed original symbols, then provides wrappers linked with the pinned static
library. `--geometry-file` substitutes the layout while retaining the selected
native `-L` preprocessing. The original generators remain usable without the
flag. This is a subprocess integration, not a stable C++ ABI or upstream plugin.

## Correct replay

Use one QASM declaration or instruction per line: the pinned upstream parser
is line-oriented and can reject multiple statements on one line.

Keep the complete **layout output folder (bundle)** together. A bundle is an
ordinary directory containing the grid, its logical-qubit identities, matching
circuit files and resource metadata; it is not a special file format. The ASCII file encodes physical cells;
`identity.json` records which logical qubit occupies each Q cell. The compiler
assigns Q sites in row-major order, so `mapped-circuit.qasm` is essential when
logical placement changes. This relabeling does not insert physical SWAP gates.
Do not remap the mapped circuit again.

```sh
/path/to/liblsqecc_adapter -i runs/example/mapped-circuit.qasm -I qasm \
  -L edpc --geometry-file runs/example/layout.txt -P wave --local \
  -r graph_search -g djikstra --disttime 10 --graceful \
  -f machine --noslices -o /tmp/replay-slices.json
```

`--local` enables pairwise local decomposition. Include it only when reproducing
a local-mode result; historical nonlocal results omitted it. Local mode is saved
in CompilerConfig and participates in cache identity. `-P wave --local` does not
enable the distinct `-P edpc` routing pipeline.

`djikstra` is the upstream flag spelling. Staggering is enabled unless
`--nostagger` is passed. Match all saved compiler settings when replaying.
The pinned wave summary reports **completed frames plus one**. Python subtracts
one in summary-only mode; when retaining slices it counts JSON frames and checks
the summary against them. This convention is not portable to arbitrary revisions
or other scheduling pipelines.

Plain upstream `-l layout.txt` uses a different preprocessing path. A file can be
syntactically accepted there without reproducing the experiment. The geometry
oracle deliberately uses `-L edpc --geometry-file ... -P wave`.

## Failure and provenance

Nonzero exit, diagnostics on stderr, malformed machine output, invalid frames and
timeouts are failures, never favorable scores. Each invocation records command,
configuration, executable/input/layout hashes, stdout, stderr and result.json.
Timeouts use Python's subprocess timeout handling, which kills and reaps the
invoked process. There is no general descendant-process-tree supervisor.

Geometry cache keys include original and mapped QASM, ASCII geometry, logical
identity, move-set version, compiler binary hash and configuration. Cache locking
is thread-local to one oracle instance; sharing the same cache directory across
independent processes is unsupported. The public API uses a new directory per run.

For an upstream revision upgrade, rebuild the adapter, review generator signatures
and lowering semantics, rerun native counter/mapping/failure tests, and replay
representative corrected circuits from all eight families. Never transfer cached
scores or claim equivalent scheduling solely because the build succeeds.
