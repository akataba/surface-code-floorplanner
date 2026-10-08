# Development guide

## Module boundaries

| Module | Responsibility |
|---|---|
| api / optimize_cli | Public orchestration, input checks, export and fresh replay |
| compiler | Fail-closed subprocess adapter and compiler configuration |
| layout | ASCII grammar, resource counting and SVG rendering |
| geometry | Immutable state/identity, reversible proposals, remapping, evaluation cache |
| supply | Explicit-count M initialization and relocation proposals |
| tempering | Simulated annealing, independent restarts and parallel-tempering exchanges, objective, budget and frontier |
| geometry_rl | Variable-grid policy, observations and PPO research implementation |
| synthesis | Exact-rotation recognition and synthesis helpers |
| scripts | Experiment protocols, reports, synthesis and native builds |
| native | Thin pinned C++ liblsqecc adapter and generator exporter |
| validation | Recorded evidence and frozen historical source snapshots |

Dependencies point from orchestration to algorithms and adapters. Algorithms accept
an oracle with `evaluate(qasm, grid)` returning compiler records; they do not shell
out directly. Keep MLflow, Optuna and plotting in experiment orchestration. Avoid
introducing an abstract framework where a small explicit function will suffice.

The original `lattice-floorplan` command and mapping modules remain for old workflows.
Use `lattice-floorplan-optimize` for current variable-geometry integration. Historical
scripts are research protocols with their own path assumptions, not portable public
APIs. Their defaults must not silently become a new benchmark protocol.

## Checks

```sh
python -m pip install -e '.[dev]'
ruff check src
python -m unittest discover -s tests
LSQECC_SLICER=/path/to/lsqecc_slicer LSQECC_ADAPTER=/path/to/liblsqecc_adapter \
  python -m unittest discover -s tests
```

This publication includes the portable core/integration tests. Historical research
runners and their tests are excluded; optional research modules remain inspectable. Native
tests skip with explicit messages unless binaries are supplied. Do not report a
skipped native test as validation against liblsqecc. Formatting checks cover active
sources only; never reformat archived `validation/**/source_snapshot` files.

For changes to proposals or objectives, test reverse probabilities, exact budget
accounting, identity-sensitive cache keys, seed repeatability, failed-candidate
handling and dimension changes. Preserve RNG call order during readability-only
refactors. Changing it needs a new experiment version and results, even if the
algorithm is mathematically equivalent. Sorting every container retroactively can
change historical seeded trajectories; inspect actual sampling pools first.

For compiler/API changes, test diagnostics, timeout, malformed output, counter
conventions, mapping, output bundles and fresh replay disagreement. Performance
claims require a separate all-eight-family experiment; unit tests are not that
comparison. Track declared hyperparameter optimization in Optuna and trials/results in MLflow with explicit
training, search, cache and verification costs.

Use descriptive names, normal four-space formatting, explicit configuration and
short responsibility-focused functions. Add docstrings describing units, invariants,
failure behavior and side effects. Public APIs should be typed. Do not blanket-catch
exceptions or turn compiler failures into numeric rewards. Do not add filesystem,
network or tracking side effects on module import. Protect entry points with main.

## Typed Google-style docstrings

Follow the [Google Python docstring convention](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).
For this repository, explicitly include each non-self/cls parameter as
`name (type): description` in `Args`, even when the signature has annotations.
Describe optional values, units, allowed values and array/tensor shapes. Use
`Returns` (or `Yields`) with the result type and meaning, and `Raises` for applicable
exceptions. Do not describe caught compiler failures as propagated exceptions.
Document dataclass constructor fields using typed `Attributes` entries. Functions
with no parameters need no empty Args section; functions returning only None need
no empty Returns section. Keep mutation, file writes, RNG effects and cache/budget
semantics in the prose.

```python
def exact_rz_sequence(angle, tolerance=1e-12):
    """Recognize a Clifford+T Z rotation within an angular tolerance.

    Args:
        angle (float): Rotation angle in radians, convertible to float.
        tolerance (float): Nonnegative absolute tolerance in radians.

    Returns:
        tuple[tuple[str, ...], float] | None: Temporal gate names and snapping
            error, or None when the angle is not recognized.
    """
```

Run both checks:

```sh
ruff check src
python scripts/check_docstrings.py
python -m unittest discover -s tests -p 'test_docstrings.py'
```

The contract checker covers all package definitions (including nested helpers),
dataclass fields and the active local benchmark/report/verifier/checker scripts.
It detects missing or mismatched typed Args, missing typed Returns/Yields, and
missing directly raised exception names. These are structural checks: reviewers
must verify actual types, shapes, constraints and exception propagation against
implementation. The checker has regression tests for missing types, renamed
parameters, missing return/exception sections, nested scopes and dataclass fields.
Historical experiment scripts and tests are outside this package contract audit;
frozen experiment source snapshots must remain untouched.

## Compatibility and limitations

The public API currently covers annealing and parallel tempering. Research RL
interfaces remain experimental. No compatibility promise is made for arbitrary
upstream revisions or serialized historical policy checkpoints. The geometry
cache supports threads within one process, not concurrent independent processes.
Physical factory footprints, probabilistic failures and physical-qubit-cycle costs
remain future modeling work.
