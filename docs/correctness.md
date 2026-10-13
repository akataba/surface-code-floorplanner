# Computational correctness: what the tests establish

Successful compilation and matching slice counts do **not** prove that a generated
lattice-surgery schedule performs the intended quantum computation. The test suite
checks several distinct layers, and reports the remaining gap explicitly.

## Run the semantic checks

From the repository root, install the synthesis dependencies and build the adapter
as described in the integration guide. Then run:

```sh
python -m pip install -e '.[haar]'
python scripts/verify_semantics.py --adapter .deps/adapter/liblsqecc_adapter \
  --report runs/semantic-verification.json
```

This command fails if required dependencies are missing, any test fails, or any
selected test is skipped. It writes `supported_checks_passed` separately from
`full_compiled_execution_verified`. The latter is currently **false**, even when
all supported checks pass. Regular test discovery may skip optional tests; the
dedicated command prevents those skips being mistaken for verification.

## Coverage

| Layer | Tests and meaning |
|---|---|
| Actual compiler inputs | All 20 corrected QASM files across eight families must match the gate lists used by the synthesis tests. This checks serialization, not an independent target proof for synthetic workloads. |
| Circuit synthesis | Restored QFT, Haar, QSP and QSVT tests compare full matrices against targets and declared approximation tolerances using the corrected corpus. QSP polynomial/parity and QSVT projected-block checks also run. |
| Exact rotations | Tests verify positive/negative pi/4 multiples and that exactly representable rotations bypass approximate synthesis. |
| Logical placement | All six assignments on each of two three-qubit geometries are checked by full-unitary comparison, with explicit basis permutation. Export/reload preserves the identity mapping. Wrong-identity and reversed-CNOT examples must fail equivalence. |
| Abstract compiler lowering | Real adapter output for H, X, Z and logical BellBasedCNOT is compared to an independent dense-matrix reference on custom grids. Both grids must also compile locally. BellBasedCNOT is treated as an **ideal logical gate** here. |
| Reference measurement protocols | An independent ideal ZZ/XX/Z-measurement CNOT construction checks all eight outcomes with corrections. T and T-dagger injection each check both outcomes with conditional S or S-dagger. Branch operators must equal the target up to their branch amplitude, and sum to a trace-preserving channel. |
| Unsupported output | The small LLI interpreter rejects measurements, magic-state requests and decorated local traces. It cannot silently certify them as identity operations. |

The reference measurement protocols are mathematical fixtures, **not reconstructions
of the compiler's emitted local instructions**. They establish what a correct
branch-aware verifier must check; they do not close the final verification gap.
The synthetic line/star/matching/dense inputs have no new independent algorithmic
target proof in this suite. These are correctness tests, not a new all-family
performance comparison.

## Important limitation in the pinned compiler

At revision `82a1cbe64717a64a289dd4b0d02e3dc38cae3890`, the upstream
`src/ls_instructions/ls_instructions_from_gates.cpp` explicitly ignores `is_dagger`
in T lowering. The new native regression reproduces identical `--printlli before`
output for T and T-dagger, although their matrices are not equal up to global phase.
That test records a **known limitation**, not a successful T-equivalence check.
The same source rejects requested Pauli corrections for Bell-based CNOTs.

This is compatible with a scheduling/resource model that omits execution details,
but those details cannot be recovered by merely counting or replaying slices.
We therefore do not claim that the emitted instructions alone constitute a fully
specified executable quantum protocol. Reported area/time improvements remain
measurements within the declared compiler model, not independently verified
end-to-end quantum executions.

To close the gap, an execution representation must expose measurement outcomes,
conditional corrections or Pauli-frame updates, ancilla preparation/disposal, and
local operation semantics. An independent interpreter must then check the induced
logical channel across branches. Physical stabilizer cycles, noise and decoding
require additional verification beyond these ideal logical tests.
