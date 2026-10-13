# Publication scope and validation

Prepared 2026-10-08 as an independent source repository. No remote, commit or
upload has been created. The original project files were untracked in a parent
repository; there was no project commit history to transfer.

## Included

- Complete Python package and native adapter/exporter source.
- Three self-contained notebooks, with saved outputs cleared.
- Curated documentation, small result summary and Read the Docs configuration.
- Native build tools and the local-compilation benchmark/report/verification tools.
- Portable core/integration tests, separated from optional research test imports.
- All 20 corrected benchmark circuits and 140 hashed starting-layout pairs.

Earlier search cost is inherited by those starting layouts. Their original
optimization runs are not reproduced by this checkout. The benchmark runner uses
`examples/benchmark-starts`; the full original compiler evidence is not included.
Historical source-path fields in the circuit manifest record provenance only.

## Excluded

Python environments, compiler binaries/checkouts, MLflow databases, full experiment
outputs, caches, internal conversation notes, cleanup records, obsolete experimental
runners and copied upstream examples. Optional research modules remain in the
package; the portable test set does not cover every research training workflow.

## Validation

- 44 tests passed with actual compiler/adapter integration enabled, no skips.
- All nine notebook code cells ran from this checkout against the rebuilt adapter.
- Strict Sphinx HTML build passed; Python wheel built successfully.
- Verified hashes for 20 circuits and 140 starting-layout/identity pairs.
- Scanned distributable text for local home-directory paths and private conversation
  URLs. This is a focused publication review, not a comprehensive security audit.

The adapter was rebuilt from the copied source against an existing pinned native
library. A fresh upstream download/system-dependency installation and Jupyter UI
launch were not performed. Native builds remain platform-dependent.

Generated `.deps`, `runs`, build files and caches used for these checks are ignored
by Git and excluded from the publication ZIP.

## Before publishing

The project license remains undecided at the owner's request; see NOTICE.md.
Choose a GitHub repository owner/name and public/private visibility before upload.
Treat this folder as a separate copy: later edits to the original project do not
automatically propagate here.

## Semantic test update — 2026-10-12

The publication now includes the corrected-corpus synthesis tests and their three
generator helpers, independent unitary/mapping tests, reference measurement gadgets,
and real abstract-LLI regression checks. `verify_semantics.py` passed 23 tests with
no skips. Full compiled quantum execution remains unverified; see
[the correctness guide](docs/correctness.md), including the pinned T/T-dagger
information-loss limitation. Passing these tests does not certify the local
measurement implementation or supply missing outcome-dependent corrections.
