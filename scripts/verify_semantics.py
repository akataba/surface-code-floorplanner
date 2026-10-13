#!/usr/bin/env python3
"""Run semantic regression tests without allowing missing dependencies or skips."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest


def main():
    """Run focused semantic checks and write their scope and result as JSON.

    Returns:
        int: Zero if all supported checks pass without skips, otherwise one.

    Raises:
        SystemExit: Required adapter or optional synthesis dependencies are missing.
        OSError: The requested report file cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("runs/semantic-verification.json"))
    args = parser.parse_args()
    if not args.adapter.is_file():
        parser.error("Build liblsqecc_adapter and supply its executable path")
    missing = [m for m in ("numpy", "qiskit", "pygridsynth") if not importlib.util.find_spec(m)]
    if missing:
        parser.error("Install the haar extra before verification; missing: " + ", ".join(missing))
    os.environ["LSQECC_ADAPTER"] = str(args.adapter.resolve())
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    names = [
        "test_semantic_mapping",
        "test_semantic_corpus",
        "test_measurement_semantics",
        "test_native_lli_semantics",
        "test_qft5",
        "test_haar",
        "test_structured",
        "test_exact_synthesis",
    ]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and not result.skipped
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "supported_checks_passed": passed,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "full_compiled_execution_verified": False,
                "limitations": [
                    "BellBasedCNOT is interpreted as an ideal gate in the abstract LLI test.",
                    "Reference measurement gadgets are not reconstructed from emitted local instructions.",
                    "Pinned T and T-dagger lowering is indistinguishable; full T execution is not certified.",
                    "Physical stabilizer cycles, noise, and decoding are not simulated.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
