#!/usr/bin/env python3
"""Audit saved local benchmark commands, identities and fresh replay evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from lattice_floorplan.geometry import GridState, remap_qasm
from lattice_floorplan.supply import supply_count


def main():
    """Audit the saved 20-circuit local study and write verification.json.

    Check frozen-source hashes, local/wave flags, null failure scores and the
    pinned slice-counter convention. Verify each saved successful count winner
    against its existing replay, area and mapped-QASM/identity bundle. This
    reads evidence, not a new compilation; failed checks raise AssertionError.

    Raises:
        AssertionError: Saved coverage, hashes, compiler mode, mapping or replay checks fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    result = json.loads((root / "results.json").read_text())
    assert len(result["circuits"]) == 20
    assert len({c["family"] for c in result["circuits"]}) == 8
    assert sum(len(c["runs"]) for c in result["circuits"]) == 280
    for rel, expected in result["protocol"]["source_hashes"].items():
        assert hashlib.sha256((root / "source_snapshot" / rel).read_bytes()).hexdigest() == expected
    records = 0
    for path in root.rglob("result.json"):
        # The four deliberately paired smoke compilations include nonlocal controls.
        if path.parent.name.startswith("smoke-"):
            continue
        record = json.loads(path.read_text())
        assert record["config"]["local"] is True, path
        assert "--local" in record["command"], path
        assert record["config"]["pipeline"] == "wave", path
        if record["status"] == "success":
            assert record["reported_slices"] == record["slices"] + 1, path
        else:
            assert record["slices"] is None, path
        records += 1
    winners = 0
    for circuit in result["circuits"]:
        for point in circuit["by_count"]:
            if not point.get("replay_verified"):
                continue
            bundle = root / point["output"]
            grid = GridState.parse(
                (bundle / "layout.txt").read_text(),
                json.loads((bundle / "identity.json").read_text())["logical_qubit_positions"],
            )
            assert supply_count(grid) == point["count"]
            assert grid.area == point["area"]
            assert (
                remap_qasm((bundle / "original-circuit.qasm").read_text(), grid)
                == (bundle / "mapped-circuit.qasm").read_text()
            )
            replay = json.loads((bundle / "fresh-replay/result.json").read_text())
            assert replay["status"] == "success" and replay["slices"] == point["slices"]
            assert point["volume"] == point["area"] * point["slices"]
            winners += 1
    audit = {
        "circuits": 20,
        "families": 8,
        "runs": 280,
        "local_compiler_records_checked": records,
        "per_count_winner_replays_verified": winners,
        "source_snapshot_hashes": "passed",
        "identity_and_export_pairs": "passed",
        "local_config_and_command": "passed",
        "tests": (root / "tests.log").read_text().splitlines()[-4:]
        if (root / "tests.log").exists()
        else "No test log supplied",
    }
    (root / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
