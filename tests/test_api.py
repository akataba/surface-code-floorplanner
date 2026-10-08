"""Boundary and replay tests for the public integration API."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lattice_floorplan.api import optimize_layout
from lattice_floorplan.compiler import CompilerConfig
from lattice_floorplan.geometry import GridState
from fixtures import GRID

QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\nh q[0];\ncx q[0],q[1];\nt q[2];\n'


class ApiTests(unittest.TestCase):
    def test_cli_passes_adapter_and_local_configuration(self):
        from lattice_floorplan.optimize_cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.qasm").write_text(QASM)
            (root / "layout.txt").write_text(GRID)
            adapter = root / "liblsqecc_adapter"
            with patch("lattice_floorplan.optimize_cli.optimize_layout", return_value={"best_area": 81, "best_slices": 10, "best_objective": 810, "accounting": {}}) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([
                        str(root / "input.qasm"), "--layout", str(root / "layout.txt"),
                        "--adapter", str(adapter), "--out", str(root / "out"), "--local",
                    ]), 0)
                self.assertEqual(run.call_args.kwargs["adapter_path"], adapter)
                self.assertTrue(run.call_args.kwargs["compiler_config"].local)

    def test_invalid_budget_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            with self.assertRaises(ValueError):
                optimize_layout(
                    QASM, GridState.parse(GRID), adapter_path="missing", output_directory=out, budget=1
                )
            self.assertFalse(out.exists())

    def test_export_replay_and_accounting(self):
        def compile_fake(self, circuit, layout, directory, *args, **kwargs):
            Path(directory).mkdir(parents=True)
            return {"status": "success", "slices": 100}

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "binary"
            binary.write_text("mock")
            out = Path(directory) / "run"
            with patch("lattice_floorplan.compiler.Compiler.run", compile_fake):
                result = optimize_layout(
                    QASM, GridState.parse(GRID), adapter_path=binary, output_directory=out, budget=13
                )
            self.assertEqual(result["accounting"]["score_requests"], 13)
            self.assertEqual(result["replay_native_calls"], 1)
            self.assertEqual(result["replay"]["slices"], result["best_slices"])
            self.assertEqual(json.loads((out / "result.json").read_text())["schema_version"], 1)
            for name in [
                "layout.txt",
                "identity.json",
                "mapped-circuit.qasm",
                "original-circuit.qasm",
                "resources.json",
            ]:
                self.assertTrue((out / name).is_file())

    def test_replay_disagreement_fails_closed(self):
        def compile_fake(self, circuit, layout, directory, *args, **kwargs):
            Path(directory).mkdir(parents=True)
            return {"status": "success", "slices": 101 if Path(directory).name == "replay" else 100}

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "binary"
            binary.write_text("mock")
            with patch("lattice_floorplan.compiler.Compiler.run", compile_fake):
                with self.assertRaisesRegex(RuntimeError, "replay"):
                    optimize_layout(
                        QASM,
                        GridState.parse(GRID),
                        adapter_path=binary,
                        output_directory=Path(directory) / "run",
                        budget=13,
                    )

    @unittest.skipUnless(
        os.environ.get("LSQECC_ADAPTER"), "Set LSQECC_ADAPTER for actual export replay"
    )
    def test_real_compiler_export_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            result = optimize_layout(
                QASM,
                GridState.parse(GRID),
                adapter_path=os.environ["LSQECC_ADAPTER"],
                compiler_config=CompilerConfig(local=True),
                output_directory=Path(directory) / "run",
                budget=13,
            )
            self.assertEqual(result["replay"]["status"], "success")
            self.assertEqual(result["replay"]["slices"], result["best_slices"])
