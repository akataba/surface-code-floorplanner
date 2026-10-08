"""Local-mode command, cache separation and pinned native counter checks."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lattice_floorplan.compiler import Compiler, CompilerConfig
from lattice_floorplan.geometry import GeometryOracle, GridState

QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\nh q[0];\ncx q[0],q[4];\nt q[2];\n'
GRID = "MrMrMrMrM\nrrrrrrrrr\nMrQrQrQrM\nrrrrrrrrr\nMrQrQrrrM\nrrrrrrrrr\nMrrrrrrrM\nrrrrrrrrr\nMrMrMrMrM\n"


class LocalCompilationTests(unittest.TestCase):
    def test_cache_separates_local_and_nonlocal(self):
        def fake_run(self, qasm, layout, directory, *args, **kwargs):
            Path(directory).mkdir()
            return {"status": "success", "slices": 14 if self.config.local else 13}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_text("stub")
            with patch.object(Compiler, "run", fake_run):
                a = GeometryOracle(Compiler(binary), root / "cache")
                b = GeometryOracle(Compiler(binary, CompilerConfig(local=True)), root / "cache")
                x = a.evaluate(QASM, GridState.parse(GRID))
                y = b.evaluate(QASM, GridState.parse(GRID))
                self.assertNotEqual(x["cache_key"], y["cache_key"])
                self.assertEqual((x["slices"], y["slices"]), (13, 14))
                b.evaluate(QASM, GridState.parse(GRID))
                self.assertEqual(b.accounting()["cache_hits"], 1)

    @unittest.skipUnless(os.environ.get("LSQECC_SLICER"), "Requires native compiler")
    def test_local_json_and_summary_counter_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            compiler = Compiler(os.environ["LSQECC_SLICER"], CompilerConfig(local=True))
            counts = []
            for retain in (False, True):
                result = compiler.run(
                    QASM, "edpc", Path(directory) / str(retain), "qasm", retain_slices=retain
                )
                self.assertEqual(result["status"], "success", result)
                self.assertIn("--local", result["command"])
                self.assertTrue(result["config"]["local"])
                self.assertEqual(result["reported_slices"], result["slices"] + 1)
                counts.append(result["slices"])
            self.assertEqual(counts, [14, 14])
