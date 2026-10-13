"""Check emitted abstract LLI and explicitly detect missing execution semantics."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from lattice_floorplan.geometry import GridState, remap_qasm
from lattice_floorplan.compiler import Compiler, CompilerConfig
from lattice_floorplan.layout import Layout
from semantic_reference import abstract_lli_matrix, qasm_matrix, phase_error

ADAPTER = os.environ.get("LSQECC_ADAPTER")
QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\ncx q[0],q[2];\nx q[1];\nz q[2];\n'
)


class LliReferenceGuards(unittest.TestCase):
    def test_no_false_certificate_for_measurements_or_local_traces(self):
        for text in (
            "RequestMagicState 3 0",
            "MeasureSinglePatch 0 Z",
            "MultiBodyMeasure 0:Z,1:Z",
            "BellBasedCNOT 0 1 3 4 [BellPrepare (1,2),(2,2)]",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                abstract_lli_matrix(text, 3)


@unittest.skipUnless(ADAPTER, "Set LSQECC_ADAPTER for real emitted-LLI checks")
class NativeLliSemanticsTests(unittest.TestCase):
    def lower(self, text, grid, directory):
        directory.mkdir()
        qasm = directory / "input.qasm"
        qasm.write_text(text)
        layout = directory / "layout.txt"
        layout.write_text(grid.text)
        result = subprocess.run(
            [
                ADAPTER,
                "-i",
                str(qasm),
                "-I",
                "qasm",
                "-L",
                "edpc",
                "--geometry-file",
                str(layout),
                "-P",
                "wave",
                "--local",
                "--printlli",
                "before",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.stderr, result.stderr)
        return result.stdout

    def test_abstract_clifford_lowering_on_custom_grids(self):
        # --printlli before verifies logical lowering, not local measurement execution.
        # Also require actual local compilation to finish for each supplied geometry.
        for i, text in enumerate(
            (
                "rrrrrrr\nrQrQrQr\nrrrrrrr\nrrrrrrr\n",
                "rrrrrrr\nrrrrrrr\nrQrQrQr\nrrrrrrr\nrrrrrrr\n",
            )
        ):
            g = GridState.parse(text)
            g = GridState(g.rows, tuple(reversed(g.qubits)))
            mapped = remap_qasm(QASM, g)
            with tempfile.TemporaryDirectory() as d:
                lli = self.lower(mapped, g, Path(d) / "lower")
                self.assertLess(
                    phase_error(abstract_lli_matrix(lli, 3), qasm_matrix(mapped)), 1e-12
                )
                record = Compiler(ADAPTER, CompilerConfig(local=True)).run(
                    mapped, Layout.parse(g.text), Path(d) / "compile", "qasm"
                )
                self.assertEqual(record["status"], "success", record)

    def test_pinned_t_sign_information_loss_is_not_certified(self):
        # This is a known-limit regression, NOT a passing T-equivalence test.
        g = GridState.parse("MrMrM\nrrrrr\nrrQrr\nrrrrr\nMrMrM\n")
        prefix = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
        with tempfile.TemporaryDirectory() as d:
            positive = self.lower(prefix + "t q[0];\n", g, Path(d) / "t")
            negative = self.lower(prefix + "tdg q[0];\n", g, Path(d) / "tdg")
        self.assertEqual(positive, negative, "Pinned lowering changed: review semantic coverage")
        self.assertGreater(
            phase_error(qasm_matrix(prefix + "t q[0];"), qasm_matrix(prefix + "tdg q[0];")), 0.1
        )
        with self.assertRaises(ValueError):
            abstract_lli_matrix(positive, 1)
