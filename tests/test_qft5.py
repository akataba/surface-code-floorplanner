"""Numerical checks for the saved five-qubit Clifford+T workload."""

import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("pygridsynth") is not None
)


@unittest.skipUnless(AVAILABLE, "Install the qft optional dependencies")
class QFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("qft5", ROOT / "scripts/qft5.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.gates = json.loads((ROOT / "examples/corrected-v3/qft5/gates.json").read_text())

    def test_full_fourier_matrix_error(self):
        import numpy as np

        indices = np.arange(32)
        target = np.exp(2j * np.pi * np.outer(indices, indices) / 32) / np.sqrt(32)
        actual = self.module.unitary(self.gates, 5)
        self.assertLess(self.module.phase_error(actual, target), 18e-6)
        self.assertLess(np.linalg.norm(actual.conj().T @ actual - np.eye(32), 2), 1e-11)

    def test_gate_set_and_output_swaps(self):
        self.assertTrue(
            all(g[0] in {"h", "s", "sdg", "t", "tdg", "x", "z", "cx"} for g in self.gates)
        )
        self.assertEqual(
            self.gates[-6:],
            [["cx", 0, 4], ["cx", 4, 0], ["cx", 0, 4], ["cx", 1, 3], ["cx", 3, 1], ["cx", 1, 3]],
        )
        self.assertEqual(sum(g[0] in ("t", "tdg") for g in self.gates), 1134)

    def test_wire_permutation_preserves_logical_unitary(self):
        import numpy as np

        mapping = [3, 0, 4, 1, 2]
        changed = [[g[0], *[mapping[q] for q in g[1:]]] for g in self.gates]
        permutation = []
        for x in range(32):
            y = 0
            for q in range(5):
                if x & (1 << (4 - q)):
                    y |= 1 << (4 - mapping[q])
            permutation.append(y)
        original = self.module.unitary(self.gates, 5)
        mapped = self.module.unitary(changed, 5)
        self.assertLess(
            np.linalg.norm(mapped[np.ix_(permutation, permutation)] - original, 2), 1e-12
        )
