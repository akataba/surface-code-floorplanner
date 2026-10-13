"""Haar sampling and synthesis checks, including explicit endianness validation."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(m) for m in ("qiskit", "numpy", "pygridsynth"))


@unittest.skipUnless(AVAILABLE, "Install the haar optional dependencies")
class HaarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import haar

        cls.haar = haar

    def test_sampling_unitary_and_seed(self):
        import numpy as np

        a = self.haar.haar_unitary(5, 101)
        np.testing.assert_array_equal(a, self.haar.haar_unitary(5, 101))
        self.assertGreater(np.linalg.norm(a - self.haar.haar_unitary(5, 102)), 1)
        self.assertLess(np.linalg.norm(a.conj().T @ a - np.eye(32), 2), 1e-12)

    def test_haar_moment_smoke_check(self):
        # Statistical regression, not a proof of the sampling law.
        import numpy as np

        x = np.array([self.haar.haar_unitary(2, s)[0, 0] for s in range(2048)])
        self.assertLess(abs(x.mean()), 0.04)
        self.assertAlmostEqual(float(np.mean(abs(x) ** 2)), 1 / 4, delta=0.025)
        self.assertAlmostEqual(float(np.mean(abs(x) ** 4)), 2 / (4 * 5), delta=0.015)

    def test_two_qubit_synthesis_and_bit_order(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sample"
            report = self.haar.generate(2, 19, out, 1e-5)
            qc = QuantumCircuit.from_qasm_file(str(out / "circuit.qasm"))
            error = self.haar.phase_error(Operator(qc).data, np.load(out / "target.npy"))
            self.assertLess(error, report["triangle_error_bound"])
            self.assertAlmostEqual(
                error, report["operator_norm_error_up_to_global_phase"], places=10
            )
            self.assertTrue(set(qc.count_ops()) <= {"h", "x", "z", "s", "sdg", "t", "tdg", "cx"})

    def test_saved_five_qubit_samples(self):
        import numpy as np

        paths = sorted((ROOT / "examples/corrected-v3").glob("haar-*/synthesis.json"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            r = json.loads(path.read_text())
            g = json.loads((path.parent / "gates.json").read_text())
            actual = self.haar.unitary([(x[0], *(4 - q for q in x[1:])) for x in g], 5)
            error = self.haar.phase_error(actual, np.load(path.parent / "target.npy"))
            self.assertLess(error, r["triangle_error_bound"])
            self.assertAlmostEqual(error, r["operator_norm_error_up_to_global_phase"], places=10)
