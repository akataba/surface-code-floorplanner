"""Check the mathematical workloads, not merely their gate-list shapes."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(m) for m in ("qiskit", "numpy", "pygridsynth"))


@unittest.skipUnless(AVAILABLE, "Install the haar optional dependencies")
class StructuredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import structured

        cls.s = structured

    def test_scalar_qsp_polynomial_and_parity(self):
        for d in (1, 2, 3, 7):
            qc, target, meta = self.s.qsp(d)
            self.assertLess(meta["response_fit_error"], 1e-12)
            self.assertLess(meta["parity_error"], 1e-12)
            self.assertNotIn("cx", qc.count_ops())

    def test_lcu_and_chebyshev_recurrence(self):
        import numpy as np
        from qiskit.quantum_info import Operator

        u, a, _, _ = self.s.lcu_block()
        um = Operator(u).data
        np.testing.assert_allclose(um, um.conj().T, atol=1e-12)
        np.testing.assert_allclose(um @ um, np.eye(32), atol=1e-12)
        old = np.eye(8)
        cur = a
        for d in range(1, 8):
            if d > 1:
                old, cur = cur, 2 * a @ cur - old
            _, target, meta = self.s.qsvt(d)
            idx = meta["projected_indices"]
            np.testing.assert_allclose(target[np.ix_(idx, idx)], cur, atol=1e-12)

    def test_saved_clifford_t_and_projected_block(self):
        import numpy as np
        import haar

        paths = sorted((ROOT / "examples/corrected-v3").glob("qs*/synthesis.json"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            report = json.loads(path.read_text())
            g = json.loads((path.parent / "gates.json").read_text())
            self.assertTrue(set(x[0] for x in g) <= {"h", "x", "z", "s", "sdg", "t", "tdg", "cx"})
            actual = haar.unitary([(x[0], *(4 - q for q in x[1:])) for x in g], 5)
            target = np.load(path.parent / "target.npy")
            error = haar.phase_error(actual, target)
            self.assertLess(error, report["triangle_error_bound"])
            self.assertAlmostEqual(
                error, report["operator_norm_error_up_to_global_phase"], places=10
            )
            if path.parent.name.startswith("qsvt"):
                phase = np.vdot(target, actual)
                actual *= np.exp(-1j * np.angle(phase))
                idx = report["workload_metadata"]["projected_indices"]
                expected = np.array(report["workload_metadata"]["projected_target_real"])
                self.assertLess(
                    np.linalg.norm(actual[np.ix_(idx, idx)] - expected, 2),
                    report["triangle_error_bound"],
                )
