import unittest, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lattice_floorplan.synthesis import exact_rz_sequence
import importlib.util
AVAILABLE = all(importlib.util.find_spec(m) for m in ("qiskit", "pygridsynth"))
if AVAILABLE:
    from qft5 import unitary, phase_error


@unittest.skipUnless(AVAILABLE, "Install the haar optional dependencies")
class ExactSynthesisTests(unittest.TestCase):
    def test_exact_rotations_positive_negative_and_noise(self):
        for k in range(-32, 33):
            a = k * math.pi / 4 + 1e-15
            seq, deviation = exact_rz_sequence(a)
            target = np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])
            self.assertLess(phase_error(unitary([(x, 0) for x in seq], 1), target), 1e-13)
            self.assertEqual(sum(x in ("t", "tdg") for x in seq), k % 2)

    def test_nonexact_angles_not_snapped(self):
        for a in [math.pi / 8, 0.123, math.pi / 4 + 1e-8, float("nan"), float("inf")]:
            self.assertIsNone(exact_rz_sequence(a))

    def test_future_synthesis_bypasses_approximation(self):
        import tempfile
        from unittest.mock import patch
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from haar import synthesize

        q = QuantumCircuit(1)
        q.rz(math.pi / 4, 0)
        q.rz(-math.pi / 4, 0)
        q.rz(math.pi / 2, 0)
        with (
            tempfile.TemporaryDirectory() as d,
            patch(
                "haar.gridsynth_gates", side_effect=AssertionError("Exact gate sent to gridsynth")
            ),
        ):
            report = synthesize(q, Operator(q).data, Path(d) / "out")
        self.assertEqual(report["t_count_including_tdg"], 2)
        self.assertEqual(report["approximate_rz_occurrences"], 0)


if __name__ == "__main__":
    unittest.main()
