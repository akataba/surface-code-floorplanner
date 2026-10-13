"""Independent ideal measurement gadgets, NOT validation of emitted local traces."""

import itertools
import unittest
import numpy as np
from semantic_reference import matrix, GATES


class MeasurementSemanticsTests(unittest.TestCase):
    def test_cnot_all_eight_branches_and_missing_correction_mutant(self):
        # Bit 0 ancilla, bit 1 control, bit 2 target. Logical input order c,t.
        embed = np.zeros((8, 4), complex)
        for i in range(4):
            embed[2 * i : 2 * i + 2, i] = 1 / np.sqrt(2)
        zz = matrix([("z", 0), ("z", 1)], 3)
        xx = matrix([("x", 0), ("x", 2)], 3)
        ideal = matrix([("cx", 0, 1)], 2)
        total = np.zeros((4, 4), complex)
        incorrect = 0
        for a, b, c in itertools.product((0, 1), repeat=3):
            pzz = (np.eye(8) + (-1) ** a * zz) / 2
            pxx = (np.eye(8) + (-1) ** b * xx) / 2
            bra = np.zeros((4, 8), complex)
            for i in range(4):
                bra[i, 2 * i + c] = 1  # Final ancilla Z measurement.
            raw = bra @ pxx @ pzz @ embed
            correction = matrix([("z", 0)] * b + [("x", 1)] * (a ^ c), 2)
            branch = correction @ raw
            phase = np.vdot(ideal, branch) / 4
            np.testing.assert_allclose(branch, phase * ideal, atol=1e-12)
            self.assertAlmostEqual(abs(phase) ** 2, 1 / 8)
            total += branch.conj().T @ branch
            incorrect += not np.allclose(raw, np.vdot(ideal, raw) / 4 * ideal)
        np.testing.assert_allclose(total, np.eye(4), atol=1e-12)
        self.assertGreater(incorrect, 0)

    def test_t_and_t_dagger_both_outcomes_with_conditional_s(self):
        for sign, gate, correction in [(1, "t", "s"), (-1, "tdg", "sdg")]:
            embed = np.zeros((4, 2), complex)
            for i in range(2):
                embed[i, i] = 1 / np.sqrt(2)
                embed[i + 2, i] = np.exp(sign * 1j * np.pi / 4) / np.sqrt(2)
            entangled = matrix([("cx", 0, 1)], 2) @ embed
            total = np.zeros((2, 2), complex)
            for outcome in (0, 1):
                branch = entangled[outcome * 2 : outcome * 2 + 2]
                if outcome:
                    branch = GATES[correction] @ branch
                phase = np.vdot(GATES[gate], branch) / 2
                np.testing.assert_allclose(branch, phase * GATES[gate], atol=1e-12)
                total += branch.conj().T @ branch
            np.testing.assert_allclose(total, np.eye(2), atol=1e-12)
