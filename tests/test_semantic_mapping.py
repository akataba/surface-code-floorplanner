"""Unitary equivalence across identity permutations and changed grid dimensions."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from lattice_floorplan.geometry import GridState, remap_qasm, write_layout_output
from lattice_floorplan.synthesis import exact_rz_sequence
from semantic_reference import matrix, qasm_matrix, phase_error

QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\ncx q[0],q[2];\nt q[1];\nsdg q[0];\ntdg q[2];\nx q[1];\nz q[0];\ns q[2];\n'


class SemanticMappingTests(unittest.TestCase):
    def test_full_unitary_all_assignments_and_two_geometries(self):
        expected = qasm_matrix(QASM)
        for text in ("rrrrrrr\nrQrQrQr\nrrrrrrr\n", "rrrrr\nrQrrr\nrrrQr\nrQrrr\nrrrrr\n"):
            base = GridState.parse(text)
            for positions in itertools.permutations(base.qubits):
                with self.subTest(shape=base.shape, positions=positions):
                    grid = GridState(base.rows, positions)
                    actual = qasm_matrix(remap_qasm(QASM, grid))
                    perm = [
                        sum(((i >> q) & 1) << grid.logical_to_site[q] for q in range(3))
                        for i in range(8)
                    ]
                    self.assertLess(phase_error(actual[np.ix_(perm, perm)], expected), 1e-12)
                    with tempfile.TemporaryDirectory() as d:
                        write_layout_output(d, grid, QASM)
                        identity = json.loads((Path(d) / "identity.json").read_text())
                        loaded = GridState.parse(
                            (Path(d) / "layout.txt").read_text(),
                            identity["logical_qubit_positions"],
                        )
                        self.assertEqual(loaded, grid)
                        self.assertEqual(
                            (Path(d) / "mapped-circuit.qasm").read_text(), remap_qasm(QASM, loaded)
                        )

    def test_wrong_identity_and_reversed_cnot_are_detected(self):
        base = GridState.parse("rrrrrrr\nrQrQrQr\nrrrrrrr\n")
        changed = GridState(base.rows, tuple(reversed(base.qubits)))
        self.assertGreater(
            phase_error(qasm_matrix(remap_qasm(QASM, changed)), qasm_matrix(QASM)), 0.1
        )
        self.assertGreater(
            phase_error(
                qasm_matrix(QASM.replace("cx q[0],q[2]", "cx q[2],q[0]")), qasm_matrix(QASM)
            ),
            0.1,
        )

    def test_exact_rotation_semantics(self):
        for k in range(-32, 33):
            theta = k * np.pi / 4 + 1e-15
            seq, _ = exact_rz_sequence(theta)
            target = np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
            self.assertLess(phase_error(matrix([(g, 0) for g in seq], 1), target), 1e-12)
        for angle in (0.123, np.pi / 8, np.pi / 4 + 1e-8):
            self.assertIsNone(exact_rz_sequence(angle))

    def test_reference_rejects_unknown_gate(self):
        with self.assertRaises(ValueError):
            qasm_matrix(QASM + "reset q[0];")
