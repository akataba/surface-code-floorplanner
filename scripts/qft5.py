#!/usr/bin/env python3
"""Full five-qubit QFT -> Clifford+T; verify against the 32x32 Fourier matrix."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import mpmath as mp
import numpy as np
from pygridsynth.gridsynth import gridsynth_gates

M = {
    "h": np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
    "x": np.array([[0, 1], [1, 0]], complex),
    "z": np.diag([1, -1]),
    "s": np.diag([1, 1j]),
    "sdg": np.diag([1, -1j]),
    "t": np.diag([1, np.exp(1j * np.pi / 4)]),
    "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]),
}


def phase_error(actual, target):
    phase = np.vdot(target, actual)
    phase = phase / abs(phase)
    return float(np.linalg.norm(actual - phase * target, 2))


def unitary(gates, n):
    u = np.eye(2**n, dtype=complex)
    for gate in gates:
        name, *qubits = gate
        if name == "cx":
            control, target = qubits
            perm = np.arange(2**n)
            for k in range(2**n):
                if k & (1 << (n - 1 - control)):
                    perm[k] = k ^ (1 << (n - 1 - target))
            u = u[perm, :]
        else:
            bit = 1 << (n - 1 - qubits[0])
            a = np.array([i for i in range(2**n) if not i & bit])
            b = a + bit
            ua, ub = u[a, :].copy(), u[b, :].copy()
            mat = M[name]
            u[a, :] = mat[0, 0] * ua + mat[0, 1] * ub
            u[b, :] = mat[1, 0] * ua + mat[1, 1] * ub
    return u


def generate(epsilon="1e-6"):
    mp.mp.dps = 60
    eps = mp.mpf(epsilon)
    if not 0 < eps < mp.mpf("0.01"):
        raise ValueError("Use 0 < epsilon < 0.01")
    rotations = {4: ["t"]}
    synthesis = {}
    for denominator in (8, 16, 32):
        theta = mp.pi / denominator
        operators = gridsynth_gates(theta, eps, seed=0)
        if set(operators) - set("HTSXZW"):
            raise ValueError(f"Unsupported synthesis string {operators}")
        temporal = [x.lower() for x in reversed(operators) if x != "W"]
        target = np.diag([np.exp(-1j * float(theta) / 2), np.exp(1j * float(theta) / 2)])
        error = phase_error(unitary([(x, 0) for x in temporal], 1), target)
        if error > float(eps) * 1.01:
            raise ValueError(f"Rotation synthesis failed validation: {error}")
        rotations[denominator] = temporal
        synthesis[str(denominator)] = {
            "operator_string": operators,
            "temporal_sequence": temporal,
            "error_up_to_global_phase": error,
        }
    inverse = {"h": "h", "x": "x", "z": "z", "s": "sdg", "t": "tdg", "sdg": "s", "tdg": "t"}
    gates = []

    def rz(q, denominator, negative=False):
        seq = rotations[denominator]
        if negative:
            seq = [inverse[x] for x in reversed(seq)]
        gates.extend((x, q) for x in seq)

    for j in range(5):
        gates.append(("h", j))
        for k in range(j + 1, 5):
            d = 2 ** (k - j + 1)
            rz(k, d)
            rz(j, d)
            gates.append(("cx", k, j))
            rz(j, d, True)
            gates.append(("cx", k, j))
    for a, b in ((0, 4), (1, 3)):
        gates.extend([("cx", a, b), ("cx", b, a), ("cx", a, b)])
    indices = np.arange(32)
    fourier = np.exp(2j * np.pi * np.outer(indices, indices) / 32) / np.sqrt(32)
    error = phase_error(unitary(gates, 5), fourier)
    bound = 18 * float(eps)
    if error > bound + 1e-12:
        raise ValueError(f"Full QFT validation failed: {error} > {bound}")
    counts = dict(Counter(g[0] for g in gates))
    report = {
        "qubits": 5,
        "q0_is": "most significant bit",
        "fourier_sign": "+",
        "final_swaps": True,
        "dropped_controlled_rotations": 0,
        "controlled_phase_gates": 10,
        "epsilon_per_approximate_rz": epsilon,
        "approximate_rz_occurrences": 18,
        "triangle_error_bound": bound,
        "operator_norm_error_up_to_global_phase": error,
        "gate_counts": counts,
        "total_gates": len(gates),
        "t_count_including_tdg": counts.get("t", 0) + counts.get("tdg", 0),
        "synthesizer": "pygridsynth",
        "version": importlib.metadata.version("pygridsynth"),
        "seed": 0,
        "rotations": synthesis,
    }
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n'
        + "\n".join(g[0] + " " + ",".join(f"q[{q}]" for q in g[1:]) + ";" for g in gates)
        + "\n"
    )
    return gates, qasm, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", default="1e-6")
    parser.add_argument("--out", default="examples/qft5")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    gates, qasm, report = generate(args.epsilon)
    (out / "qft5_clifford_t.qasm").write_text(qasm)
    (out / "gates.json").write_text(json.dumps(gates) + "\n")
    (out / "synthesis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rotations"}, indent=2))


if __name__ == "__main__":
    main()
