#!/usr/bin/env python3
"""Sample full Haar U(2**n), synthesize Clifford+T, and verify dense matrices."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import mpmath as mp
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from pygridsynth.gridsynth import gridsynth_gates
from qft5 import phase_error, unitary
from lattice_floorplan.synthesis import exact_rz_sequence


def haar_unitary(n, seed):
    """Complex Ginibre QR with diagonal-phase correction (Mezzadri)."""
    if not 1 <= n <= 6:
        raise ValueError("Dense verification supports 1..6 qubits")
    rng = np.random.default_rng(seed)
    d = 2**n
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    diagonal = np.diag(r)
    return q * (diagonal / np.abs(diagonal))[None, :]


def generate(n, seed, out, epsilon=1e-6):
    target = haar_unitary(n, seed)
    qc = QuantumCircuit(n)
    qc.unitary(target, range(n))
    return synthesize(
        qc,
        target,
        out,
        seed,
        epsilon,
        {"ensemble": f"Haar U({2**n}) via complex Gaussian QR with diagonal phase correction"},
        optimization_level=1,
    )


def synthesize(qc, target, out, seed=0, epsilon=1e-6, metadata=None, optimization_level=0):
    """Synthesize an explicit circuit without replacing it by a dense unitary gate."""
    if not 0 < epsilon < 0.01:
        raise ValueError("Use 0 < epsilon < 0.01")
    n = qc.num_qubits
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "target.npy", target)
    exact = transpile(
        qc,
        basis_gates=["rz", "h", "s", "sdg", "x", "cx"],
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    decomposition_error = phase_error(Operator(exact).data, target)
    if decomposition_error > 1e-10:
        raise ValueError("Continuous decomposition failed")
    mp.mp.dps = 60
    gates = []
    rotations = []
    continuous = []
    exact_count = 0
    snap_bound = 0.0
    for instruction in exact.data:
        name = instruction.operation.name
        if name == "barrier":
            continue
        qs = [exact.find_bit(q).index for q in instruction.qubits]
        continuous.append([name, qs, [float(v) for v in instruction.operation.params]])
        if name == "rz":
            angle = float(instruction.operation.params[0])
            recognized = exact_rz_sequence(angle)
            if recognized is not None:
                seq, deviation = recognized
                operators = None
                exact_count += 1
                snap_bound += deviation / 2
            else:
                operators = gridsynth_gates(mp.mpf(angle), mp.mpf(epsilon), seed=seed)
                if set(operators) - set("HTSXZW"):
                    raise ValueError("Unexpected synthesis gate")
                seq = [g.lower() for g in reversed(operators) if g != "W"]
            rz = np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])
            error = phase_error(unitary([(g, 0) for g in seq], 1), rz)
            if error > epsilon * 1.01:
                raise ValueError("Rotation verification failed")
            rotations.append(
                {
                    "angle": angle,
                    "qubit": qs[0],
                    "operators": operators,
                    "temporal_sequence": list(seq),
                    "method": "exact_clifford_t" if recognized is not None else "gridsynth",
                    "error": error,
                }
            )
            gates.extend((g, qs[0]) for g in seq)
            if len(rotations) % 100 == 0:
                print(f"seed {seed}: synthesized {len(rotations)} rotations", flush=True)
        else:
            if name not in {"h", "s", "sdg", "x", "cx"}:
                raise ValueError(name)
            gates.append((name, *qs))
    # Our dense simulator uses q0 MSB; Qiskit uses q0 LSB.
    actual = unitary([(g[0], *(n - 1 - q for q in g[1:])) for g in gates], n)
    error = phase_error(actual, target)
    bound = (len(rotations) - exact_count) * epsilon + snap_bound + decomposition_error
    if error > bound + 1e-10:
        raise ValueError(f"Full unitary error {error} > {bound}")
    counts = dict(Counter(g[0] for g in gates))
    report = {
        "qubits": n,
        "seed": seed,
        "workload_metadata": metadata or {},
        "basis_order": "q0 least significant bit (Qiskit)",
        "epsilon_per_rz": epsilon,
        "approximate_rz_occurrences": len(rotations) - exact_count,
        "exact_rz_occurrences": exact_count,
        "exact_recognition_angle_tolerance": 1e-12,
        "snapping_operator_error_bound": snap_bound,
        "triangle_error_bound": bound,
        "operator_norm_error_up_to_global_phase": error,
        "continuous_decomposition_error": decomposition_error,
        "target_unitarity_error": float(np.linalg.norm(target.conj().T @ target - np.eye(2**n), 2)),
        "total_gates": len(gates),
        "gate_counts": counts,
        "t_count_including_tdg": counts.get("t", 0) + counts.get("tdg", 0),
        "versions": {
            k: importlib.metadata.version(k) for k in ("qiskit", "numpy", "pygridsynth", "mpmath")
        },
        "rotation_synthesis": rotations,
    }
    if metadata and "ensemble" in metadata:
        report["ensemble"] = metadata["ensemble"]
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        + f"qreg q[{n}];\n"
        + "\n".join(g[0] + " " + ",".join(f"q[{q}]" for q in g[1:]) + ";" for g in gates)
        + "\n"
    )
    (out / "circuit.qasm").write_text(qasm)
    (out / "gates.json").write_text(json.dumps(gates) + "\n")
    (out / "continuous.json").write_text(
        json.dumps({"global_phase": float(exact.global_phase), "gates": continuous}) + "\n"
    )
    (out / "synthesis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rotation_synthesis"}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qubits", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--out", default="examples/haar5")
    args = p.parse_args()
    for seed in args.seeds:
        generate(args.qubits, seed, Path(args.out) / f"seed-{seed}", args.epsilon)


if __name__ == "__main__":
    main()
