#!/usr/bin/env python3
"""Explicit QSP sequences and LCU Chebyshev QSVT, with functional verification."""

import argparse
import json
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator
from haar import synthesize


def qsp_response(x, phases):
    w = np.array([[x, 1j * np.sqrt(1 - x * x)], [1j * np.sqrt(1 - x * x), x]])

    def phase(p):
        return np.diag([np.exp(1j * p), np.exp(-1j * p)])

    u = phase(phases[0])
    for p in phases[1:]:
        u = phase(p) @ w @ u
    return u


def qsp(degree):
    signals = [-0.7, -0.3, 0.1, 0.45, 0.8]
    phases = [float(0.27 * np.cos(j * np.pi / degree) + 0.13) for j in range(degree + 1)]
    qc = QuantumCircuit(5)
    for q, x in enumerate(signals):
        qc.rz(-2 * phases[0], q)
        for p in phases[1:]:
            qc.rx(-2 * np.arccos(x), q)
            qc.rz(-2 * p, q)
            qc.barrier(q)
    expected = np.array([[1]], complex)
    for x in reversed(signals):
        expected = np.kron(expected, qsp_response(x, phases))
    target = Operator(qc).data
    error = float(np.linalg.norm(target - expected, 2))
    if error > 1e-12:
        raise ValueError("QSP scalar-product verification failed")
    xs = np.linspace(-1, 1, 101)
    ys = np.array([qsp_response(x, phases)[0, 0] for x in xs])
    coefficients = np.polynomial.chebyshev.chebfit(xs, ys, degree)
    fit_error = float(np.max(abs(np.polynomial.chebyshev.chebval(xs, coefficients) - ys)))
    parity_error = float(
        max(abs(coefficients[j]) for j in range(degree + 1) if j % 2 != degree % 2)
    )
    if max(fit_error, parity_error) > 1e-10:
        raise ValueError("QSP polynomial verification failed")
    return (
        qc,
        target,
        {
            "family": "QSP five independent signal-processing chains",
            "degree": degree,
            "signals": signals,
            "phases": phases,
            "functional_error": error,
            "response_fit_error": fit_error,
            "parity_error": parity_error,
            "response_chebyshev_coefficients": [[z.real, z.imag] for z in coefficients],
            "scope": "Known phase-defined polynomial response; not an inversion or Hamiltonian-simulation algorithm. No two-qubit gates: routing control.",
        },
    )


def lcu_block():
    weights = [0.4, 0.3, 0.2, 0.1]
    words = ["ZII", "XXI", "IZX", "IIZ"]
    prep = StatePreparation(np.sqrt(weights))
    u = QuantumCircuit(5)
    u.append(prep, [0, 1])
    for index, word in enumerate(words):
        for a in (0, 1):
            if not (index >> a) & 1:
                u.x(a)
        for k, p in enumerate(word):
            if p == "X":
                u.ccx(0, 1, k + 2)
            elif p == "Z":
                u.h(k + 2)
                u.ccx(0, 1, k + 2)
                u.h(k + 2)
        for a in (0, 1):
            if not (index >> a) & 1:
                u.x(a)
    u.append(prep.inverse(), [0, 1])
    paulis = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Z": np.diag([1, -1])}
    a = np.zeros((8, 8), complex)
    for weight, word in zip(weights, words):
        term = np.array([[1]])
        for p in reversed(word):
            term = np.kron(term, paulis[p])
        a += weight * term
    return u, a, weights, words


def qsvt(degree):
    u, a, weights, words = lcu_block()
    um = Operator(u).data
    indices = np.arange(8) * 4
    block_error = float(np.linalg.norm(um[np.ix_(indices, indices)] - a, 2))
    if block_error > 1e-12:
        raise ValueError("LCU block encoding failed")
    qc = QuantumCircuit(5)
    for _ in range(degree):
        qc.compose(u, inplace=True)
        # R=2|00><00|-I on two low-order ancillas: diag(1,-1,-1,-1).
        qc.z(0)
        qc.z(1)
        qc.cz(0, 1)
        qc.barrier()
    target = Operator(qc).data
    eigenvalues, v = np.linalg.eigh(a)
    polynomial = (v * np.cos(degree * np.arccos(np.clip(eigenvalues, -1, 1)))) @ v.conj().T
    error = float(np.linalg.norm(target[np.ix_(indices, indices)] - polynomial, 2))
    if error > 1e-11:
        raise ValueError("Chebyshev QSVT verification failed")
    return (
        qc,
        target,
        {
            "family": "LCU Chebyshev QSVT / qubitized walk",
            "degree": degree,
            "ancillas": [0, 1],
            "system": [2, 3, 4],
            "weights": weights,
            "pauli_words_data_low_to_high": words,
            "eigenvalues": eigenvalues.tolist(),
            "block_encoding_error": block_error,
            "functional_error": error,
            "projected_indices": indices.tolist(),
            "projected_target_real": polynomial.real.tolist(),
            "scope": "Projected (R U)^d equals T_d(A); Hermitian involutory U, so U dagger = U. Phase pi/2 implements R up to global phase. Explicit PREPARE/SELECT retained.",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="examples/structured5")
    p.add_argument("--degrees", type=int, nargs="+", default=[3, 7])
    args = p.parse_args()
    for degree in args.degrees:
        if degree < 1:
            raise ValueError("Degree must be positive")
        for name, builder in [("qsp", qsp), ("qsvt", qsvt)]:
            qc, target, meta = builder(degree)
            report = synthesize(qc, target, Path(args.out) / f"{name}-d{degree}", 0, 1e-6, meta)
            print(name, degree, report["total_gates"], flush=True)


if __name__ == "__main__":
    main()
