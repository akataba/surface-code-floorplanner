"""Small independent dense reference model for tests, not a physical simulator.

Uses little-endian qubit indices. Unsupported syntax raises instead of being
silently treated as identity. Logical BellBasedCNOT is an ideal CNOT only; its
local measurement implementation and correction records are not certified here.
"""

import re
import numpy as np

GATES = {
    "h": np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
    "x": np.array([[0, 1], [1, 0]], complex),
    "z": np.diag([1, -1]).astype(complex),
    "s": np.diag([1, 1j]),
    "sdg": np.diag([1, -1j]),
    "t": np.diag([1, np.exp(1j * np.pi / 4)]),
    "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]),
}


def matrix(gates, n):
    """Compute a full unitary with q0 as the least significant bit."""
    out = np.eye(2**n, dtype=complex)
    indices = np.arange(2**n)
    for name, *qs in gates:
        if any(q < 0 or q >= n for q in qs):
            raise ValueError("Out-of-range qubit")
        if name == "cx" and len(qs) == 2 and qs[0] != qs[1]:
            perm = indices ^ (((indices >> qs[0]) & 1) << qs[1])
            out = out[perm]
        elif name in GATES and len(qs) == 1:
            lo = indices[(indices & (1 << qs[0])) == 0]
            hi = lo | (1 << qs[0])
            a, b = out[lo].copy(), out[hi].copy()
            g = GATES[name]
            out[lo], out[hi] = g[0, 0] * a + g[0, 1] * b, g[1, 0] * a + g[1, 1] * b
        else:
            raise ValueError(f"Unsupported gate: {name} {qs}")
    return out


def qasm_matrix(text):
    """Parse the explicit Clifford+T subset used in semantic fixtures."""
    n = None
    gates = []
    for part in re.sub(r"//[^\n]*", "", text).split(";"):
        part = part.strip()
        if not part or part == "OPENQASM 2.0" or part == 'include "qelib1.inc"':
            continue
        m = re.fullmatch(r"qreg q\[(\d+)\]", part)
        if m:
            if n is not None:
                raise ValueError("Multiple registers")
            n = int(m[1])
            continue
        m = re.fullmatch(r"(h|x|z|s|sdg|t|tdg) q\[(\d+)\]", part)
        c = re.fullmatch(r"cx q\[(\d+)\],\s*q\[(\d+)\]", part)
        if m:
            gates.append((m[1], int(m[2])))
        elif c:
            gates.append(("cx", int(c[1]), int(c[2])))
        else:
            raise ValueError(f"Unsupported QASM: {part}")
    if n is None:
        raise ValueError("Missing register")
    return matrix(gates, n)


def abstract_lli_matrix(text, n):
    """Interpret only ideal unitary LLI; reject measurements/resources/local traces."""
    gates = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "RotateSingleCellPatch" and len(parts) == 2:
            # Patch boundary reorientation is not a logical gate.
            if not 0 <= int(parts[1]) < n:
                raise ValueError("Bad patch ID")
            continue
        if parts[0] in ("HGate", "XGate", "ZGate") and len(parts) == 2:
            gates.append((parts[0][0].lower(), int(parts[1])))
        elif parts[0] == "BellBasedCNOT" and len(parts) == 5:
            gates.append(("cx", int(parts[1]), int(parts[2])))
        else:
            raise ValueError(f"Cannot certify LLI semantics: {line}")
    return matrix(gates, n)


def phase_error(actual, expected):
    """Operator norm discrepancy modulo a single global phase."""
    phase = np.vdot(expected, actual)
    if abs(phase) == 0:
        return float(np.linalg.norm(actual - expected, 2))
    return float(np.linalg.norm(actual - expected * phase / abs(phase), 2))
