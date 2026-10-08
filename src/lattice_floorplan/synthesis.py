"""Recognize numerically exact Clifford+T Z rotations before approximation."""

import math

# Temporal gate order, equivalent to Rz(k*pi/4) up to global phase.
EXACT_RZ = {
    0: (),
    1: ("t",),
    2: ("s",),
    3: ("s", "t"),
    4: ("z",),
    5: ("z", "t"),
    6: ("sdg",),
    7: ("tdg",),
}


def exact_rz_sequence(angle, tolerance=1e-12):
    """Return (sequence, angle error) or None. Snapping error must be budgeted.

    This is floating-point recognition, not a claim of symbolic equality. Reject
    large-angle loss of precision and keep arbitrary near-Clifford rotations outside
    this declared absolute tolerance on the approximation path.

    Args:
        angle (float): Z-rotation angle in radians; must be convertible to float.
        tolerance (float): Nonnegative absolute angular tolerance in radians; default 1e-12.

    Returns:
        tuple[tuple[str, ...], float] | None: Temporal Clifford+T gate names and absolute
            snapping error in radians, or None if recognition is rejected.

    Raises:
        TypeError: angle cannot be converted to a float.
        ValueError: A supplied angle string is not a valid float.
    """
    value = float(angle)
    if not math.isfinite(value) or abs(value) > 1e6:
        return None
    k = round(value / (math.pi / 4))
    error = abs(value - k * math.pi / 4)
    return (EXACT_RZ[k % 8], error) if error <= tolerance else None
