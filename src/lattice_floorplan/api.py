"""Supported orchestration API for compiler-scored, variable-size grid search.

Algorithms live in ``tempering``; compiler invocation lives in ``compiler``.
This module validates the integration boundary and writes a replayable bundle.
It does not synthesize circuits or perform experiment tracking implicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .compiler import Compiler, CompilerConfig, PIN
from .geometry import GeometryOracle, GridState, remap_qasm, valid_geometry, write_layout_output
from .tempering import TemperingConfig, search

SearchMethod = Literal["sa", "restart_sa", "parallel_tempering"]


def optimize_layout(
    qasm: str,
    initial_grid: GridState,
    *,
    adapter_path: str | Path,
    output_directory: str | Path,
    method: SearchMethod = "sa",
    budget: int = 256,
    seed: int = 61,
    search_config: TemperingConfig | None = None,
    compiler_config: CompilerConfig | None = None,
) -> dict[str, Any]:
    """Optimize an already synthesized, single-register OpenQASM 2 circuit.

    ``adapter_path`` must be the geometry adapter built against the pinned liblsqecc,
    not the stock slicer. ``initial_grid.qubits[i]`` is logical qubit i's cell.
    ``budget`` includes the baseline, calibration, failed and null proposals.
    Supply count stays fixed; enable ``search_config.move_supply`` to relocate M.

    The output directory must not exist. Success writes a complete grid/QASM/
    identity bundle, all observed Pareto points, compiler evidence and result.json.
    Pass CompilerConfig(local=True) explicitly for current local evaluation;
    omitting compiler_config retains the legacy nonlocal default. Area is tiles
    and the default objective is tile count times compiled slices, not physical
    qubit-cycles. Cache hits consume search slots but make no new native call.

    Returns a dictionary containing best_grid, logical_qubit_positions,
    best_area, best_slices, best_objective, history, pareto, accounting and replay.
    A fresh replay must match the winning score before this function returns.
    Replay is one additional native call outside the search budget. On failure,
    partial evidence is retained and ValueError/RuntimeError/OSError is raised.

    Args:
        qasm (str): Already synthesized, single-register OpenQASM 2 source matching the grid
            data-qubit count.
        initial_grid (GridState): Initial geometry with qubits[i] giving logical qubit i’s
            zero-based cell.
        adapter_path (str | pathlib.Path): liblsqecc adapter executable built against the pinned
            liblsqecc source.
        output_directory (str | pathlib.Path): New destination directory; must not already
            exist.
        method (str): Search algorithm: 'sa', 'restart_sa' or 'parallel_tempering'.
        budget (int): Charged evaluation slots, including the baseline and repeated requests;
            not necessarily native compiler calls.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.
        search_config (TemperingConfig | None): Search settings; None selects TemperingConfig
            defaults.
        compiler_config (CompilerConfig | None): Backend settings; pass local=True explicitly.
            None retains legacy local=False.

    Returns:
        dict[str, Any]: Search result with best_grid, logical_qubit_positions, best_area,
            best_slices, best_objective, history, pareto, accounting and verified replay evidence.

    Raises:
        ValueError: Configuration, initial geometry, QASM register or starting compilation is
            invalid.
        OSError: Binary/evidence files cannot be accessed or output_directory already exists.
        RuntimeError: Fresh export replay fails or differs from the winning slice count.
    """
    config = search_config or TemperingConfig()
    backend = compiler_config or CompilerConfig()
    if backend.pipeline != "wave":
        raise ValueError("Geometry optimization requires the pinned wave pipeline")
    if method not in ("sa", "restart_sa", "parallel_tempering"):
        raise ValueError(f"Unsupported search method: {method}")
    if budget < 1 + config.calibration_queries + config.replicas:
        raise ValueError("Budget must cover baseline, calibration and search replicas")
    if not valid_geometry(initial_grid):
        raise ValueError("Initial grid requires interior Q patches and connected routing")
    remap_qasm(qasm, initial_grid)  # Validate register/identity agreement before writing.
    compiler = Compiler(adapter_path, backend)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    oracle = GeometryOracle(compiler, output / "native-evidence")
    result = search(
        {"name": "user-circuit", "family": "user", "qasm": qasm},
        oracle,
        initial_grid,
        method,
        budget,
        seed,
        config,
    )
    winner = GridState.parse(result["best_grid"], result["logical_qubit_positions"])
    write_layout_output(output, winner, qasm, method)
    for index, point in enumerate(result["pareto"]):
        candidate = GridState.parse(point["grid"], point["logical_qubit_positions"])
        write_layout_output(output / "pareto" / str(index), candidate, qasm)
    # A new oracle directory bypasses the search cache, checking the exported pair.
    from .layout import Layout

    replay = compiler.run(
        (output / "mapped-circuit.qasm").read_text(),
        Layout.read(output / "layout.txt"),
        output / "replay",
        "qasm",
        retain_slices=False,
        preprocess_layout="edpc",
    )
    result.update(
        schema_version=1,
        native_pin=PIN,
        accounting=oracle.accounting(),
        replay=replay,
        replay_native_calls=1,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if replay["status"] != "success" or replay["slices"] != result["best_slices"]:
        raise RuntimeError(f"Export replay failed or disagreed; inspect {output / 'replay'}")
    return result
