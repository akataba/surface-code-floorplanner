"""Portable installed command for the current geometry-search API."""

import argparse
import json
from pathlib import Path

from .api import optimize_layout
from .compiler import CompilerConfig
from .geometry import GridState
from .tempering import TemperingConfig


def main(argv=None):
    """Run the geometry-search API from argv and print winner/accounting JSON.

    None reads process arguments. Requires a liblsqecc adapter and new output
    directory; --identity restores saved logical positions. Pass --local for
    current local compilation. Returns 0 on success; handled errors exit 2.

    Args:
        argv (Sequence[str] | None): Command-line arguments without the program name; None reads
            sys.argv[1:].

    Returns:
        int: Zero after successful search, export and replay.

    Raises:
        SystemExit: Argument parsing fails, help is requested, or a handled input/file error
            exits with status 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qasm", type=Path)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument(
        "--identity", type=Path, help="identity.json when resuming an exported grid"
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--method", choices=["sa", "restart_sa", "parallel_tempering"], default="sa"
    )
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--move-supply", action="store_true")
    parser.add_argument("--local", action="store_true", help="Enable pairwise local compilation")
    args = parser.parse_args(argv)
    try:
        identity = (
            json.loads(args.identity.read_text())["logical_qubit_positions"]
            if args.identity
            else None
        )
        grid = GridState.parse(args.layout.read_text(), identity)
        result = optimize_layout(
            args.qasm.read_text(),
            grid,
            adapter_path=args.adapter,
            output_directory=args.out,
            method=args.method,
            budget=args.budget,
            seed=args.seed,
            search_config=TemperingConfig(move_supply=args.move_supply),
            compiler_config=CompilerConfig(timeout=args.timeout, local=args.local),
        )
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("best_area", "best_slices", "best_objective", "accounting")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
