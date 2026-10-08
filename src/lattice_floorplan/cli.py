"""Legacy inspection, native benchmarking and LLI mapping command-line tools."""

import argparse
import csv
import json
import os
from pathlib import Path
from .compiler import Compiler, CompilerConfig
from .layout import Layout
from .search import optimize, optimize_grid


def binary_path(value):
    """Resolve explicit path, LSQECC_SLICER, local config or bundled path, in that order.

    Args:
        value (str | os.PathLike[str] | None): Explicit binary path or None to consult
            environment/configuration fallbacks.

    Returns:
        str | os.PathLike[str]: Selected executable path; existence is not checked for
            explicit/configured paths.

    Raises:
        ValueError: Pass --binary, set LSQECC_SLICER, or run scripts/bootstrap.py.
    """
    if value:
        return value
    if os.environ.get("LSQECC_SLICER"):
        return os.environ["LSQECC_SLICER"]
    config = Path(".local.json")
    if config.exists():
        return json.loads(config.read_text())["binary"]
    candidate = Path(".deps/liblsqecc/build/lsqecc_slicer")
    if candidate.exists():
        return candidate
    raise ValueError("Pass --binary, set LSQECC_SLICER, or run scripts/bootstrap.py")


def discover(root):
    """Find parseable ASCII .txt layouts recursively and return paths with resources.

    Args:
        root (str | os.PathLike[str]): Filesystem root containing the inputs or evidence
            described above.

    Returns:
        list[dict[str, Any]]: Parseable layout paths and resource summaries.
    """
    found = []
    for path in sorted(Path(root).rglob("*.txt")):
        try:
            layout = Layout.read(path)
            found.append({"path": str(path), "resources": layout.resources()})
        except (ValueError, UnicodeError):
            continue
    return found


def main(argv=None):
    """Dispatch inspection, compilation or historical LLI search from argv.

    None reads process arguments. Returns 0 on success, 1 for failed benchmark
    records; argument/file errors exit with code 2. Use --local explicitly for
    current evaluations. The optimize command is fixed-site mapping search;
    lattice-floorplan-optimize is the separate variable-geometry interface.

    Args:
        argv (Sequence[str] | None): Command-line arguments without the program name; None reads
            sys.argv[1:].

    Returns:
        int: Zero on success or one if a benchmark contains failed compilations.

    Raises:
        ValueError: Choose --layout, --layout-dir, or --builtin.
        SystemExit: Argument parsing fails, help is requested, or a handled input/file error
            exits with status 2.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate lattice-surgery layouts with real liblsqecc compilations"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("layouts", help="Discover valid ASCII layout files")
    p.add_argument("directory", nargs="?", default="examples/upstream")
    p = sub.add_parser("inspect", help="Validate a layout and optionally render it")
    p.add_argument("layout")
    p.add_argument("--svg")
    for name in ("benchmark", "optimize", "grid-search"):
        p = sub.add_parser(name)
        p.add_argument("circuit")
        p.add_argument("--binary")
        p.add_argument("--out", required=True)
        p.add_argument("--pipeline", choices=["wave", "stream"], default="wave")
        p.add_argument(
            "--router", choices=["graph_search", "graph_search_cached"], default="graph_search"
        )
        p.add_argument("--graph-search", choices=["djikstra", "astar"], default="djikstra")
        p.add_argument("--distillation-time", type=int, default=10)
        p.add_argument("--no-stagger", action="store_true")
        p.add_argument("--local", action="store_true")
        p.add_argument("--timeout", type=float, default=15)
        if name == "benchmark":
            p.add_argument("--layout", action="append", default=[])
            p.add_argument("--layout-dir")
            p.add_argument(
                "--builtin",
                action="append",
                choices=["compact", "compact_no_clogging", "edpc"],
                default=[],
            )
            p.add_argument("--format", choices=["lli", "qasm"], default="lli")
        else:
            p.add_argument("--layout", required=True)
            p.add_argument("--budget", type=int, default=32)
            p.add_argument("--seed", type=int, default=0)
            if name == "optimize":
                p.add_argument(
                    "--cooling", choices=["linear", "exponential", "logarithmic"], default="linear"
                )
                p.add_argument(
                    "--method", choices=["random", "anneal", "reinforce"], default="anneal"
                )
    args = parser.parse_args(argv)
    try:
        if args.command == "layouts":
            print(json.dumps(discover(args.directory), indent=2))
            return 0
        if args.command == "inspect":
            layout = Layout.read(args.layout)
            if args.svg:
                Path(args.svg).write_text(layout.svg(Path(args.layout).stem))
            print(json.dumps(layout.resources(), indent=2))
            return 0
        config = CompilerConfig(
            args.pipeline,
            args.router,
            args.graph_search,
            args.distillation_time,
            not args.no_stagger,
            args.timeout,
            local=args.local,
        )
        compiler = Compiler(binary_path(args.binary), config)
        text = Path(args.circuit).read_text()
        if args.command in ("optimize", "grid-search"):
            layout = Layout.read(args.layout)
            if args.command == "optimize":
                result = optimize(
                    compiler,
                    text,
                    layout,
                    args.out,
                    args.method,
                    args.budget,
                    args.seed,
                    args.cooling,
                )
            else:
                result = optimize_grid(compiler, text, layout, args.out, args.budget, args.seed)
            print(json.dumps(result, indent=2))
            return 0
        paths = args.layout + (
            [r["path"] for r in discover(args.layout_dir)] if args.layout_dir else []
        )
        candidates = [(p, Layout.read(p)) for p in dict.fromkeys(paths)] + [
            (b, b) for b in dict.fromkeys(args.builtin)
        ]
        if not candidates:
            raise ValueError("Choose --layout, --layout-dir, or --builtin")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        results = []
        for i, (name, layout) in enumerate(candidates):
            result = compiler.run(text, layout, out / f"run-{i:03d}", args.format)
            result["layout_name"] = name
            result["comparison"] = (
                "descriptive; resources may differ; generated layouts also change preprocessing"
            )
            results.append(result)
            print(f"{name}: {result['status']}, slices={result['slices']}")
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        with (out / "results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "layout",
                    "status",
                    "slices",
                    "instructions",
                    "area",
                    "wall_seconds",
                    "reason",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "layout": r["layout_name"],
                        "status": r["status"],
                        "slices": r["slices"],
                        "instructions": r["instructions"],
                        "area": (r["resources"] or {}).get("area"),
                        "wall_seconds": r["wall_seconds"],
                        "reason": r["reason"],
                    }
                )
        return 0 if all(r["status"] == "success" for r in results) else 1
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
