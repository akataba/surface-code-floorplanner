#!/usr/bin/env python3
"""Re-evaluate corrected layouts, then optimize with explicit local compilation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from lattice_floorplan.compiler import Compiler, CompilerConfig
from lattice_floorplan.geometry import GridState, GeometryOracle, write_layout_output
from lattice_floorplan.layout import Layout
from lattice_floorplan.supply import initialize_padded_supply, supply_count
from lattice_floorplan.tempering import TemperingConfig, search

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"line", "star", "matching", "dense", "QFT", "Haar", "QSP", "QSVT"}


def main():
    """Run the frozen all-eight-family local study using explicit native tool paths.

    Reads corrected circuits and preserved warm-start layout/identity inputs.
    Writes protocol, source snapshots, native comparisons and seeded searches
    to --out. Each count/seed budget includes two initialization slots; separate
    prior evaluations, native baselines and winner replays are recorded outside
    that budget. Uses wave plus local compilation and fixed replenishment 10.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.budget < 15:
        parser.error("Budget must cover two initialization slots and search calibration")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        parser.error("Use a new output directory")
    cfg = CompilerConfig(local=True, timeout=30)
    algorithm = TemperingConfig(move_supply=True)
    native = Compiler(args.native, cfg)
    adapter = Compiler(args.adapter, cfg)
    manifest = json.loads((ROOT / "examples/corrected-v3/manifest.json").read_text())["entries"]
    assert len(manifest) == 20 and {t["family"] for t in manifest} == FAMILIES
    previous_root = ROOT / "examples/benchmark-starts"
    previous = {}
    for entry in json.loads((previous_root / "manifest.json").read_text())["entries"]:
        previous.setdefault(entry["circuit"], []).append(entry)
    counts = [1, 2, 4, 8, 16, 24, 32]
    seeds = [61, 62]
    grids = {}
    for mode in ["compact", "compact_no_clogging", "edpc"]:
        text = subprocess.check_output([str(args.exporter.resolve()), mode, "5"], text=True)
        grids[mode] = GridState.parse(text)
        write_layout_output(
            out / "native-grids" / mode, grids[mode], "OPENQASM 2.0;\nqreg q[5];\n", mode
        )
    protocol = {
        "compiler": asdict(cfg),
        "algorithm": asdict(algorithm),
        "counts": counts,
        "seeds": seeds,
        "charged_slots_per_count_seed": args.budget,
        "evaluation": manifest,
        "native_sha256": native.binary_sha256,
        "adapter_sha256": adapter.binary_sha256,
        "scope": "wave --local; NOT full EDPC scheduling; abstract M cells, latency10",
        "objective": "area_times_slices",
        "area_cap": None,
        "initialization": "two charged slots: prior corrected nonlocal count winner and count-adjusted native EDPC; padded construction for counts above16; select successful lower local volume",
        "prior_cost": "warm starts inherit earlier nonlocal search cost; not an equal-total-budget comparison",
        "costs": "140 prior local evaluations +120 native/export evaluations separate from280 searches up to32 slots; each successful count winner freshly replayed",
        "hpo": "none; fixed parameters, supply counts are design variables",
        "source_hashes": {},
    }
    for rel in [
        "scripts/benchmark_local.py",
        "src/lattice_floorplan/compiler.py",
        "src/lattice_floorplan/geometry.py",
        "src/lattice_floorplan/tempering.py",
        "src/lattice_floorplan/supply.py",
    ]:
        source = ROOT / rel
        protocol["source_hashes"][rel] = hashlib.sha256(source.read_bytes()).hexdigest()
        dest = out / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    oracle = GeometryOracle(adapter, out / "search-evidence")

    def run_circuit(task):
        """Compile native references, evaluate prior grids and search all supply counts for one
        task.

        Args:
            task (Mapping[str, Any]): Corrected corpus manifest entry with name, family and
                input paths.

        Returns:
            dict[str, Any]: Circuit-native comparisons, prior local evaluations, seeded searches
                and per-count winner evidence.

        Raises:
            RuntimeError: The input violates the constraints described above.
        """
        qasm = (ROOT / task["path"]).read_text()
        assert hashlib.sha256(qasm.encode()).hexdigest() == task["sha256"]
        task = {**task, "qasm": qasm}
        folder = out / "circuits" / task["name"]
        folder.mkdir(parents=True)
        baselines = []
        for mode, grid in grids.items():
            stock = native.run(
                qasm, mode, folder / "native" / mode / "stock", "qasm", retain_slices=False
            )
            exported = adapter.run(
                qasm,
                Layout.parse(grid.text),
                folder / "native" / mode / "exported",
                "qasm",
                retain_slices=False,
                preprocess_layout=mode,
            )
            equivalent = all(stock[k] == exported[k] for k in ["status", "slices", "instructions"])
            if not equivalent:
                raise RuntimeError(f"Native/export mismatch: {task['name']} {mode}")
            baselines.append(
                {
                    "mode": mode,
                    "area": grid.area,
                    "resources": Layout.parse(grid.text).resources(),
                    "native": stock,
                    "exported": exported,
                    "equivalent": equivalent,
                }
            )
        prior = []
        prior_grids = {}
        for point in previous[task["name"]]:
            n = point["count"]
            bundle = previous_root / point["bundle"]
            grid = GridState.parse(
                (bundle / "layout.txt").read_text(),
                json.loads((bundle / "identity.json").read_text())["logical_qubit_positions"],
            )
            prior_grids[n] = grid
            rec = oracle.evaluate(qasm, grid)
            prior.append(
                {
                    "count": n,
                    "area": grid.area,
                    "local": rec,
                }
            )
        runs = []
        for n in counts:
            for seed in seeds:
                starts = [
                    ("prior_nonlocal", prior_grids[n]),
                    ("native_count_adjusted", initialize_padded_supply(grids["edpc"], n, seed)),
                ]
                inits, valid = [], []
                for label, grid in starts:
                    rec = (
                        oracle.evaluate(qasm, grid)
                        if grid
                        else {"status": "construction_failed", "slices": None}
                    )
                    inits.append(
                        {
                            "source": label,
                            "status": rec["status"],
                            "slices": rec["slices"],
                            "area": grid.area if grid else None,
                            "cache_key": rec.get("cache_key"),
                        }
                    )
                    if rec["status"] == "success":
                        valid.append((grid.area * rec["slices"], label, grid))
                row = {
                    "count": n,
                    "seed": seed,
                    "initializations": inits,
                    "queries": 2,
                    "status": "no_successful_initialization",
                }
                if valid:
                    _, label, grid = min(valid, key=lambda x: (x[0], x[1]))
                    result = search(task, oracle, grid, "sa", args.budget - 2, seed, algorithm)
                    winner = GridState.parse(result["best_grid"], result["logical_qubit_positions"])
                    assert supply_count(winner) == n
                    dest = folder / f"count-{n}-seed-{seed}"
                    write_layout_output(dest, winner, qasm, task["name"])
                    (dest / "search.json").write_text(json.dumps(result, indent=2))
                    row.update(
                        status="success",
                        queries=args.budget,
                        chosen_initialization=label,
                        area=winner.area,
                        slices=result["best_slices"],
                        volume=result["best_objective"],
                        output=str(dest.relative_to(out)),
                        failed_candidates=result["failed_candidates"],
                    )
                runs.append(row)
                (folder / "runs.json").write_text(json.dumps(runs, indent=2))
                print(task["name"], n, seed, row["status"], row.get("volume"), flush=True)
        by_count = []
        for n in counts:
            choices = [r for r in runs if r["count"] == n and r["status"] == "success"]
            if not choices:
                by_count.append({"count": n, "status": "no_successful_initialization"})
                continue
            best = min(choices, key=lambda r: (r["volume"], r["slices"]))
            dest = out / best["output"]
            replay = adapter.run(
                (dest / "mapped-circuit.qasm").read_text(),
                Layout.read(dest / "layout.txt"),
                dest / "fresh-replay",
                "qasm",
                retain_slices=False,
                preprocess_layout="edpc",
            )
            assert replay["status"] == "success" and replay["slices"] == best["slices"]
            by_count.append({**best, "replay_verified": True})
        successes = [r for r in runs if r["status"] == "success"]
        row = {
            "name": task["name"],
            "family": task["family"],
            "native": baselines,
            "prior_re_evaluated": prior,
            "runs": runs,
            "by_count": by_count,
            "winner": min(successes, key=lambda r: r["volume"]) if successes else None,
        }
        (folder / "summary.json").write_text(json.dumps(row, indent=2))
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        circuits = list(pool.map(run_circuit, manifest))
    result = {
        "protocol": protocol,
        "circuits": circuits,
        "accounting": oracle.accounting(),
        "charged_search_slots": sum(r["queries"] for c in circuits for r in c["runs"]),
        "native_and_export_calls": 120,
        "winner_replay_calls": sum(
            r.get("replay_verified", False) for c in circuits for r in c["by_count"]
        ),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow

    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "validation/neural-pilot/mlflow.db"))
    mlflow.set_experiment("local-wave-corrected-layouts")
    with mlflow.start_run(run_name=out.name):
        mlflow.log_params(
            {
                "local": True,
                "pipeline": "wave",
                "latency": 10,
                "budget": args.budget,
                "seeds": str(seeds),
                "counts": str(counts),
                "objective": "area_times_slices",
            }
        )
        mlflow.log_metrics(
            {
                **oracle.accounting(),
                "charged_search_slots": result["charged_search_slots"],
                "winner_replay_calls": result["winner_replay_calls"],
            }
        )
        for f in ["protocol.json", "results.json"]:
            mlflow.log_artifact(str(out / f))
        (out / "mlflow.json").write_text(
            json.dumps(
                {
                    "run_id": mlflow.active_run().info.run_id,
                    "experiment_id": mlflow.active_run().info.experiment_id,
                },
                indent=2,
            )
        )
    print("FINISHED", out, flush=True)


if __name__ == "__main__":
    main()
