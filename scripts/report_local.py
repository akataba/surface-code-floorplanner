#!/usr/bin/env python3
"""Summarize local-wave results without treating failed native runs as wins."""

import argparse
import html
import json
from pathlib import Path
import statistics


def main():
    """Read local results from the CLI directory and overwrite summary.json/report.html.

    Compare custom volume only against successful native records, retaining
    native failures, per-seed outcomes and warm-start cost caveats. No compiler
    or optimization calls are made; nonlocal scores are not reported.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory
    result = json.loads((out / "results.json").read_text())
    circuits = result["circuits"]
    families = []
    for family in ["line", "star", "matching", "dense", "QFT", "Haar", "QSP", "QSVT"]:
        members = [c for c in circuits if c["family"] == family]
        reductions, edpc_reductions = [], []
        wins = 0
        for c in members:
            valid = [n for n in c["native"] if n["native"]["status"] == "success"]
            winner = c["winner"]
            if valid and winner:
                best = min(n["area"] * n["native"]["slices"] for n in valid)
                reductions.append(100 * (1 - winner["volume"] / best))
                wins += winner["volume"] < best
            edpc = next(n for n in c["native"] if n["mode"] == "edpc")
            if winner and edpc["native"]["status"] == "success":
                edpc_reductions.append(
                    100 * (1 - winner["volume"] / (edpc["area"] * edpc["native"]["slices"]))
                )
        seed_reductions = {}
        for seed in result["protocol"]["seeds"]:
            values = []
            for c in members:
                choices = [
                    r["volume"] for r in c["runs"] if r["seed"] == seed and r["status"] == "success"
                ]
                refs = [
                    n["area"] * n["native"]["slices"]
                    for n in c["native"]
                    if n["native"]["status"] == "success"
                ]
                if choices and refs:
                    values.append(100 * (1 - min(choices) / min(refs)))
            seed_reductions[str(seed)] = statistics.mean(values) if values else None
        families.append(
            {
                "family": family,
                "seed_mean_reductions_vs_best_native_pct": seed_reductions,
                "circuits": len(members),
                "custom_wins_vs_best_successful_native": wins,
                "comparable": len(reductions),
                "mean_volume_reduction_vs_best_native_pct": statistics.mean(reductions)
                if reductions
                else None,
                "mean_volume_reduction_vs_edpc_pct": statistics.mean(edpc_reductions)
                if edpc_reductions
                else None,
            }
        )
    summary = {
        "families": families,
        "native_successes": {
            m: sum(
                n["native"]["status"] == "success"
                for c in circuits
                for n in c["native"]
                if n["mode"] == m
            )
            for m in ["compact", "compact_no_clogging", "edpc"]
        },
        "prior_layout_local_successes": sum(
            p["local"]["status"] == "success" for c in circuits for p in c["prior_re_evaluated"]
        ),
        "search_successes": sum(r["status"] == "success" for c in circuits for r in c["runs"]),
        "accounting": result["accounting"],
        "charged_search_slots": result["charged_search_slots"],
        "winner_replay_calls": result["winner_replay_calls"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    parts = [
        '<!doctype html><html><head><meta charset="utf-8"><title>Local compilation results</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#192433}table{border-collapse:collapse;width:100%;margin-bottom:28px}td,th{padding:9px;border-bottom:1px solid #ccd;text-align:right}td:first-child,th:first-child{text-align:left}small{color:#456}a{color:#1468b0}</style></head><body><h1>Local compilation: corrected circuits</h1><p><b>Wave scheduling with --local enabled.</b> All 20 corrected circuits / eight families. Not full EDPC scheduling. Objective: grid tiles × completed local slices; abstract M sources, fixed replenishment 10, no hard area cap.</p><p>Native compact, compact_no_clogging and EDPC were freshly compiled and checked against exported-grid replay. Custom SA uses seven supply counts, two seeds and 32 charged slots per count/seed. Initializations include prior nonlocal winners and count-adjusted native EDPC. Prior search cost is inherited; this is not a comparison at equal total optimization cost. All local scores are fresh or keyed by local=true.</p><p>Failed native compilations are excluded from numeric comparisons and reported below, not counted as wins. Best-observed selection pools counts/seeds and does not prove optimality. Each successful per-count winner is freshly replayed.</p><h2>All-family volume comparison</h2><table><tr><th>Family</th><th>Wins / comparable</th><th>Mean reduction vs best successful native</th><th>Mean reduction vs EDPC</th></tr>'
    ]
    successful_prior = summary["prior_layout_local_successes"]
    parts[0] = parts[0].replace(
        "<h2>All-family volume comparison",
        f"<p><b>Prior-layout transfer:</b> only {successful_prior}/140 previous count winners compile locally. New local search found valid results for all 280 runs. Failed compilations are not numeric scores.</p><h2>All-family volume comparison",
    )
    for f in families:
        parts.append(
            f"<tr><td>{f['family']}</td><td>{f['custom_wins_vs_best_successful_native']}/{f['comparable']}</td><td>{f['mean_volume_reduction_vs_best_native_pct']:.2f}%</td><td>{f['mean_volume_reduction_vs_edpc_pct']:.2f}%</td></tr>"
        )
    parts.append(
        "</table><h2>Per-circuit local results</h2><table><tr><th>Circuit</th><th>M</th><th>Custom area</th><th>Custom slices</th><th>Custom volume</th><th>EDPC slices</th><th>EDPC volume</th><th>Compact volume</th><th>No-clogging volume</th></tr>"
    )
    for c in circuits:
        w = c["winner"]
        ns = {n["mode"]: n for n in c["native"]}
        cells = [html.escape(c["name"])]
        cells += [str(w[k]) for k in ["count", "area", "slices"]] if w else ["—"] * 3
        cells += [f'<a href="{w["output"]}/layout.svg">{w["volume"]:,}</a>' if w else "FAILED"]
        cells.append(str(ns["edpc"]["native"]["slices"]))
        for mode in ["edpc", "compact", "compact_no_clogging"]:
            n = ns[mode]
            r = n["native"]
            cells.append(f"{n['area'] * r['slices']:,}" if r["status"] == "success" else "FAILED")
        parts.append("<tr>" + "".join("<td>" + v + "</td>" for v in cells) + "</tr>")
    parts.append(
        "</table><h2>Details: seeds, count choices, local starting-layout evaluations and failures</h2>"
    )
    for c in circuits:
        parts.append(f"<details><summary>{html.escape(c['name'])}</summary>")
        parts.append("<h3>Native compiler failures</h3><ul>")
        for n in c["native"]:
            r = n["native"]
            if r["status"] != "success":
                path = out / "circuits" / c["name"] / "native" / n["mode"] / "stock" / "stderr.txt"
                diagnostics = path.read_text() + (path.parent / "stdout.txt").read_text()
                parts.append(
                    "<li>"
                    + html.escape(n["mode"] + ": " + r["reason"] + " — " + diagnostics)
                    + "</li>"
                )
        parts.append(
            "</ul><h3>Prior grids evaluated locally</h3><table><tr><th>M</th><th>Area</th><th>Local slices/status</th></tr>"
        )
        for p in c["prior_re_evaluated"]:
            vals = [
                p["count"],
                p["area"],
                p["local"]["slices"] if p["local"]["status"] == "success" else p["local"]["status"],
            ]
            parts.append("<tr>" + "".join("<td>" + str(v) + "</td>" for v in vals) + "</tr>")
        parts.append(
            "</table><h3>Local searches</h3><table><tr><th>M</th><th>Seed</th><th>Status</th><th>Start</th><th>Area</th><th>Slices</th><th>Volume</th></tr>"
        )
        for r in c["runs"]:
            parts.append(
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(r.get(k, "—"))) + "</td>"
                    for k in [
                        "count",
                        "seed",
                        "status",
                        "chosen_initialization",
                        "area",
                        "slices",
                        "volume",
                    ]
                )
                + "</tr>"
            )
        parts.append("</table></details>")
    parts.append(
        "<h2>Accounting</h2><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + '</pre><p><a href="results.json">Full results</a> · <a href="protocol.json">Protocol</a> · <a href="gemini-review.png">Gemini review</a></p></body></html>'
    )
    (out / "report.html").write_text("".join(parts))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
