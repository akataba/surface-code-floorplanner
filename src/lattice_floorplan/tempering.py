"""Compiler-scored SA, restart SA and replica exchange on variable-size grids."""

import math
import random
import statistics
import time
from dataclasses import dataclass, asdict
from .geometry import as_state, single_candidates, propose, geometry_scope, state_digest


@dataclass(frozen=True)
class TemperingConfig:
    """Search controls; temperatures are calibrated in the selected objective units.

    Attributes:
        cold_accept (float): Target uphill acceptance at the cold end;
            0<cold_accept<hot_accept<1.
        hot_accept (float): Target uphill acceptance at the hot end.
        replicas (int): At least two chains for tempering/restart configuration.
        calibration_queries (int): Positive number of charged temperature probes.
        swap_sweeps (int): Positive number of complete replica sweeps between exchange events.
        objective (str): 'area_times_slices' or 'augmented_chebyshev'.
        area_weight (float): Chebyshev area weight strictly between zero and one.
        move_supply (bool): Include fixed-count M relocation as a seventh family when True.
    """

    cold_accept: float = 0.02
    hot_accept: float = 0.9
    replicas: int = 4
    calibration_queries: int = 8
    swap_sweeps: int = 2
    objective: str = "area_times_slices"
    area_weight: float = 0.5
    move_supply: bool = False

    def __post_init__(self):
        """Reject invalid objective names, acceptance targets, area weights or counts.

        Raises:
            ValueError: Unknown objective, invalid area weight/acceptance targets, or invalid replica/calibration/sweep counts.
        """
        if self.objective not in ("area_times_slices", "augmented_chebyshev"):
            raise ValueError("Unknown objective")
        if not 0 < self.area_weight < 1:
            raise ValueError("Area weight must be strictly between zero and one")
        if not 0 < self.cold_accept < self.hot_accept < 1:
            raise ValueError("Require 0<cold<hot<1")
        if self.replicas < 2 or self.calibration_queries < 1 or self.swap_sweeps < 1:
            raise ValueError("Positive counts required")


def local_log_accept(old, new, temperature, old_degree, new_degree):
    """Return the untruncated log Metropolis-Hastings ratio for uniform neighbors.

    old/new are objective energies; temperature and both degrees must be positive.
    The degree term corrects asymmetric neighbor counts.

    Args:
        old (float): Current objective energy.
        new (float): Proposed objective energy.
        temperature (float): Positive temperature in the same units as the objective energy.
        old_degree (int): Positive number of valid outgoing neighbors from the current state.
        new_degree (int): Positive number of valid outgoing neighbors from the proposed state.

    Returns:
        float: Untruncated log acceptance ratio.
    """
    return -(new - old) / temperature + math.log(old_degree) - math.log(new_degree)


def swap_log_accept(ei, ej, ti, tj):
    """Return the log replica-swap ratio for energies ei/ej at positive ti/tj.

    Args:
        ei (float): Energy of the replica at temperature ti.
        ej (float): Energy of the replica at temperature tj.
        ti (float): Positive temperature of the first replica.
        tj (float): Positive temperature of the second replica.

    Returns:
        float: Log replica-exchange acceptance ratio.
    """
    return (1 / ti - 1 / tj) * (ei - ej)


def accepted(log_ratio, rng):
    """Apply min(1, exp(log_ratio)); favorable moves consume no random draw.

    Args:
        log_ratio (float): Log acceptance ratio before truncation to probability one.
        rng (random.Random): Caller-owned pseudorandom generator; sampling advances its state.

    Returns:
        bool: Whether the proposal is accepted.
    """
    return log_ratio >= 0 or rng.random() < math.exp(log_ratio)


def search(task, oracle, grid, method, budget, seed, config=None):
    """Run an exactly budgeted search and retain measured nondominated candidates.

    The oracle supplies status/slices records. Failed and null proposals consume
    queries, as do cache hits; budget is not a count of native subprocesses.
    Failed candidates are excluded from energy arithmetic and rejected, not
    converted to infinite penalties. Independent RNG streams isolate calibration,
    chains and swaps.
    Export and fresh verification are the caller's responsibility (see api).

    Args:
        task (Mapping[str, str]): Circuit record containing name, family and synthesized qasm.
        oracle (GeometryOracle): Object providing evaluate(qasm, grid) -> dict with status and
            slices; failures have slices=None. The configured backend sets local mode.
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        method (str): Search algorithm: 'sa', 'restart_sa' or 'parallel_tempering'.
        budget (int): Charged evaluation slots, including the baseline and repeated requests;
            not necessarily native compiler calls.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.
        config (TemperingConfig | None): Search controls; None uses TemperingConfig defaults.

    Returns:
        dict[str, Any]: Best grid/identity and score, baseline, exact query history, failed/null
            counts, observed Pareto points, calibration and swap diagnostics.

    Raises:
        ValueError: Unknown method, insufficient budget, or failed starting compilation.
    """
    config = config or TemperingConfig()
    grid = as_state(grid)
    if method not in ("sa", "restart_sa", "parallel_tempering"):
        raise ValueError("Unknown method")
    if budget < 1 + config.calibration_queries + config.replicas:
        raise ValueError("Budget too small")
    if config.move_supply:
        from .supply import propose_with_supply

        proposal = propose_with_supply
    else:
        proposal = propose
    started = time.monotonic()
    history = []
    seen = set()
    failures = 0
    best = None
    best_grid = None
    best_record = None
    pareto = []
    reference_slices = None

    def energy(slices, g):
        """Compute tile-slice volume or the baseline-normalized Chebyshev objective.

        Args:
            slices (int): Positive completed compiler-frame count for a successful candidate.
            g (GridState): Candidate state with explicit logical coordinates.

        Returns:
            float | int: Candidate energy in the configured objective units.
        """
        if config.objective == "area_times_slices":
            return slices * g.area
        a = config.area_weight * g.area / grid.area
        t = (1 - config.area_weight) * slices / reference_slices
        return grid.area * reference_slices * (max(a, t) + 0.01 * (a + t))

    def evaluate(g, phase, **extra):
        """Charge one oracle request and update history, best state and observed frontier.

        Args:
            g (GridState): Candidate state with explicit logical coordinates.
            phase (str): History label identifying baseline, calibration or search evaluation.
            **extra (Any): Additional phase/proposal fields merged into the history record.

        Returns:
            dict[str, Any]: Oracle record, also appended to this search history.
        """
        nonlocal failures, best, best_grid, best_record, pareto, reference_slices
        rec = oracle.evaluate(task["qasm"], g)
        seen.add(g)
        if rec["status"] == "success":
            if reference_slices is None:
                reference_slices = rec["slices"]
            if best is None or (energy(rec["slices"], g), rec["slices"], g.area) < (
                energy(best, best_grid),
                best,
                best_grid.area,
            ):
                best = rec["slices"]
                best_grid = g
                best_record = rec
            if not any(p["slices"] <= rec["slices"] and p["area"] <= g.area for p in pareto):
                pareto = [
                    p for p in pareto if not (rec["slices"] <= p["slices"] and g.area <= p["area"])
                ]
                pareto.append(
                    {
                        "slices": rec["slices"],
                        "area": g.area,
                        "grid": g.text,
                        **g.identity_record(),
                        "cache_key": rec.get("cache_key"),
                    }
                )
        else:
            failures += 1
        history.append(
            {
                "query": len(history) + 1,
                "phase": phase,
                "status": rec["status"],
                "slices": rec["slices"],
                "objective": energy(rec["slices"], g) if rec["status"] == "success" else None,
                "best": best,
                "area": g.area,
                "grid_sha256": state_digest(g),
                "cache_key": rec.get("cache_key"),
                **extra,
            }
        )
        return rec

    first = evaluate(grid, "baseline")
    if first["status"] != "success":
        raise ValueError("Starting grid did not compile")
    baseline = first["slices"]
    crng = random.Random(seed)
    candidates = single_candidates(grid)
    # Startup temperature probes do not constrain subsequent compound/resize proposals.
    calibration = []
    for j in range(config.calibration_queries):
        if j == 0:
            probes = crng.sample(candidates, min(len(candidates), config.calibration_queries))
        if j < len(probes):
            a, g = probes[j]
            meta = {"action": ["single", *a]}
        else:
            p = proposal(grid, crng)
            g = p.state
            meta = p.metadata()
        rec = evaluate(g, "calibration", **meta)
        if rec["status"] == "success":
            calibration.append(energy(rec["slices"], g) - energy(baseline, grid))
    positive = [x for x in calibration if x > 0]
    nonzero = [abs(x) for x in calibration if x]
    scale = statistics.median(positive or nonzero or [1])
    source = (
        "positive_deltas" if positive else ("absolute_deltas" if nonzero else "one_slice_fallback")
    )
    low = scale / -math.log(config.cold_accept)
    high = scale / -math.log(config.hot_accept)
    temperatures = [
        low * (high / low) ** (i / (config.replicas - 1)) for i in range(config.replicas)
    ]
    remaining = budget - len(history)
    chains = [
        {"grid": grid, "energy": energy(baseline, grid), "label": i} for i in range(config.replicas)
    ]
    rngs = [random.Random(seed + 100003 * (i + 1)) for i in range(config.replicas)]
    swap_rng = random.Random(seed + 999983)

    def step(i, temp):
        """Charge one chain proposal and apply its temperature and proposal-ratio correction.

        Args:
            i (int): Zero-based replica/chain index.
            temp (float): Positive chain temperature in objective units.
        """
        chain = chains[i]
        p = proposal(chain["grid"], rngs[i])
        rec = evaluate(
            p.state,
            "search",
            chain=i,
            temperature=temp,
            previous_energy=chain["energy"],
            **p.metadata(),
        )
        take = False
        if rec["status"] == "success" and p.null_reason is None:
            take = accepted(
                -(energy(rec["slices"], p.state) - chain["energy"]) / temp
                + p.log_reverse_over_forward,
                rngs[i],
            )
        if take:
            chain["grid"] = p.state
            chain["energy"] = energy(rec["slices"], p.state)
        history[-1]["accepted"] = take
        history[-1]["current_energy"] = chain["energy"]

    swaps = []
    attempts = [0] * (config.replicas - 1)
    accepts = [0] * (config.replicas - 1)
    roundtrips = [0] * config.replicas
    phases = [0] * config.replicas
    phases[0] = 1
    if method == "parallel_tempering":
        events = 0
        for k in range(remaining):
            step(k % config.replicas, temperatures[k % config.replicas])
            if (k + 1) % (config.replicas * config.swap_sweeps) == 0:
                for j in range(events % 2, config.replicas - 1, 2):
                    attempts[j] += 1
                    logp = swap_log_accept(
                        chains[j]["energy"],
                        chains[j + 1]["energy"],
                        temperatures[j],
                        temperatures[j + 1],
                    )
                    take = accepted(logp, swap_rng)
                    swaps.append(
                        {
                            "after_query": len(history),
                            "edge": j,
                            "accepted": take,
                            "energies": [chains[j]["energy"], chains[j + 1]["energy"]],
                        }
                    )
                    if take:
                        chains[j], chains[j + 1] = chains[j + 1], chains[j]
                        accepts[j] += 1
                events += 1
                cold = chains[0]["label"]
                hot = chains[-1]["label"]
                if phases[cold] == 2:
                    roundtrips[cold] += 1
                phases[cold] = 1
                if phases[hot] == 1:
                    phases[hot] = 2
    else:
        lengths = (
            [remaining]
            if method == "sa"
            else [
                remaining // config.replicas + (i < remaining % config.replicas)
                for i in range(config.replicas)
            ]
        )
        for i, length in enumerate(lengths):
            for k in range(length):
                step(i, high * (low / high) ** (k / max(1, length - 1)))
    assert len(history) == budget and best_record["status"] == "success"
    return {
        "circuit": task["name"],
        "family": task["family"],
        "method": method,
        "budget": budget,
        "seed": seed,
        "config": asdict(config),
        "geometry_scope": {
            **geometry_scope(config.objective),
            "supply_placement_search": config.move_supply,
            "proposal_distribution": (
                "seven equal families including supply relocation"
                if config.move_supply
                else geometry_scope()["proposal_distribution"]
            ),
        },
        "baseline": baseline,
        "baseline_area": grid.area,
        "best_slices": best,
        "best_area": best_grid.area,
        "objective": config.objective,
        "area_weight": config.area_weight,
        "baseline_objective": energy(baseline, grid),
        "best_objective": energy(best, best_grid),
        "fractional_improvement": (energy(baseline, grid) - energy(best, best_grid))
        / (energy(baseline, grid)),
        "slice_fractional_improvement": (baseline - best) / baseline,
        "best_grid": best_grid.text,
        **best_grid.identity_record(),
        "best_record": best_record,
        "queries": len(history),
        "unique_grids": len(seen),
        "failed_candidates": failures,
        "null_proposals": sum(h.get("null_reason") is not None for h in history),
        "calibration": {
            "deltas": calibration,
            "scale": scale,
            "source": source,
            "temperatures": temperatures,
        },
        "history": history,
        "swaps": swaps,
        "swap_attempts": attempts,
        "swap_accepts": accepts,
        "roundtrips": roundtrips,
        "pareto": sorted(pareto, key=lambda p: p["area"]),
        "wall_seconds_with_shared_cache": time.monotonic() - started,
        "sampling_scope": "MH proposal ratios verified; unbounded-area search has no claimed normalizable Boltzmann equilibrium",
    }
