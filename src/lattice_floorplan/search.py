"""Small, transparent per-instance search baselines; no generalization claim."""

import json
import math
import random
from pathlib import Path
from .circuit import Circuit


class MappingEnv:
    """Sequentially choose which logical ID occupies the next row-major Q site.

    The first n Q sites are used, matching upstream allocation. Unused sites are
    not additional placement actions. Reward is strictly decreasing in successful
    slice count and bounded above failure: 1/(1+T/T_ref), failure=0.
    """

    def __init__(self, circuit, layout, evaluator, reference_slices):
        """Bind the terminal mapping evaluator and positive slice reference; validate capacity.

        Args:
            circuit (Circuit): Parsed LLI declarations and operations.
            layout (Layout): Parsed ASCII layout with row-major Q sites.
            evaluator (Callable[[tuple[int, ...]], dict[str, Any]]): Terminal mapping callback
                returning status and slices (None on failure).
            reference_slices (int): Positive reference slice count used to normalize successful
                terminal reward.

        Raises:
            ValueError: Insufficient sites or invalid reference count.
        """
        if len(layout.sites) < len(circuit.ids) or reference_slices < 1:
            raise ValueError("Insufficient sites or invalid reference count")
        self.circuit, self.layout = circuit, layout
        self.evaluator, self.reference_slices = evaluator, reference_slices
        self.reset()

    def reset(self):
        """Clear the partial site order and return the initial observation.

        Returns:
            dict[str, Any]: Empty site_order, site coordinates, logical IDs, operations and
                action_mask.
        """
        self.order = []
        return self.observation()

    def observation(self):
        """Return the partial mapping, circuit operations and unused-logical-ID mask.

        Returns:
            dict[str, Any]: Current site_order and action_mask aligned to logical_ids, plus
                sites and operations.
        """
        return {
            "site_order": self.order.copy(),
            "sites": self.layout.sites[: len(self.circuit.ids)],
            "action_mask": [q not in self.order for q in self.circuit.ids],
            "logical_ids": self.circuit.ids,
            "operations": self.circuit.operations,
        }

    def step(self, action):
        """Assign a logical ID to the next site and return observation, reward, done, info.

        Only the final assignment calls the evaluator. Its reward is
        1/(1+slices/reference_slices) on success and zero on failure;
        intermediate rewards are zero. Invalid or terminal actions raise ValueError.

        Args:
            action (int): Unused declared logical ID assigned to the next row-major site.

        Returns:
            tuple[dict[str, Any], float, bool, dict[str, Any]]: Next observation, reward,
                completion flag and terminal compiler record (empty before completion).

        Raises:
            ValueError: Episode is complete or the requested logical ID is unavailable.
        """
        if len(self.order) == len(self.circuit.ids):
            raise ValueError("Episode finished; reset first")
        if action not in self.circuit.ids or action in self.order:
            raise ValueError("Action is unavailable")
        self.order.append(action)
        done = len(self.order) == len(self.circuit.ids)
        info, reward = {}, 0.0
        if done:
            info = self.evaluator(tuple(self.order))
            reward = (
                1 / (1 + info["slices"] / self.reference_slices)
                if info["status"] == "success"
                else 0.0
            )
        return self.observation(), reward, done, info


def optimize(
    compiler, circuit_text, layout, directory, method="anneal", budget=32, seed=0, cooling="linear"
):
    """Budget includes identity evaluation. No cache: every trial is a compiler call.

    Args:
        compiler (Compiler): Configured native compiler adapter; set CompilerConfig(local=True)
            for current local evaluations.
        circuit_text (str): Circuit source in the input format accepted by the caller; no
            synthesis is performed.
        layout (Layout): Parsed ASCII layout with row-major Q sites.
        directory (str | os.PathLike[str]): Destination directory for the artifacts described
            above.
        method (str): Historical mapping algorithm: 'random', 'anneal' or 'reinforce'.
        budget (int): Charged evaluation slots, including the baseline and repeated requests;
            not necessarily native compiler calls.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.
        cooling (str): Temperature schedule: 'linear', 'exponential' or 'logarithmic'.

    Returns:
        dict[str, Any]: Baseline/best slices, winning declaration order, charged compiler calls
            and trial history; artifacts are written to directory.

    Raises:
        ValueError: Invalid method, cooling or budget, insufficient Q sites, or failed identity compilation.
    """
    if (
        method not in ("random", "anneal", "reinforce")
        or cooling not in ("linear", "exponential", "logarithmic")
        or budget < 1
    ):
        raise ValueError("Unknown method or budget < 1")
    circuit = Circuit.parse(circuit_text)
    if len(layout.sites) < len(circuit.ids):
        raise ValueError("Layout has fewer Q sites than circuit qubits")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng, history, best = random.Random(seed), [], None

    def evaluate(order):
        """Compile one declaration ordering, append its evidence and retain the best success.

        Args:
            order (Sequence[int]): Permutation of declared logical IDs in compiler allocation
                order.

        Returns:
            dict[str, Any]: Compiler record augmented with site_order.
        """
        nonlocal best
        result = compiler.run(
            circuit.mapped(order), layout, directory / f"trial-{len(history):04d}"
        )
        result["site_order"] = list(order)
        history.append(result)
        if result["status"] == "success" and (best is None or result["slices"] < best["slices"]):
            best = result
        return result

    current = evaluate(circuit.ids)
    if current["status"] != "success":
        raise ValueError(
            f"Identity mapping must compile before optimization; inspect {directory}/trial-0000"
        )
    reference = current["slices"]
    order, n = list(circuit.ids), len(circuit.ids)
    logits = [[0.0] * n for _ in range(n)]
    baseline = 0.5
    env = MappingEnv(circuit, layout, evaluate, reference)
    for trial in range(1, budget):
        if method == "reinforce":
            env.reset()
            trajectory = []
            for site in range(n):
                available = [i for i, q in enumerate(circuit.ids) if q not in env.order]
                maximum = max(logits[site][i] for i in available)
                weights = [math.exp(logits[site][i] - maximum) for i in available]
                probs = [w / sum(weights) for w in weights]
                selected = rng.choices(available, weights=probs, k=1)[0]
                trajectory.append((site, available, probs, selected))
                _, reward, _, result = env.step(circuit.ids[selected])
            # REINFORCE: gradient log pi(a|s) = one_hot(a) - probabilities.
            advantage = reward - baseline
            for site, available, probs, selected in trajectory:
                for i, p in zip(available, probs):
                    logits[site][i] += 0.3 * advantage * ((i == selected) - p)
            baseline = 0.9 * baseline + 0.1 * reward
        else:
            candidate = order.copy()
            if method == "random":
                rng.shuffle(candidate)
            elif n > 1:
                a, b = rng.sample(range(n), 2)
                candidate[a], candidate[b] = candidate[b], candidate[a]
            result = evaluate(candidate)
            if result["status"] == "success":
                factor = {
                    "linear": 1 - trial / budget,
                    "exponential": 0.01 ** (trial / budget),
                    "logarithmic": 1 / math.log(trial + 2),
                }[cooling]
                temperature = max(0.01, reference * 0.2 * factor)
                delta = result["slices"] - current["slices"]
                if (
                    method == "random"
                    or delta <= 0
                    or rng.random() < math.exp(-delta / temperature)
                ):
                    order, current = candidate, result
    summary = {
        "method": method,
        "seed": seed,
        "budget": budget,
        "compiler_calls": len(history),
        "cooling": cooling if method == "anneal" else None,
        "reference_slices": reference,
        "best_slices": best["slices"],
        "best_site_order": best["site_order"],
        "successful_trials": sum(r["status"] == "success" for r in history),
        "history": [
            {"slices": r["slices"], "status": r["status"], "site_order": r["site_order"]}
            for r in history
        ],
        "scope": "Per-circuit optimization; no held-out generalization result",
    }
    (directory / "best.lli").write_text(circuit.mapped(best["site_order"]))
    (directory / "layout.txt").write_text(layout.text)
    if method == "reinforce":
        (directory / "policy.json").write_text(
            json.dumps(
                {"logical_ids": circuit.ids, "logits": logits, "baseline": baseline}, indent=2
            )
        )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def optimize_grid(compiler, circuit_text, layout, directory, budget=32, seed=0):
    """Greedy stochastic Q/r relocation; factory output access is protected.

    Row-major mapping changes as sites move; this is joint geometry/mapping search,
    not a pure geometry ablation. Illegal candidates are recorded as rejected.

    Args:
        compiler (Compiler): Configured native compiler adapter; set CompilerConfig(local=True)
            for current local evaluations.
        circuit_text (str): Circuit source in the input format accepted by the caller; no
            synthesis is performed.
        layout (Layout): Parsed ASCII layout with row-major Q sites.
        directory (str | os.PathLike[str]): Destination directory for the artifacts described
            above.
        budget (int): Charged evaluation slots, including the baseline and repeated requests;
            not necessarily native compiler calls.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.

    Returns:
        dict[str, Any]: Slice-score summary and trial history; the winning ASCII/SVG files are
            written to directory.

    Raises:
        ValueError: Budget is nonpositive or the starting grid fails compilation.
    """
    if budget < 1:
        raise ValueError("Budget must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    current = layout
    first = compiler.run(circuit_text, current, directory / "trial-0000")
    if first["status"] != "success":
        raise ValueError("Starting grid does not compile; inspect trial-0000")
    best, history, rejected = first, [first], 0
    for trial in range(1, budget):
        options = []
        for source in current.sites:
            for target in current.neighbors(source):
                try:
                    options.append(current.move_data(source, target))
                except ValueError:
                    pass
        if not options:
            break
        candidate = rng.choice(options)
        result = compiler.run(circuit_text, candidate, directory / f"trial-{trial:04d}")
        history.append(result)
        if result["status"] == "success" and result["slices"] <= best["slices"]:
            best, current = result, candidate
        else:
            rejected += 1
    (directory / "best-layout.txt").write_text(current.text)
    (directory / "best-layout.svg").write_text(current.svg("Best evaluated grid"))
    summary = {
        "method": "grid-local-search",
        "seed": seed,
        "budget": budget,
        "compiler_calls": len(history),
        "reference_slices": first["slices"],
        "best_slices": best["slices"],
        "rejected_trials": rejected,
        "scope": "Joint geometry/mapping; fixed tile counts and factory output access",
        "history": [{"slices": r["slices"], "status": r["status"]} for r in history],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
