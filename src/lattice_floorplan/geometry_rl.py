"""PPO proposals over variable-size grids and explicit qubit identities.

Candidate scorer supports an unbounded number of data destinations. Compound
and resize actions are stochastic macros. Learned acceptance is an optimization
heuristic, not a claim of Boltzmann sampling.
"""

from dataclasses import dataclass, asdict
import random
import math
import time
import numpy as np
import torch
from torch import nn
from .geometry import (
    as_state,
    propose_action,
    sample_action,
    spatial_observation,
    geometry_scope,
    Proposal,
    MOVE_SET_VERSION,
    state_digest,
)
from .ppo import gae, clipped_loss
from .neural import seed_everything


@dataclass
class GridConfig:
    """Variable-geometry PPO and annealing-acceptance controls.

    training_budget includes the baseline. temperature_fraction scales initial
    tile-slice energy and end_factor cools it; entropy, clip and epochs govern PPO.

    Attributes:
        learning_rate (float): Adam step size.
        entropy (float): Entropy bonus coefficient.
        clip (float): PPO likelihood-ratio clipping width.
        temperature_fraction (float): Initial acceptance temperature relative to baseline
            volume.
        end_factor (float): Multiplicative temperature factor at the episode horizon.
        batch_episodes (int): Complete episodes per rollout batch.
        epochs (int): Optimization passes per rollout batch.
        training_budget (int): At least two charged evaluations per training episode, including
            baseline.
    """

    learning_rate: float = 0.0003
    entropy: float = 0.01
    clip: float = 0.2
    temperature_fraction: float = 0.001
    end_factor: float = 0.1
    batch_episodes: int = 8
    epochs: int = 4
    training_budget: int = 8


class GridEnv:
    """Compiler-scored geometry episode with best-volume rewards and annealed state transitions."""

    def __init__(self, task, oracle, grid, budget, seed, config):
        """Charge the baseline evaluation and initialize an independent seeded episode.

        task supplies name, family, qasm, features and adjacency. budget must
        allow a baseline and proposal. A failed baseline raises ValueError;
        the oracle configuration determines local compilation explicitly.

        Args:
            task (Mapping[str, Any]): Circuit record with name, family, qasm, features (N,F) and
                adjacency (N,N).
            oracle (GeometryOracle): Object providing evaluate(qasm, grid) -> dict with status
                and slices; failures have slices=None. The configured backend sets local mode.
            grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
                identities in row-major Q order.
            budget (int): Charged evaluation slots, including the baseline and repeated
                requests; not necessarily native compiler calls.
            seed (int): Seed for reproducible sampling; global RNG effects, if any, are
                described above.
            config (GridConfig): Geometry PPO learning, reward-independent acceptance and budget
                controls.

        Raises:
            ValueError: Budget is below two or the initial geometry fails compilation.
        """
        self.task = task
        self.oracle = oracle
        self.grid = self.best_grid = as_state(grid)
        self.budget = budget
        self.rng = random.Random(seed)
        self.config = config
        if budget < 2:
            raise ValueError("Need baseline and proposal")
        first = oracle.evaluate(task["qasm"], self.grid)
        if first["status"] != "success":
            raise ValueError("Starting geometry does not compile")
        self.baseline_slices = first["slices"]
        self.best_slices = first["slices"]
        self.baseline = self.current = self.best = first["slices"] * self.grid.area
        self.baseline_area = self.grid.area
        self.best_record = first
        self.queries = 1
        self.failures = 0
        self.seen = {self.grid}
        self.done = False
        self.pareto = [
            {
                "slices": first["slices"],
                "area": self.grid.area,
                "grid": self.grid.text,
                **self.grid.identity_record(),
            }
        ]
        self.history = [
            {
                "query": 1,
                "status": "success",
                "slices": self.best_slices,
                "best": self.best,
                "grid_sha256": state_digest(self.grid),
                "accepted": True,
                "area": self.grid.area,
            }
        ]

    def observation(self):
        """Return spatial/action features and normalized current/best volume and progress.

        Returns:
            dict[str, Any]: planes, context, actions and action_features from
                spatial_observation.
        """
        return spatial_observation(
            self.grid,
            self.task["features"],
            self.task["adjacency"],
            self.current,
            self.baseline,
            self.best,
            (self.queries - 1) / (self.budget - 1),
        )

    def step(self, action, uniform_proposal=False):
        """Evaluate one geometry action, update the episode and return its reward.

        Failure reward is -0.2; success earns 10*max(0,best-volume)/baseline.
        Null and failed proposals still consume a query. Acceptance uses a
        cooling volume temperature; only uniform_proposal=True adds the known
        uniform-kernel proposal ratio. Learned proposals are a search heuristic,
        not exact Boltzmann sampling. Raises ValueError after episode completion.

        Args:
            action (tuple[str | int, ...]): Geometry action from proposal_actions/sample_action,
                including stochastic macro descriptors.
            uniform_proposal (bool): Apply the uniform-kernel proposal correction when True;
                learned proposals omit it.

        Returns:
            float: Best-volume improvement reward, zero without improvement, or -0.2 for
                compiler failure.

        Raises:
            ValueError: Episode finished.
        """
        if self.done:
            raise ValueError("Episode finished")
        p = (
            propose_action(self.grid, action, self.rng)
            if not action[0].startswith("null_")
            else Proposal(self.grid, action[0][5:], action, null_reason="no_action")
        )
        r = self.oracle.evaluate(self.task["qasm"], p.state)
        self.seen.add(p.state)
        self.queries += 1
        take = False
        reward = 0.0
        if r["status"] != "success":
            self.failures += 1
            reward = -0.2
        else:
            if not any(
                x["slices"] <= r["slices"] and x["area"] <= p.state.area for x in self.pareto
            ):
                self.pareto = [
                    x
                    for x in self.pareto
                    if not (r["slices"] <= x["slices"] and p.state.area <= x["area"])
                ]
                self.pareto.append(
                    {
                        "slices": r["slices"],
                        "area": p.state.area,
                        "grid": p.state.text,
                        **p.state.identity_record(),
                    }
                )
            score = r["slices"] * p.state.area
            reward = 10 * max(0, self.best - score) / max(1, self.baseline)
            if (score, p.state.area) < (self.best, self.best_grid.area):
                self.best = score
                self.best_slices = r["slices"]
                self.best_grid = p.state
                self.best_record = r
            temp = max(
                1e-9,
                self.baseline
                * self.config.temperature_fraction
                * self.config.end_factor ** ((self.queries - 1) / (self.budget - 1)),
            )
            logp = -(score - self.current) / temp + (
                p.log_reverse_over_forward if uniform_proposal else 0.0
            )
            take = p.null_reason is None and (logp >= 0 or self.rng.random() < math.exp(logp))
            if take:
                self.grid = p.state
                self.current = score
        self.history.append(
            {
                "query": self.queries,
                **p.metadata(),
                "status": r["status"],
                "slices": r["slices"],
                "objective": r["slices"] * p.state.area if r["status"] == "success" else None,
                "best": self.best,
                "grid_sha256": state_digest(p.state),
                "accepted": take,
                "area": p.state.area,
            }
        )
        self.done = self.queries == self.budget
        return reward

    def result(self):
        """Return best-volume grid/identity, compiler evidence, query history and frontier.

        Returns:
            dict[str, Any]: Best grid/identity, volume/slices/area, compiler record, history and
                observed frontier.
        """
        return {
            "circuit": self.task["name"],
            "family": self.task["family"],
            "geometry_scope": geometry_scope(),
            "null_proposals": sum(h.get("null_reason") is not None for h in self.history),
            "objective": "area_times_slices",
            "baseline_objective": self.baseline,
            "best_objective": self.best,
            "baseline": self.baseline_slices,
            "baseline_area": self.baseline_area,
            "best_slices": self.best_slices,
            "best_area": self.best_grid.area,
            "improvement": self.baseline - self.best,
            "fractional_improvement": (self.baseline - self.best) / self.baseline,
            "queries": self.queries,
            "unique_grids": len(self.seen),
            "failed_candidates": self.failures,
            "best_grid": self.best_grid.text,
            **self.best_grid.identity_record(),
            "best_record": self.best_record,
            "history": self.history,
            "pareto": sorted(self.pareto, key=lambda x: x["area"]),
        }


def observations(envs):
    """Return a list of variable-size environment observations without spatial padding.

    Args:
        envs (Sequence[GridEnv]): Nonempty collection of nonterminal environments to encode.

    Returns:
        list[dict[str, Any]]: One unpadded spatial observation per environment.
    """
    return [e.observation() for e in envs]


class GridPolicy(nn.Module):
    """Spatial CNN actor/critic scoring a variable number of geometry action descriptors."""

    def __init__(self):
        """Build seven-plane convolution, pooled context, candidate actor and scalar critic."""
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(7, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        # Circuit features summarized rather than flattened to a fixed qubit count.
        self.shared = nn.Sequential(nn.Linear(64 + 7, 128), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(128 + 10, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        """Return padded action logits (B,max_actions) and values (B,).

        Individual grids are encoded separately; missing action slots use -1e9.
        Compound/resize logits select macros, whose endpoints remain stochastic.

        Args:
            obs (Sequence[Mapping[str, Any]]): Observation dictionaries with float32 planes
                (7,H,W), context vector and action_features (A,10); H,W,A may differ per item.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Padded logits (B,max_actions) and values (B,);
                padding is -1e9.
        """
        logits = []
        values = []
        for o in obs:
            spatial = self.spatial(torch.as_tensor(o["planes"]).unsqueeze(0))
            ctx = torch.as_tensor(o["context"])
            prefix = ctx[:-3]
            summary = torch.stack(
                [prefix.mean(), prefix.std(unbiased=False), prefix.min(), prefix.max(), *ctx[-3:]]
            )[None]
            h = self.shared(torch.cat([spatial, summary], -1))
            af = torch.as_tensor(o["action_features"])
            logits.append(self.actor(torch.cat([h.expand(len(af), -1), af], -1)).squeeze(-1))
            values.append(self.critic(h).squeeze())
        return nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=-1e9), torch.stack(
            values
        )


def load_policy_checkpoint(saved):
    """Load a matching move-set state_dict in eval mode; reject legacy geometry versions.

    Args:
        saved (Mapping[str, Any]): Checkpoint containing move_set_version and a compatible Torch
            state_dict.

    Returns:
        GridPolicy: Loaded network in evaluation mode.

    Raises:
        ValueError: The checkpoint move-set version is incompatible.
        KeyError: state_dict is missing.
        RuntimeError: Torch parameter names or shapes do not match the current policy.
    """
    if saved.get("move_set_version") != MOVE_SET_VERSION:
        raise ValueError(
            "Legacy fixed-grid PPO checkpoint is incompatible; train a geometry-v3 policy"
        )
    model = GridPolicy()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def learn(tasks, oracle, grid, config, seed, episodes):
    """Train grid PPO and return an eval-mode policy with query and learning logs.

    Episodes must divide into config.batch_episodes. Oracle requests include
    baselines and failures; actual subprocess costs are tracked by the oracle.
    Uses terminal GAE with gamma=1 and rejects nonfinite training losses.

    Args:
        tasks (Sequence[Mapping[str, Any]]): Circuit records containing name, family, qasm,
            features and adjacency.
        oracle (GeometryOracle): Object providing evaluate(qasm, grid) -> dict with status and
            slices; failures have slices=None. The configured backend sets local mode.
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        config (GridConfig): Geometry PPO learning, reward-independent acceptance and budget
            controls.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.
        episodes (int): Number of training episodes; must form complete batches where required.

    Returns:
        tuple[GridPolicy, dict[str, Any]]: Eval-mode policy and training/query accounting log.

    Raises:
        ValueError: Incomplete rollout batches.
        FloatingPointError: Nonfinite grid PPO loss.
    """
    if episodes % config.batch_episodes:
        raise ValueError("Incomplete rollout batches")
    seed_everything(seed)
    rng = random.Random(seed)
    model = GridPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    curve = []
    started = time.monotonic()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for batch in range(episodes // config.batch_episodes):
        envs = [
            GridEnv(
                rng.choice(tasks),
                oracle,
                grid,
                config.training_budget,
                rng.randrange(2**30),
                config,
            )
            for _ in range(config.batch_episodes)
        ]
        buffers = []
        actions = []
        old = []
        values = []
        rewards = []
        for step in range(config.training_budget - 1):
            obs = observations(envs)
            with torch.no_grad():
                logits, v = model(obs)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
            buffers.extend(obs)
            actions.append(a)
            old.append(logp)
            values.append(v)
            rewards.append(
                torch.tensor([e.step(o["actions"][int(i)]) for e, o, i in zip(envs, obs, a)])
            )
        adv, ret = gae(torch.stack(rewards), torch.stack(values), 1.0, 0.95)
        adv = adv.flatten()
        ret = ret.flatten()
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        actions = torch.cat(actions)
        old = torch.cat(old)
        losses = []
        kls = []
        for epoch in range(config.epochs):
            logits, v = model(buffers)
            d = torch.distributions.Categorical(logits=logits)
            lp = d.log_prob(actions)
            loss = (
                clipped_loss(lp, old, adv, config.clip)
                + 0.5 * (v - ret).square().mean()
                - config.entropy * d.entropy().mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite grid PPO loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
            with torch.no_grad():
                lr = lp - old
                kls.append(float((lr.exp() - 1 - lr).mean()))
        point = {
            "episodes": (batch + 1) * config.batch_episodes,
            "mean_improvement": float(np.mean([(e.baseline - e.best) / e.baseline for e in envs])),
            "failed_candidates": sum(e.failures for e in envs),
            "loss": float(np.mean(losses)),
            "approx_kl": float(np.mean(kls)),
        }
        curve.append(point)
    model.eval()
    change = (
        sum(float((v - before[k]).square().sum()) for k, v in model.state_dict().items()) ** 0.5
    )
    return model, {
        "move_set_version": MOVE_SET_VERSION,
        "episodes": episodes,
        "search_queries": episodes * config.training_budget,
        "wall_seconds": time.monotonic() - started,
        "parameter_change_l2": change,
        "curve": curve,
        "config": asdict(config),
    }


@torch.no_grad()
def search(task, oracle, grid, config, seed, budget=16, model=None):
    """Run one budgeted grid episode and return its best-observed result.

    model=None selects uniform move families with proposal-ratio correction;
    otherwise sample learned action logits with heuristic annealed acceptance.
    No export or fresh replay is performed here. The oracle sets compiler mode.

    Args:
        task (Mapping[str, Any]): Circuit record with name, family, qasm, features (N,F) and
            adjacency (N,N).
        oracle (GeometryOracle): Object providing evaluate(qasm, grid) -> dict with status and
            slices; failures have slices=None. The configured backend sets local mode.
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        config (GridConfig): Geometry PPO learning, reward-independent acceptance and budget
            controls.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.
        budget (int): Charged evaluation slots, including the baseline and repeated requests;
            not necessarily native compiler calls.
        model (GridPolicy | None): Geometry action policy, or None for uniform proposal
            families.

    Returns:
        dict[str, Any]: Best observed episode result, including grid, identity, evidence and
            charged history.
    """
    env = GridEnv(task, oracle, grid, budget, seed, config)
    rng = random.Random(seed + 104729)
    while not env.done:
        if model is None:
            action = sample_action(env.grid, rng)
        else:
            obs = observations([env])
            logits, _ = model(obs)
            i = rng.choices(range(len(obs[0]["actions"])), weights=logits[0].softmax(-1).tolist())[
                0
            ]
            action = obs[0]["actions"][i]
        env.step(action, uniform_proposal=model is None)
    return env.result()
