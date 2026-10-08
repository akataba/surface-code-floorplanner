"""Masked PPO for budgeted constructive mapping and three-qubit repair.

Identity +31 complete candidates; no oracle extrema in observations or rewards.
"""

from dataclasses import dataclass, asdict
import itertools
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .neural import N, SITE_COORDS, seed_everything

TRIPLES = tuple(itertools.combinations(range(N), 3))


@dataclass
class PPOConfig:
    """Training and budget controls for historical fixed-five-site masked PPO.

    Attributes:
        learning_rate (float): Adam step size.
        entropy (float): Entropy bonus coefficient.
        clip (float): PPO likelihood-ratio clipping width.
        gamma (float): Discount factor; training requires 1.0.
        gae_lambda (float): GAE trace-decay parameter in [0,1].
        epochs (int): Optimization passes per rollout batch.
        batch_episodes (int): Number of synchronized episodes per training batch.
        minibatch (int): Positive number of transitions per optimizer minibatch.
        width (int): Hidden graph embedding dimension.
        reward_scale (float): Positive divisor of best-so-far slice improvements.
        budget (int): At least two table queries per episode, including identity.
    """

    learning_rate: float = 0.0003
    entropy: float = 0.01
    clip: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    epochs: int = 4
    batch_episodes: int = 16
    minibatch: int = 256
    width: int = 64
    reward_scale: float = 20.0
    budget: int = 32


class SearchEnv:
    """Budgeted table-search environment for constructive mappings or triple repair.

    Identity is charged first; only completed candidates consume later queries.
    Geometry and five site coordinates stay fixed. Rewards track best-so-far
    slice improvement, not tile-slice volume.
    """

    def __init__(self, task, mode, budget=32, scale=20.0):
        """Validate mode/budget/scale and initialize the charged identity mapping.

        Args:
            task (Task): Five-qubit task with features, adjacency and complete mapping cost
                table.
            mode (str): Action design: 'constructive' builds a mapping; 'repair' selects and
                reassigns a triple.
            budget (int): Charged evaluation slots, including the baseline and repeated
                requests; not necessarily native compiler calls.
            scale (float): Positive slice-improvement divisor used in the reward formula.

        Raises:
            ValueError: Unknown action mode, budget below two, or nonpositive reward scale.
        """
        if mode not in ("constructive", "repair"):
            raise ValueError("Unknown action design")
        if budget < 2 or scale <= 0:
            raise ValueError("Invalid search budget or reward scale")
        self.task = task
        self.mode = mode
        self.budget = budget
        self.scale = scale
        self.best_mapping = tuple(range(N))
        self.reference = task.costs[self.best_mapping]
        self.best = self.reference
        self.queries = 1
        self.seen = {self.best_mapping}
        self.history = [self.best]
        self.query_mappings = [self.best_mapping]
        self.reward_sum = 0.0
        self.done = False
        self.reset_proposal()

    def reset_proposal(self):
        """Start an empty mapping or enter the triple-selection phase for repair."""
        if self.mode == "constructive":
            self.partial = [-1] * N
            self.pending = list(range(N))
            self.phase = 0
        else:
            self.partial = list(self.best_mapping)
            self.pending = []
            self.phase = 1

    def legal(self):
        """Return the ten-action mask for the current phase; reject terminal observations.

        Returns:
            numpy.ndarray: Boolean mask of shape (10,) for current phase.

        Raises:
            ValueError: Terminal observation.
        """
        if self.done:
            raise ValueError("Terminal observation")
        if self.phase == 1:
            return np.ones(10, dtype=bool)
        return np.array([a not in self.partial if a < N else False for a in range(10)])

    def observation(self):
        """Return node features, adjacency, assignment, progress scalars and legal mask.

        Returns:
            tuple[numpy.ndarray, ...]: Float32 nodes (5,44), adjacency/assignment (5,5), scalars
                (3,) and boolean legal mask (10,).
        """
        partial = np.zeros((N, N), np.float32)
        incumbent = np.zeros((N, N), np.float32)
        focal = np.zeros((N, 1), np.float32)
        pending = np.zeros((N, 1), np.float32)
        for q, site in enumerate(self.partial):
            if site >= 0:
                partial[q, site] = 1
        for q, site in enumerate(self.best_mapping):
            incumbent[q, site] = 1
        if self.pending:
            focal[self.pending[0], 0] = 1
        for q in self.pending:
            pending[q, 0] = 1
        nodes = np.concatenate([self.task.features, partial, incumbent, focal, pending], axis=-1)
        scalars = np.array(
            [
                self.phase,
                (self.queries - 1) / (self.budget - 1),
                (self.reference - self.best) / self.scale,
            ],
            np.float32,
        )
        return nodes, self.task.adjacency, partial, scalars, self.legal()

    def step(self, action):
        """Apply a legal phase-specific action and return scalar reward.

        Intermediate choices yield zero. A complete mapping queries the table
        and earns max(0, previous_best-cost)/scale, then starts another proposal
        unless the budget is exhausted. Illegal/terminal actions raise ValueError.

        Args:
            action (int): Legal index in 0..9; phase determines site assignment or triple
                selection.

        Returns:
            float: Nonnegative scaled incumbent improvement, or zero for incomplete proposals.

        Raises:
            ValueError: Illegal action.
            AssertionError: Invalid permutation.
        """
        if self.done or action not in range(10) or not self.legal()[action]:
            raise ValueError("Illegal action")
        if self.phase == 1:
            self.pending = list(TRIPLES[action])
            self.phase = 0
            for q in self.pending:
                self.partial[q] = -1
            return 0.0
        q = self.pending.pop(0)
        self.partial[q] = action
        if self.pending:
            return 0.0
        candidate = tuple(self.partial)
        if sorted(candidate) != list(range(N)):
            raise AssertionError("Invalid permutation")
        cost = self.task.costs[candidate]
        self.queries += 1
        self.seen.add(candidate)
        self.query_mappings.append(candidate)
        reward = max(0, self.best - cost) / self.scale
        if cost < self.best:
            self.best = cost
            self.best_mapping = candidate
        self.reward_sum += reward
        self.history.append(self.best)
        self.done = self.queries == self.budget
        if not self.done:
            self.reset_proposal()
        return reward

    def result(self, seed):
        """Return seed-tagged query history and final regret; oracle extrema are report-only.

        Args:
            seed (int): Seed for reproducible sampling; global RNG effects, if any, are
                described above.

        Returns:
            dict[str, Any]: Seed-tagged best mapping, regret, query counts, reward sum and
                history.
        """
        optimum = min(self.task.costs.values())
        spread = max(self.task.costs.values()) - optimum
        return {
            "circuit": self.task.name,
            "seed": seed,
            "best": self.best,
            "reference": self.reference,
            "optimum": optimum,
            "regret": self.best - optimum,
            "normalized_regret": (self.best - optimum) / max(1, spread),
            "unique_queries": len(self.seen),
            "table_queries": self.queries,
            "best_mapping": list(self.best_mapping),
            "history": self.history,
            "reward_sum": self.reward_sum,
            "query_mappings": [list(x) for x in self.query_mappings],
        }


def observations(envs):
    """Stack nonterminal environment observations into a tuple of five CPU tensors.

    Args:
        envs (Sequence[SearchEnv]): Nonempty collection of nonterminal environments to encode.

    Returns:
        tuple[torch.Tensor, ...]: Five CPU tensors with a leading batch dimension.
    """
    return tuple(torch.from_numpy(np.stack(a)) for a in zip(*(e.observation() for e in envs)))


class PPOPolicy(nn.Module):
    """Shared graph actor/critic with separate fixed-site assignment and triple-repair heads."""

    def __init__(self, width=64):
        """Build width-sized graph layers, fixed coordinates and ten triple-action indices.

        Args:
            width (int): Positive hidden-feature dimension used by the neural layers.
        """
        super().__init__()
        self.width = width
        self.input = nn.Linear(44, width)
        self.own = nn.ModuleList(nn.Linear(width, width) for _ in range(2))
        self.neighbor = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.register_buffer("coordinates", torch.tensor(SITE_COORDS, dtype=torch.float32) / 8)
        self.register_buffer("triples", torch.tensor(TRIPLES))
        self.site = nn.Linear(width + 3, width)
        self.actor = nn.Sequential(nn.Linear(3 * width + 3, width), nn.Tanh(), nn.Linear(width, 1))
        self.destroy = nn.Sequential(
            nn.Linear(3 * width + 3, width), nn.Tanh(), nn.Linear(width, 1)
        )
        self.critic = nn.Sequential(nn.Linear(2 * width + 3, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, obs):
        """Return phase-selected, legality-masked logits (B,10) and critic values (B,).

        Args:
            obs (tuple[torch.Tensor, ...]): Batched nodes (B,5,44), adjacency/assignment
                (B,5,5), scalars (B,3), and legal bool mask (B,10), as returned by observations.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Masked phase-specific logits (B,10) and values
                (B,).
        """
        nodes, adj, assignment, scalars, legal = obs
        h = F.relu(self.input(nodes))
        for own, neighbor in zip(self.own, self.neighbor):
            h = F.relu(own(h) + neighbor(torch.bmm(adj, h)))
        focal = (h * nodes[:, :, -2:-1]).sum(1)
        context = torch.cat([h.mean(1), focal, scalars], -1)
        occupants = torch.bmm(assignment.transpose(1, 2), h)
        occupied = assignment.sum(1)[:, :, None]
        sites = F.relu(
            self.site(torch.cat([occupants, self.coordinates.expand(len(h), -1, -1), occupied], -1))
        )
        site_logits = self.actor(
            torch.cat([context[:, None, :].expand(-1, N, -1), sites], -1)
        ).squeeze(-1)
        site_logits = F.pad(site_logits, (0, 5), value=-1e9)
        triples = h[:, self.triples].mean(2)
        destroy_logits = self.destroy(
            torch.cat([context[:, None, :].expand(-1, 10, -1), triples], -1)
        ).squeeze(-1)
        logits = torch.where(scalars[:, 0:1] > 0.5, destroy_logits, site_logits).masked_fill(
            ~legal, -1e9
        )
        return logits, self.critic(context).squeeze(-1)


def gae(rewards, values, gamma=1.0, lam=0.95):
    """Complete episodes [time,batch], zero terminal bootstrap.

    Args:
        rewards (torch.Tensor): Floating complete-episode rewards with shape (time, batch).
        values (torch.Tensor): Critic values with the same shape/device/dtype as rewards;
            terminal bootstrap is zero.
        gamma (float): Reward discount factor; these final-best training routines require 1.0.
        lam (float): GAE trace-decay parameter in [0, 1].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Advantages and value targets with shape (time,batch).
    """
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


def clipped_loss(logps, old_logps, advantages, clip):
    """Return the negative mean PPO clipped surrogate from aligned log-probs and advantages.

    Args:
        logps (torch.Tensor): Current-policy action log-probabilities for the sampled
            transitions.
        old_logps (torch.Tensor): Fixed behavior-policy log-probabilities aligned with logps.
        advantages (torch.Tensor): Floating advantage estimates aligned with the log-probability
            samples.
        clip (float): Nonnegative PPO probability-ratio clipping width.

    Returns:
        torch.Tensor: Scalar negative clipped policy surrogate loss.
    """
    ratio = (logps - old_logps).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()


@torch.no_grad()
def collect(model, tasks, mode, config):
    """Sample synchronized complete episodes without gradients and compute GAE targets.

    Returns flattened observations/actions/old log-probs/advantages/returns plus
    final environments. Task batches must share the mode and episode horizon.

    Args:
        model (PPOPolicy): Mapping policy used to sample rollout actions without gradients.
        tasks (Sequence[Task]): Historical five-qubit tasks with feature arrays and complete
            permutation-to-slices tables.
        mode (str): Action design: 'constructive' builds a mapping; 'repair' selects and
            reassigns a triple.
        config (PPOConfig): Masked mapping PPO controls, including episode budget and reward
            scale.

    Returns:
        tuple: Flattened observations (five tensors), actions, old log-probabilities,
            advantages, returns and final SearchEnv list, in that order.

    Raises:
        AssertionError: Unsynchronized episodes.
    """
    envs = [SearchEnv(t, mode, config.budget, config.reward_scale) for t in tasks]
    obs_steps = []
    actions = []
    logps = []
    values = []
    rewards = []
    while not envs[0].done:
        obs = observations(envs)
        logits, value = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        obs_steps.append(obs)
        actions.append(a)
        logps.append(dist.log_prob(a))
        values.append(value)
        rewards.append(
            torch.tensor([env.step(int(act)) for env, act in zip(envs, a)], dtype=torch.float32)
        )
    if not all(e.done for e in envs):
        raise AssertionError("Unsynchronized episodes")
    obs = tuple(torch.cat([step[i] for step in obs_steps]) for i in range(5))
    values = torch.stack(values)
    advantages, returns = gae(torch.stack(rewards), values, config.gamma, config.gae_lambda)
    return obs, torch.cat(actions), torch.cat(logps), advantages.flatten(), returns.flatten(), envs


def learn(tasks, mode, config, seed, episodes, callback=None):
    """Train masked PPO on mapping tables and return the model and training log.

    Requires complete rollout batches and gamma=1 for the final-best objective.
    Calls callback(progress) once per batch, seeds global RNGs and invokes no
    native compiler. Training cost is reported as table queries.

    Args:
        tasks (Sequence[Task]): Historical five-qubit tasks with feature arrays and complete
            permutation-to-slices tables.
        mode (str): Action design: 'constructive' builds a mapping; 'repair' selects and
            reassigns a triple.
        config (PPOConfig): Masked mapping PPO controls, including episode budget and reward
            scale.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.
        episodes (int): Number of training episodes; must form complete batches where required.
        callback (Callable[[dict[str, Any]], None] | None): Optional callback receiving one
            training progress record per batch.

    Returns:
        tuple[PPOPolicy, dict[str, Any]]: Eval-mode policy plus learning curve and table-query
            accounting.

    Raises:
        ValueError: Episodes do not divide into full batches or gamma differs from one.
        FloatingPointError: Nonfinite PPO loss.
    """
    if episodes % config.batch_episodes:
        raise ValueError("Episodes must divide into complete batches")
    if config.gamma != 1:
        raise ValueError("Final-best objective requires gamma=1")
    seed_everything(seed)
    rng = random.Random(seed)
    model = PPOPolicy(config.width)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    curve = []
    started = time.monotonic()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    for batch in range(episodes // config.batch_episodes):
        obs, actions, old_logps, adv, returns, envs = collect(
            model, [rng.choice(tasks) for _ in range(config.batch_episodes)], mode, config
        )
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        losses = []
        kls = []
        clipfracs = []
        entropies = []
        value_losses = []
        for epoch in range(config.epochs):
            for ids in torch.randperm(len(actions)).split(config.minibatch):
                logits, value = model(tuple(o[ids] for o in obs))
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[ids])
                entropy = dist.entropy().mean()
                policy_loss = clipped_loss(logp, old_logps[ids], adv[ids], config.clip)
                value_loss = F.mse_loss(value, returns[ids])
                loss = policy_loss + 0.5 * value_loss - config.entropy * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                with torch.no_grad():
                    logratio = logp - old_logps[ids]
                    ratio = logratio.exp()
                    kls.append(float(((ratio - 1) - logratio).mean()))
                    clipfracs.append(float(((ratio - 1).abs() > config.clip).float().mean()))
                losses.append(float(loss.detach()))
                entropies.append(float(entropy.detach()))
                value_losses.append(float(value_loss.detach()))
        point = {
            "episode": (batch + 1) * config.batch_episodes,
            "mean_reward": float(np.mean([e.reward_sum for e in envs])),
            "loss": float(np.mean(losses)),
            "approx_kl": float(np.mean(kls)),
            "clip_fraction": float(np.mean(clipfracs)),
            "entropy": float(np.mean(entropies)),
            "value_loss": float(np.mean(value_losses)),
        }
        curve.append(point)
        if callback:
            callback(point)
    changed = (
        sum(
            float((v - initial[k]).square().sum())
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        )
        ** 0.5
    )
    model.eval()
    return model, {
        "curve": curve,
        "episodes": episodes,
        "table_queries": episodes * config.budget,
        "wall_seconds": time.monotonic() - started,
        "parameter_l2_change": changed,
        "config": asdict(config),
    }


@torch.no_grad()
def evaluate(model, tasks, seeds, mode, config=None):
    """Evaluate every task/seed using sampled policy actions or uniform legal actions if None.

    Args:
        model (PPOPolicy | None): Mapping policy, or None for uniform sampling over legal
            actions.
        tasks (Sequence[Task]): Historical five-qubit tasks with feature arrays and complete
            permutation-to-slices tables.
        seeds (Sequence[int]): Evaluation seeds; each task is evaluated separately for each
            seed.
        mode (str): Action design: 'constructive' builds a mapping; 'repair' selects and
            reassigns a triple.
        config (PPOConfig | None): Masked mapping PPO controls; evaluate uses defaults when
            None.

    Returns:
        list[dict[str, Any]]: Best mapping and reporting metrics for every task/seed pair.
    """
    config = config or PPOConfig()
    pairs = [(t, s) for t in tasks for s in seeds]
    envs = [SearchEnv(t, mode, config.budget, config.reward_scale) for t, s in pairs]
    rngs = [random.Random(s) for t, s in pairs]
    while not envs[0].done:
        if model is None:
            actions = [rng.choice(np.flatnonzero(e.legal()).tolist()) for rng, e in zip(rngs, envs)]
        else:
            logits, _ = model(observations(envs))
            probabilities = logits.softmax(-1).tolist()
            actions = [rng.choices(range(10), weights=p)[0] for rng, p in zip(rngs, probabilities)]
        for env, a in zip(envs, actions):
            env.step(a)
    return [e.result(s) for e, (t, s) in zip(envs, pairs)]
