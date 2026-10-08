"""Small circuit-conditioned constructive RL models (optional torch dependency).

This pilot is fixed to five EDPC sites. Circuit observations summarize gates and
interactions; they are not a full instruction-DAG encoder. No oracle minima enter
model inputs or training rewards.
"""

from collections import deque
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N = 5
GATES = ("h", "s", "sdg", "t", "tdg", "x", "cx")
SITE_COORDS = ((2, 2), (2, 4), (2, 6), (4, 2), (4, 4))


@dataclass
class Task:
    """Five-qubit feature arrays and a complete permutation-to-slices lookup table.

    features has shape (5,32), adjacency (5,5); cost keys map logical qubits to
    fixed site indices. This historical mapping task does not optimize geometry.

    Attributes:
        name (str): Circuit identifier.
        features (numpy.ndarray): Float32 per-wire feature matrix (5,32).
        adjacency (numpy.ndarray): Float32 normalized interaction matrix (5,5).
        costs (dict[tuple[int, ...], int]): All 120 permutations mapped to successful slice
            counts.
        source_hash (str): SHA-256 fingerprint of source QASM.
    """

    name: str
    features: np.ndarray
    adjacency: np.ndarray
    costs: dict
    source_hash: str

    @property
    def reference(self):
        """Return the slice cost of the identity permutation from the lookup table.

        Returns:
            int: Identity-mapping slice count.
        """
        return self.costs[tuple(range(N))]


def features(gates):
    """Temporal quartiles, per-wire dependency depth, and CNOT interaction graph.

    Args:
        gates (Sequence[Sequence[str | int]]): Temporal gate sequence; each element is a gate
            name followed by zero-based wire indices.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Float32 features (5,32) and row-normalized
            adjacency (5,5).

    Raises:
        ValueError: The input violates the constraints described above.
    """
    x = np.zeros((N, len(GATES) * 4 + 4), np.float32)
    adj = np.zeros((N, N), np.float32)
    depth = np.zeros(N)
    first = np.full(N, len(gates))
    last = np.zeros(N)
    for t, g in enumerate(gates):
        name = g[0]
        qs = g[1:]
        if name == "z":
            name = "s"  # two-S-equivalent feature category, gates remain unchanged
        if name not in GATES:
            raise ValueError(f"Unsupported feature gate {name}")
        idx = min(3, 4 * t // max(1, len(gates))) * len(GATES) + GATES.index(name)
        level = 1 + max(depth[q] for q in qs)
        for q in qs:
            x[q, idx] += 1
            depth[q] = level
            first[q] = min(first[q], t)
            last[q] = t
        if name == "cx":
            a, b = qs
            adj[a, b] += 1
            adj[b, a] += 1
    # Fixed transforms, not fit on test data.
    x[:, :-4] = np.log1p(x[:, :-4]) / math.log1p(max(2, len(gates)))
    x[:, -4] = depth / max(1, depth.max())
    x[:, -3] = first / max(1, len(gates))
    x[:, -2] = last / max(1, len(gates))
    x[:, -1] = math.log1p(len(gates)) / 15
    adj += np.eye(N)
    adj /= adj.sum(axis=1, keepdims=True)
    return x, adj


def load_tasks(root, split):
    """Load one manifest split, verify QASM hashes and require 120 successful mappings.

    Returns Task objects; raises ValueError for hash/table/split errors and
    propagates file errors. Historical oracle files may have been deleted.

    Args:
        root (str | os.PathLike[str]): Filesystem root containing the inputs or evidence
            described above.
        split (str): Exact manifest split name such as train, validation or test.

    Returns:
        list[Task]: Verified tasks for the requested split.

    Raises:
        ValueError: QASM checksum mismatch, invalid JSON/gate features, incomplete or
            unsuccessful oracle, duplicate mappings, or empty requested split.
        OSError: Required manifest, circuit, gate or oracle files cannot be read.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = []
    for entry in manifest["entries"]:
        if entry["split"] != split:
            continue
        qasm = (root / entry["qasm"]).read_bytes()
        sha = hashlib.sha256(qasm).hexdigest()
        if sha != entry["sha256"]:
            raise ValueError("Circuit checksum mismatch")
        gates = json.loads((root / entry["gates"]).read_text())
        x, a = features(gates)
        rows = json.loads((root / entry["name"] / "oracle.json").read_text())
        if len(rows) != 120 or any(r["status"] != "success" for r in rows):
            raise ValueError("Requires complete valid oracle")
        costs = {tuple(r["mapping"]): r["slices"] for r in rows}
        if len(costs) != 120:
            raise ValueError("Duplicate oracle mappings")
        tasks.append(Task(entry["name"], x, a, costs, sha))
    if not tasks:
        raise ValueError(f"No tasks in split {split}")
    return tasks


def load_transfer(root):
    """Load historical structured-five-qubit features and oracle tables under root.

    Args:
        root (str | os.PathLike[str]): Filesystem root containing the inputs or evidence
            described above.

    Returns:
        list[Task]: Historical structured-circuit tasks and mapping costs.

    Raises:
        OSError: Historical input or oracle files are missing/unreadable.
        ValueError: Input JSON or gate features are invalid.
    """
    root = Path(root)
    entries = json.loads((root / "validation/structured5/oracle.json").read_text())
    tasks = []
    for entry in entries:
        path = root / "examples/structured5" / entry["circuit"]
        x, a = features(json.loads((path / "gates.json").read_text()))
        tasks.append(
            Task(
                entry["circuit"],
                x,
                a,
                {tuple(r["mapping"]): r["slices"] for r in entry["rows"]},
                hashlib.sha256((path / "circuit.qasm").read_bytes()).hexdigest(),
            )
        )
    return tasks


def batch_observations(tasks, prefixes):
    """Encode partial five-site permutations into CPU tensors and legal-action masks.

    Returns features (B,5,32), adjacency (B,5,5), assignment (B,5,5),
    focal marker (B,5,1), legal mask (B,5) and step indices (B,).
    Raises ValueError for duplicate/out-of-range actions or terminal prefixes.

    Args:
        tasks (Sequence[Task]): Historical five-qubit tasks with feature arrays and complete
            permutation-to-slices tables.
        prefixes (Sequence[Sequence[int]]): Nonterminal logical-to-site assignment prefixes, one
            per task; legal sites are 0 through 4.

    Returns:
        tuple[torch.Tensor, ...]: CPU features (B,5,32), adjacency/assignment (B,5,5), focal
            (B,5,1), bool legal mask (B,5), int64 steps (B,).

    Raises:
        ValueError: A prefix is terminal, contains duplicates, or uses a site outside 0..4.
    """
    assign = np.zeros((len(tasks), N, N), np.float32)
    current = np.zeros((len(tasks), N, 1), np.float32)
    legal = np.ones((len(tasks), N), bool)
    steps = []
    for b, prefix in enumerate(prefixes):
        if len(prefix) >= N:
            raise ValueError("Terminal state must not be bootstrapped")
        if len(set(prefix)) != len(prefix) or any(a not in range(N) for a in prefix):
            raise ValueError("Invalid prefix")
        for q, site in enumerate(prefix):
            assign[b, q, site] = 1
            legal[b, site] = False
        current[b, len(prefix), 0] = 1
        steps.append(len(prefix))
    return (
        torch.tensor(np.stack([t.features for t in tasks])),
        torch.tensor(np.stack([t.adjacency for t in tasks])),
        torch.tensor(assign),
        torch.tensor(current),
        torch.tensor(legal),
        torch.tensor(steps),
    )


class Encoder(nn.Module):
    """Two-layer interaction-graph encoder for the five fixed EDPC placement sites."""

    def __init__(self, width):
        """Build width-dimensional node/site embeddings with fixed coordinate buffers.

        Args:
            width (int): Positive hidden-feature dimension used by the neural layers.
        """
        super().__init__()
        self.width = width
        self.input = nn.Linear(len(GATES) * 4 + 4 + N + 1, width)
        self.self_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.neighbor_layers = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(2)]
        )
        coords = torch.tensor(SITE_COORDS, dtype=torch.float32) / 8
        self.register_buffer("coordinates", coords)
        self.site = nn.Linear(width + 3, width)

    def forward(self, obs):
        """Return batched context, candidate embeddings and the unchanged legal-site mask.

        Args:
            obs (tuple[torch.Tensor, ...]): Six tensors from batch_observations: features
                (B,5,32), adjacency (B,5,5), assignment (B,5,5), focal (B,5,1), legal bool mask
                (B,5), and int64 steps (B,).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Context (B,2*width+1), candidates
                (B,5,3*width+1) and legal mask (B,5).
        """
        x, adj, assignment, current, legal, steps = obs
        h = F.relu(self.input(torch.cat([x, assignment, current], dim=-1)))
        for own, neighbor in zip(self.self_layers, self.neighbor_layers):
            h = F.relu(own(h) + neighbor(torch.bmm(adj, h)))
        focal = h[torch.arange(len(h)), steps]
        context = torch.cat([h.mean(dim=1), focal, steps[:, None].float() / N], dim=-1)
        occupants = torch.bmm(assignment.transpose(1, 2), h)
        sites = F.relu(
            self.site(
                torch.cat(
                    [
                        occupants,
                        self.coordinates.expand(len(h), -1, -1),
                        (~legal)[:, :, None].float(),
                    ],
                    dim=-1,
                )
            )
        )
        candidates = torch.cat([context[:, None, :].expand(-1, N, -1), sites], dim=-1)
        return context, candidates, legal


class DuelingDQN(nn.Module):
    """Shared graph encoder with bootstrapped value/advantage heads for fixed-site mapping."""

    def __init__(self, width=64, heads=5):
        """Create the requested number of dueling heads with a shared width-sized encoder.

        Args:
            width (int): Positive hidden-feature dimension used by the neural layers.
            heads (int): Positive number of bootstrapped dueling heads.
        """
        super().__init__()
        self.encoder = Encoder(width)
        self.heads = heads
        self.values = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(2 * width + 1, width), nn.ReLU(), nn.Linear(width, 1))
                for _ in range(heads)
            ]
        )
        self.advantages = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(3 * width + 1, width), nn.ReLU(), nn.Linear(width, 1))
                for _ in range(heads)
            ]
        )

    def forward(self, obs):
        """Return Q values of shape (batch, heads, sites).

        Advantages are centered over legal sites only. Returned illegal-site
        values are not masked; callers must mask before selecting actions.

        Args:
            obs (tuple[torch.Tensor, ...]): Six tensors from batch_observations: features
                (B,5,32), adjacency (B,5,5), assignment (B,5,5), focal (B,5,1), legal bool mask
                (B,5), and int64 steps (B,).

        Returns:
            torch.Tensor: Q values (B,heads,5); illegal actions are not masked in this output.
        """
        context, candidates, legal = self.encoder(obs)
        qs = []
        for v, a in zip(self.values, self.advantages):
            advantage = a(candidates).squeeze(-1)
            mean = (advantage * legal).sum(-1, keepdim=True) / legal.sum(-1, keepdim=True)
            qs.append(v(context) + advantage - mean)
        return torch.stack(qs, dim=1)  # [batch,head,site]; masks applied before choosing actions


class ActorCritic(nn.Module):
    """Categorical fixed-site placement actor with a shared graph encoder and value head."""

    def __init__(self, width=64):
        """Build the shared encoder, candidate actor and scalar critic at the given width.

        Args:
            width (int): Positive hidden-feature dimension used by the neural layers.
        """
        super().__init__()
        self.encoder = Encoder(width)
        self.actor = nn.Sequential(nn.Linear(3 * width + 1, width), nn.ReLU(), nn.Linear(width, 1))
        self.critic = nn.Sequential(nn.Linear(2 * width + 1, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, obs):
        """Return masked site logits (B,5) and state values (B,) from encoded observations.

        Args:
            obs (tuple[torch.Tensor, ...]): Six tensors from batch_observations: features
                (B,5,32), adjacency (B,5,5), assignment (B,5,5), focal (B,5,1), legal bool mask
                (B,5), and int64 steps (B,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Masked logits (B,5) and values (B,).
        """
        context, candidates, legal = self.encoder(obs)
        return self.actor(candidates).squeeze(-1).masked_fill(~legal, -1e9), self.critic(
            context
        ).squeeze(-1)


@dataclass
class Config:
    """Hyperparameters for historical five-site DQN/actor-critic table training.

    scale normalizes terminal slice improvement; training requires gamma=1.
    DQN uses replay, bootstrap masks and periodic target-network updates.
    These settings neither select geometry nor enable local compiler execution.

    Attributes:
        algorithm (str): Use 'dqn' for DQN; other values follow the actor-critic branch.
        width (int): Positive hidden embedding dimension.
        heads (int): Positive number of DQN bootstrap heads.
        learning_rate (float): Adam step size.
        scale (float): Positive terminal slice-improvement divisor.
        batch_size (int): DQN replay batch size.
        target_interval (int): Positive number of optimizer updates between target copies.
        updates_per_episode (int): DQN replay updates per episode once the replay buffer is
            large enough.
        epsilon_end (float): Final DQN exploration probability in [0,1].
        bootstrap_probability (float): Probability in [0,1] of enabling each bootstrap head for
            an episode.
        entropy (float): Actor-critic entropy coefficient.
        gamma (float): Discount factor; training requires 1.0.
    """

    algorithm: str = "dqn"
    width: int = 64
    heads: int = 5
    learning_rate: float = 0.001
    scale: float = 50.0
    batch_size: int = 64
    target_interval: int = 50
    updates_per_episode: int = 2
    epsilon_end: float = 0.15
    bootstrap_probability: float = 0.8
    entropy: float = 0.03
    gamma: float = 1.0


def seed_everything(seed):
    """Seed Python, NumPy and Torch and enable deterministic single-thread Torch.

    Mutates process-global RNG and Torch settings; this is not an isolated RNG.

    Args:
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def double_targets(online, target, tasks, prefixes, rewards, done, gamma=1.0):
    """No terminal network evaluation; independent online selection and target value.

    Args:
        online (DuelingDQN): Online network used to select legal maximizing actions.
        target (DuelingDQN): Target network with matching head count used to value selected
            actions.
        tasks (Sequence[Task]): Historical five-qubit tasks with feature arrays and complete
            permutation-to-slices tables.
        prefixes (Sequence[Sequence[int]]): Nonterminal logical-to-site assignment prefixes, one
            per task; legal sites are 0 through 4.
        rewards (Sequence[float]): Immediate rewards, one per transition.
        done (Sequence[bool]): Terminal flags; terminal entries are not passed to the networks.
        gamma (float): Reward discount factor; these final-best training routines require 1.0.

    Returns:
        torch.Tensor: Float32 Bellman targets (transitions, heads) with zero terminal bootstrap.
    """
    y = torch.tensor(rewards, dtype=torch.float32)[:, None].repeat(1, online.heads)
    indices = [i for i, d in enumerate(done) if not d]
    if indices:
        obs = batch_observations([tasks[i] for i in indices], [prefixes[i] for i in indices])
        with torch.no_grad():
            best = online(obs).masked_fill(~obs[4][:, None, :], -1e9).argmax(-1)
            values = target(obs).gather(-1, best[:, :, None]).squeeze(-1)
            y[indices] += gamma * values
    return y


@torch.no_grad()
def propose(model, task, rng, epsilon=0.0):
    """Return one complete five-site mapping without a cost query.

    DQN chooses one head per mapping and uses epsilon-greedy legal actions.
    Actor-critic samples its categorical logits; epsilon applies only to DQN.

    Args:
        model (DuelingDQN | ActorCritic): Placement policy; sampling runs without gradients.
        task (Task): Five-qubit feature/cost task used to condition placement.
        rng (random.Random): Caller-owned pseudorandom generator; sampling advances its state.
        epsilon (float): Legal-action exploration probability in [0, 1]; only DQN uses this
            parameter.

    Returns:
        tuple[int, ...]: Complete five-element logical-to-site permutation.
    """
    prefix = []
    head = rng.randrange(model.heads) if isinstance(model, DuelingDQN) else None
    for _ in range(N):
        obs = batch_observations([task], [prefix])
        legal = [a for a in range(N) if a not in prefix]
        if isinstance(model, DuelingDQN):
            q = model(obs)[0, head].masked_fill(~obs[4][0], -1e9)
            action = rng.choice(legal) if rng.random() < epsilon else int(q.argmax())
        else:
            logits, _ = model(obs)
            p = logits[0].softmax(-1).tolist()
            action = rng.choices(range(N), weights=p)[0]
        prefix.append(action)
    return tuple(prefix)


def train(tasks, config, seed, episodes=600, callback=None):
    """Train a mapping model against precomputed tables and return model plus logs.

    Terminal reward is (identity_slices-candidate_slices)/config.scale, with
    zero intermediate rewards and gamma=1. No native compiler is invoked.
    The callback receives (model, progress) every 50 episodes and at completion.
    Uses process-global seeding; rejects nonunit gamma or nonpositive scale.

    Args:
        tasks (Sequence[Task]): Historical five-qubit tasks with feature arrays and complete
            permutation-to-slices tables.
        config (Config): Training hyperparameters; gamma must equal 1 and scale must be
            positive.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.
        episodes (int): Number of training episodes; must form complete batches where required.
        callback (Callable[[torch.nn.Module, dict[str, Any]], None] | None): Optional progress
            callback receiving the model and log record.

    Returns:
        tuple[torch.nn.Module, dict[str, Any]]: Trained DuelingDQN/ActorCritic and
            training/query accounting log.

    Raises:
        ValueError: This experiment fixes gamma=1 and positive scale.
    """
    if config.gamma != 1 or config.scale <= 0:
        raise ValueError("This experiment fixes gamma=1 and positive scale")
    seed_everything(seed)
    rng = random.Random(seed)
    replay = deque(maxlen=10000)
    model = (
        DuelingDQN(config.width, config.heads)
        if config.algorithm == "dqn"
        else ActorCritic(config.width)
    )
    target = copy.deepcopy(model) if config.algorithm == "dqn" else None
    if target is not None:
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    seen = set()
    curve = []
    updates = 0
    started = time.monotonic()
    last_loss = 0
    for episode in range(1, episodes + 1):
        task = rng.choice(tasks)
        prefix = []
        trajectory = []
        head = rng.randrange(config.heads) if config.algorithm == "dqn" else None
        epsilon = config.epsilon_end + (1 - config.epsilon_end) * max(
            0, 1 - episode / (episodes * 0.6)
        )
        if config.algorithm == "dqn":
            mask = np.array(
                [rng.random() < config.bootstrap_probability for _ in range(config.heads)],
                np.float32,
            )
            if not mask.any():
                mask[head] = 1
        for _ in range(N):
            obs = batch_observations([task], [prefix])
            legal = [a for a in range(N) if a not in prefix]
            if config.algorithm == "dqn":
                with torch.no_grad():
                    q = model(obs)[0, head].masked_fill(~obs[4][0], -1e9)
                action = rng.choice(legal) if rng.random() < epsilon else int(q.argmax())
            else:
                logits, value = model(obs)
                dist = torch.distributions.Categorical(logits=logits[0])
                action = int(dist.sample())
            old = tuple(prefix)
            prefix.append(action)
            if config.algorithm == "dqn":
                trajectory.append((task, old, action, tuple(prefix)))
            else:
                trajectory.append((dist.log_prob(torch.tensor(action)), value[0], dist.entropy()))
        cost = task.costs[tuple(prefix)]
        reward = (task.reference - cost) / config.scale
        seen.add((task.name, tuple(prefix)))
        if config.algorithm == "dqn":
            for t, (task, old, action, new) in enumerate(trajectory):
                replay.append(
                    (task, old, action, new, reward if t == N - 1 else 0.0, t == N - 1, mask.copy())
                )
            if len(replay) >= config.batch_size:
                for _ in range(config.updates_per_episode):
                    batch = rng.sample(list(replay), config.batch_size)
                    bt, old, actions, new, rewards, done, masks = zip(*batch)
                    obs = batch_observations(bt, old)
                    predicted = (
                        model(obs)
                        .gather(
                            -1, torch.tensor(actions)[:, None, None].expand(-1, config.heads, 1)
                        )
                        .squeeze(-1)
                    )
                    y = double_targets(model, target, bt, new, rewards, done)
                    m = torch.tensor(np.stack(masks))
                    losses = F.smooth_l1_loss(predicted, y, reduction="none")
                    loss = (losses * m).sum() / m.sum().clamp_min(1)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    updates += 1
                    last_loss = float(loss.detach())
                    if updates % config.target_interval == 0:
                        target.load_state_dict(model.state_dict())
        else:
            lp, val, ent = zip(*trajectory)
            lp = torch.stack(lp)
            val = torch.stack(val)
            ent = torch.stack(ent)
            advantage = reward - val
            loss = (
                -(lp * advantage.detach()).mean()
                + 0.5 * (advantage**2).mean()
                - config.entropy * ent.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            updates += 1
            last_loss = float(loss.detach())
        if episode % 50 == 0 or episode == episodes:
            point = {
                "episode": episode,
                "loss": last_loss,
                "reward": reward,
                "unique_queries": len(seen),
                "table_queries": episode + len(tasks),
                "duplicate_fraction": 1 - len(seen) / episode,
                "gradient_updates": updates,
                "wall_seconds": time.monotonic() - started,
            }
            curve.append(point)
            if callback:
                callback(model, point)
    return model, {
        "config": asdict(config),
        "seed": seed,
        "episodes": episodes,
        "curve": curve,
        "table_queries": episodes + len(tasks),
        "reference_queries": len(tasks),
        "unique_candidate_queries": len(seen),
        "gradient_updates": updates,
        "wall_seconds": time.monotonic() - started,
        "native_compiler_calls": 0,
    }


def evaluate(model, tasks, seeds, epsilon=0.15, budget=32):
    """Return per-task/seed best-of-budget mapping results from lookup tables.

    Identity counts as the first query, including repeated candidate requests.
    Oracle extrema enter final regret reporting only, not action selection.
    This performs no native calls and does not generate new grid geometry.

    Args:
        model (DuelingDQN | ActorCritic): Trained five-site placement network; evaluation does
            not update its weights.
        tasks (Sequence[Task]): Historical five-qubit tasks with feature arrays and complete
            permutation-to-slices tables.
        seeds (Sequence[int]): Evaluation seeds; each task is evaluated separately for each
            seed.
        epsilon (float): Legal-action exploration probability in [0, 1]; only DQN uses this
            parameter.
        budget (int): Charged evaluation slots, including the baseline and repeated requests;
            not necessarily native compiler calls.

    Returns:
        list[dict[str, Any]]: Per-task/seed best mapping, slices, regret, history and
            table-query accounting.
    """
    rows = []
    for task in tasks:
        for seed in seeds:
            rng = random.Random(seed)
            best = task.reference
            mapping = tuple(range(N))
            seen = {mapping}
            history = [best]
            for _ in range(budget - 1):
                candidate = propose(model, task, rng, epsilon)
                score = task.costs[candidate]
                seen.add(candidate)
                if score < best:
                    best = score
                    mapping = candidate
                history.append(best)
            optimum = min(task.costs.values())
            spread = max(task.costs.values()) - optimum
            rows.append(
                {
                    "circuit": task.name,
                    "seed": seed,
                    "best": best,
                    "reference": task.reference,
                    "optimum": optimum,
                    "regret": best - optimum,
                    "normalized_regret": (best - optimum) / max(1, spread),
                    "unique_queries": len(seen),
                    "table_queries": budget,
                    "best_mapping": list(mapping),
                    "history": history,
                }
            )
    return rows


def metrics(rows):
    """Aggregate regret, optimum-hit rate, unique-query counts and best slices across rows.

    Args:
        rows (Sequence[Mapping[str, int | float]]): Nonempty evaluation records containing
            regret, normalized_regret, unique_queries and best.

    Returns:
        dict[str, float]: Mean regret, normalized regret, optimum-hit rate, unique queries and
            best slices.
    """
    return {
        "mean_regret": float(np.mean([r["regret"] for r in rows])),
        "mean_normalized_regret": float(np.mean([r["normalized_regret"] for r in rows])),
        "optimum_hit_rate": float(np.mean([r["regret"] == 0 for r in rows])),
        "mean_unique_queries": float(np.mean([r["unique_queries"] for r in rows])),
        "mean_best": float(np.mean([r["best"] for r in rows])),
    }
