"""Variable-size, identity-explicit static geometry and native compiler scoring.

No four-neighbor rule, fixed shape, row-rank rule or single-patch-only rule.
Retained: interior Q positions, connected routing, resource counts/types/timing.
Resizing inserts/deletes routing-only rows/columns, translating resource positions.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import re
import threading
import numpy as np
from .layout import Layout

MOVE_SET_VERSION = "geometry-v3-variable-identity"
PROPOSAL_KINDS = ("single", "compound", "insert_row", "insert_col", "delete_row", "delete_col")


def geometry_scope(objective="area_times_slices"):
    """Return serializable search assumptions, move distribution and objective label.

    Args:
        objective (str): Objective label recorded in the returned scope metadata.

    Returns:
        dict[str, Any]: Serializable move-set, constraint, distribution and objective metadata.
    """
    return {
        "version": MOVE_SET_VERSION,
        "shape": "variable; no configured maximum",
        "identity": "explicit logical-qubit coordinates; unrestricted initial assignment",
        "four_routing_neighbors_required": False,
        "compound_sizes": "2 through data-qubit count",
        "resize": "insert/delete routing-only rows/columns; resource coordinates translate",
        "retained": [
            "interior data patches",
            "connected routing",
            "resource counts/types and compiler timing",
            "static single-cell patches",
        ],
        "proposal_distribution": "uniform over six kinds; uniform k for compound; uniform unique endpoints for other kinds",
        "invalid_proposals": "counted self-loops; no hidden retry",
        "objective": objective,
    }


@dataclass(frozen=True)
class GridState:
    """Immutable ASCII cells plus logical-qubit coordinates, independent of row order.

    Attributes:
        rows (tuple[str, ...]): Rectangular ASCII tile rows.
        qubits (tuple[tuple[int, int], ...]): Entry i is logical qubit i’s cell, independent of
            row-major compiler order.
    """

    rows: tuple
    qubits: tuple  # index is logical identity, value is (row,column)

    def __post_init__(self):
        """Validate ASCII and require one distinct logical identity for every Q cell.

        Raises:
            ValueError: Explicit identities must cover each Q cell exactly once.
        """
        layout = Layout.parse("\n".join(self.rows))
        if len(set(self.qubits)) != len(self.qubits) or set(self.qubits) != set(layout.sites):
            raise ValueError("Explicit identities must cover each Q cell exactly once")

    @classmethod
    def parse(cls, text, qubits=None):
        """Build a state from ASCII and optional ordered logical coordinates.

        qubits[i] is logical qubit i's zero-based (row, column). If omitted,
        identities follow row-major Q order. Supply explicit identities when
        reloading an optimized layout. Invalid coverage raises ValueError.

        Args:
            text (str): Rectangular ASCII layout containing every logical Q site exactly once.
            qubits (Sequence[Sequence[int]] | None): Ordered (row, column) coordinates, one per
                logical qubit; None uses row-major Q order.

        Returns:
            GridState: ASCII rows with a complete explicit logical identity tuple.
        """
        g = Layout.parse(text)
        return cls(g.rows, tuple(tuple(x) for x in qubits) if qubits is not None else g.sites)

    @property
    def text(self):
        """Return newline-terminated ASCII, without encoding logical identities.

        Returns:
            str: ASCII grid with trailing newline.
        """
        return "\n".join(self.rows) + "\n"

    @property
    def shape(self):
        """Return (height, width) in grid tiles.

        Returns:
            tuple[int, int]: Grid height and width in tiles.
        """
        return len(self.rows), len(self.rows[0])

    @property
    def area(self):
        """Return rectangular grid tile count, including routing and resource cells.

        Returns:
            int: Rectangular tile count including routing and resources.
        """
        return self.shape[0] * self.shape[1]

    @property
    def logical_to_site(self):
        """Map each logical qubit index to its row-major compiler Q-site index.

        Returns:
            tuple[int, ...]: Entry q is the compiler wire carrying logical qubit q.
        """
        ranks = {p: i for i, p in enumerate(sorted(self.qubits))}
        return tuple(ranks[p] for p in self.qubits)

    def identity_record(self):
        """Return JSON-compatible logical coordinates and compiler-wire interpretation.

        Returns:
            dict[str, Any]: logical_qubit_positions, logical_to_compiler_index and
                interpretation.
        """
        return {
            "logical_qubit_positions": [list(x) for x in self.qubits],
            "logical_to_compiler_index": list(self.logical_to_site),
            "interpretation": "compiler wire logical_to_compiler_index[q] carries logical qubit q",
        }


def as_state(grid):
    """Return a GridState unchanged or parse ASCII with row-major initial identities.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.

    Returns:
        GridState: Original state or parsed ASCII with row-major identities.
    """
    return grid if isinstance(grid, GridState) else GridState.parse(grid)


def state_digest(grid):
    """Hash both ASCII geometry and ordered logical identities using SHA-256.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.

    Returns:
        str: SHA-256 digest of ASCII and ordered identities.
    """
    g = as_state(grid)
    return hashlib.sha256((g.text + json.dumps(g.qubits)).encode()).hexdigest()


@lru_cache(maxsize=20000)
def valid_geometry(grid):
    """Check interior Q positions and one nonempty connected routing component.

    Accepts hashable GridState or ASCII text. Does not require four routing
    neighbors, impose an area cap, or guarantee successful native compilation.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.

    Returns:
        bool: Whether Q cells are interior and routing is nonempty and connected.
    """
    g = as_state(grid)
    h, w = g.shape
    if any(not (0 < r < h - 1 and 0 < c < w - 1) for r, c in g.qubits):
        return False
    routes = {(r, c) for r, row in enumerate(g.rows) for c, x in enumerate(row) if x == "r"}
    if not routes:
        return False
    stack = [next(iter(routes))]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if n in routes and n not in seen:
                seen.add(n)
                stack.append(n)
    return seen == routes


def relocated(grid, replacements):
    """Atomic assignment; selected patches may occupy one another's old cells.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        replacements (Mapping[int, tuple[int, int]]): Logical-qubit indices mapped to proposed
            zero-based destination coordinates.

    Returns:
        GridState | None: Changed valid state, or None for invalid/unchanged assignment.
    """
    g = as_state(grid)
    h, w = g.shape
    sites = list(g.qubits)
    selected = set(replacements)
    if not selected or any(q < 0 or q >= len(sites) for q in selected):
        return None
    old = {sites[q] for q in selected}
    for q, cell in replacements.items():
        r, c = cell
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return None
        if g.rows[r][c] != "r" and cell not in old:
            return None
        sites[q] = (r, c)
    if len(set(sites)) != len(sites):
        return None
    rows = [list(row) for row in g.rows]
    for r, c in old:
        rows[r][c] = "r"
    for q in selected:
        r, c = sites[q]
        rows[r][c] = "Q"
    result = GridState(tuple("".join(row) for row in rows), tuple(sites))
    return result if result != g and valid_geometry(result) else None


@lru_cache(maxsize=4000)
def single_candidates(grid):
    """Return cached ((qubit, row, column), state) pairs for valid Q-to-r moves.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.

    Returns:
        tuple[tuple[tuple[int, int, int], GridState], ...]: Pairs of (qubit,row,column) actions
            and valid endpoints.
    """
    g = as_state(grid)
    h, w = g.shape
    out = []
    for q in range(len(g.qubits)):
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if g.rows[r][c] == "r":
                    other = relocated(g, {q: (r, c)})
                    if other is not None:
                        out.append(((q, r, c), other))
    return tuple(out)


def resize_at(grid, kind, index):
    """Insert or delete one routing-only row/column and translate identities.

    kind is insert_row, insert_col, delete_row or delete_col; index is zero-based.
    Return None for invalid indices, occupied deletions or invalid geometry.
    Raise ValueError for an unknown kind. No maximum dimensions are imposed.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        kind (str): Resize operation: insert_row, insert_col, delete_row or delete_col.
        index (int): Zero-based row or column index; insertion permits the end position.

    Returns:
        GridState | None: Resized state, or None if the index/deletion/geometry is invalid.

    Raises:
        ValueError: Unknown resize kind.
    """
    g = as_state(grid)
    h, w = g.shape
    rows = [list(row) for row in g.rows]
    qs = list(g.qubits)
    if kind == "insert_row":
        if not 0 <= index <= h:
            return None
        rows.insert(index, ["r"] * w)
        qs = [(r + (r >= index), c) for r, c in qs]
    elif kind == "insert_col":
        if not 0 <= index <= w:
            return None
        for row in rows:
            row.insert(index, "r")
        qs = [(r, c + (c >= index)) for r, c in qs]
    elif kind == "delete_row":
        if not 0 <= index < h or h <= 1 or any(x != "r" for x in rows[index]):
            return None
        rows.pop(index)
        qs = [(r - (r > index), c) for r, c in qs]
    elif kind == "delete_col":
        if not 0 <= index < w or w <= 1 or any(row[index] != "r" for row in rows):
            return None
        for row in rows:
            row.pop(index)
        qs = [(r, c - (c > index)) for r, c in qs]
    else:
        raise ValueError("Unknown resize kind")
    result = GridState(tuple("".join(row) for row in rows), tuple(qs))
    return result if valid_geometry(result) else None


@lru_cache(maxsize=4000)
def resize_candidates(grid, kind):
    """Return (index, state) resize endpoints, deduplicated by full GridState.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        kind (str): Resize operation: insert_row, insert_col, delete_row or delete_col.

    Returns:
        tuple[tuple[int, GridState], ...]: Representative indices and distinct valid endpoints.
    """
    g = as_state(grid)
    axis = 0 if kind.endswith("row") else 1
    length = g.shape[axis] + kind.startswith("insert")
    unique = {}
    for i in range(length):
        other = resize_at(g, kind, i)
        if other is not None:
            unique.setdefault(other, i)
    # Deduplicate identical endpoints, so proposal probabilities do not hide multiplicities.
    return tuple((i, other) for other, i in unique.items())


@dataclass(frozen=True)
class Proposal:
    """Immutable proposed state, action and acceptance-probability metadata.

    log_reverse_over_forward is log(q(reverse)/q(forward)), not a score delta.
    null_reason marks a self-loop that still consumes the caller's query budget.

    Attributes:
        state (GridState): Proposed endpoint or unchanged state for a null draw.
        kind (str): Selected proposal family.
        action (tuple[str | int, ...]): Concrete action or macro descriptor.
        changed_qubits (tuple[int, ...]): Logical IDs changed by relocation.
        log_reverse_over_forward (float): Log uniform-kernel reverse/forward probability ratio.
        null_reason (str | None): Explanation of a self-loop, or None for a non-null move.
    """

    state: GridState
    kind: str
    action: tuple
    changed_qubits: tuple = ()
    log_reverse_over_forward: float = 0.0
    null_reason: str | None = None

    def metadata(self):
        """Return JSON-compatible action, ratio, null reason and candidate dimensions.

        Returns:
            dict[str, Any]: Action, changed identities, null reason, log ratio and candidate
                shape.
        """
        return {
            "proposal_kind": self.kind,
            "action": list(self.action),
            "changed_qubits": list(self.changed_qubits),
            "null_reason": self.null_reason,
            "log_proposal_ratio": self.log_reverse_over_forward,
            "candidate_shape": list(self.state.shape),
        }


def compound_proposal(grid, k, rng):
    """Randomly reassign exactly k logical identities, including possible swaps.

    Invalid or partially unchanged draws become self-loops without retrying.
    k must be between two and the number of qubits or ValueError is raised.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        k (int): Number of changed logical qubits, from two through the grid data-qubit count.
        rng (random.Random): Caller-owned pseudorandom generator; sampling advances its state.

    Returns:
        Proposal: Sampled endpoint or explicitly labeled self-loop.

    Raises:
        ValueError: Compound size must be between 2 and data count.
    """
    g = as_state(grid)
    n = len(g.qubits)
    h, w = g.shape
    if not 2 <= k <= n:
        raise ValueError("Compound size must be between 2 and data count")
    selected = sorted(rng.sample(range(n), k))
    unselected = {p for q, p in enumerate(g.qubits) if q not in selected}
    pool = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if g.rows[r][c] in "Qr" and (r, c) not in unselected
    ]
    # Ordered sample: every bijection to selected identities is possible, including swaps.
    dest = rng.sample(pool, k)
    replacements = dict(zip(selected, dest))
    action = ("compound", k)
    if any(g.qubits[q] == p for q, p in replacements.items()):
        return Proposal(g, "compound", action, null_reason="not_exactly_k_changed")
    other = relocated(g, replacements)
    if other is None:
        return Proposal(g, "compound", action, null_reason="routing_constraint")
    return Proposal(other, "compound", action, tuple(selected))


def reverse_resize(kind):
    """Exchange the insert/delete prefix of a supported resize action name.

    Args:
        kind (str): Resize operation: insert_row, insert_col, delete_row or delete_col.

    Returns:
        str: Opposite insert/delete action name.
    """
    return (
        kind.replace("insert", "delete")
        if kind.startswith("insert")
        else kind.replace("delete", "insert")
    )


def proposal_actions(grid):
    """Return concrete single moves plus compound-size and resize macros.

    Macro endpoints are sampled later; inclusion does not guarantee a non-null
    proposal or compiler feasibility. This ordering indexes policy logits.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.

    Returns:
        tuple[tuple[str | int, ...], ...]: Ordered single actions and stochastic macro
            descriptors.
    """
    g = as_state(grid)
    return (
        tuple(("single", *a) for a, _ in single_candidates(g))
        + tuple(("compound", k) for k in range(2, len(g.qubits) + 1))
        + tuple((kind,) for kind in PROPOSAL_KINDS[2:])
    )


def propose_action(grid, action, rng):
    """Resolve an action to a Proposal using rng for stochastic macro endpoints.

    Single moves and unique resize endpoints include proposal-degree ratios.
    Invalid single actions raise ValueError; unavailable macros yield self-loops.
    The ratio describes the uniform kernel, not arbitrary learned action odds.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        action (tuple[str | int, ...]): Geometry action from proposal_actions/sample_action,
            including stochastic macro descriptors.
        rng (random.Random): Caller-owned pseudorandom generator; sampling advances its state.

    Returns:
        Proposal: Candidate and uniform-kernel ratio, possibly a self-loop.

    Raises:
        ValueError: Invalid single action.
    """
    g = as_state(grid)
    kind = action[0]
    if kind == "single":
        q, r, c = action[1:]
        other = relocated(g, {q: (r, c)})
        if other is None:
            raise ValueError("Invalid single action")
        ratio = math.log(len(single_candidates(g))) - math.log(len(single_candidates(other)))
        return Proposal(other, kind, action, (q,), ratio)
    if kind == "compound":
        return compound_proposal(g, action[1], rng)
    candidates = resize_candidates(g, kind)
    if not candidates:
        return Proposal(g, kind, action, null_reason="no_resize_endpoint")
    _, other = rng.choice(candidates)
    ratio = math.log(len(candidates)) - math.log(
        len(resize_candidates(other, reverse_resize(kind)))
    )
    return Proposal(other, kind, action, (), ratio)


def sample_action(grid, rng):
    """Sample one of six equal families and its action; unavailable choices are null.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        rng (random.Random): Caller-owned pseudorandom generator; sampling advances its state.

    Returns:
        tuple[str | int, ...]: Action descriptor, possibly null_single or null_compound.
    """
    g = as_state(grid)
    kind = rng.choice(PROPOSAL_KINDS)
    if kind == "single":
        candidates = single_candidates(g)
        return ("single", *rng.choice(candidates)[0]) if candidates else ("null_single",)
    if kind == "compound":
        return (
            (kind, rng.randrange(2, len(g.qubits) + 1)) if len(g.qubits) > 1 else ("null_compound",)
        )
    return (kind,)


def propose(grid, rng):
    """Sample and resolve a geometry proposal without retrying invalid draws.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        rng (random.Random): Caller-owned pseudorandom generator; sampling advances its state.

    Returns:
        Proposal: Sampled endpoint or counted self-loop without retry.
    """
    g = as_state(grid)
    a = sample_action(g, rng)
    if a[0].startswith("null_"):
        return Proposal(g, a[0][5:], a, null_reason="no_action")
    return propose_action(g, a, rng)


def remap_qasm(qasm, grid):
    """Relabel circuit wires for explicit initial placement; does not insert SWAP gates.

    Supports the benchmark's single quantum-register OpenQASM 2 input. Both input
    preparation and output interpretation follow the saved wire map.

    Args:
        qasm (str): Already synthesized, single-register OpenQASM 2 source matching the grid
            data-qubit count.
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.

    Returns:
        str: QASM with logical indices relabeled to compiler wires; no SWAP gates added.

    Raises:
        ValueError: Expected one quantum register matching data count.
    """
    g = as_state(grid)
    source = re.sub(r"//[^\n]*", "", qasm)
    regs = re.findall(r"\bqreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;", source)
    if len(regs) != 1 or int(regs[0][1]) != len(g.qubits):
        raise ValueError("Expected one quantum register matching data count")
    name = regs[0][0]
    mapping = g.logical_to_site
    pattern = re.compile(r"\b" + re.escape(name) + r"\s*\[\s*(\d+)\s*\]")

    def wire(m):
        """Translate one matched quantum-wire reference through the explicit identity map.

        Args:
            m (re.Match[str]): Regex match whose first capture is a quantum-register index.

        Returns:
            str: Remapped register reference.

        Raises:
            ValueError: Quantum register index out of bounds.
        """
        i = int(m.group(1))
        if i >= len(mapping):
            raise ValueError("Quantum register index out of bounds")
        return f"{name}[{mapping[i]}]"

    statements = []
    for statement in source.split(";"):
        if statement.lstrip().startswith(("qreg ", "creg ", "include ", "OPENQASM ")):
            statements.append(statement)
        else:
            statements.append(pattern.sub(wire, statement))
    return ";".join(statements)


class GeometryOracle:
    """Thread-safe per-instance cache of pinned EDPC-preprocessing/wave measurements.

    Cache identity includes the circuit, geometry, logical assignment and backend.
    Independent processes must use separate roots. Failures are cached as failures.
    """

    def __init__(self, compiler, root):
        """Create a cache root, per-key thread locks and zeroed request/call/hit counters.

        Args:
            compiler (Compiler): Configured native compiler adapter; set
                CompilerConfig(local=True) for current local evaluations.
            root (str | os.PathLike[str]): Filesystem root containing the inputs or evidence
                described above.
        """
        self.compiler = compiler
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.locks = {}
        self.requests = 0
        self.native_calls = 0
        self.hits = 0

    def evaluate(self, qasm, grid):
        """Return a cached or newly compiled score for a circuit and explicit grid.

        Cache keys include original/mapped QASM, cells, logical coordinates,
        move-set version, executable hash and all CompilerConfig fields,
        including local. New records use EDPC preprocessing and summary counting.
        Failures are cached with null scores. All requests count, including hits.
        Separate processes must use separate roots; locks are per instance.

        Args:
            qasm (str): Already synthesized, single-register OpenQASM 2 source matching the grid
                data-qubit count.
            grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
                identities in row-major Q order.

        Returns:
            dict[str, Any]: Compiler record plus cache_key, geometry hash, scope and logical
                identity. Failures retain slices=None.

        Raises:
            ValueError: QASM/register identity is invalid or compiler options are incompatible.
            OSError: Cache/evidence I/O fails, including an incomplete preexisting cache
                destination.
        """
        from dataclasses import asdict

        g = as_state(grid)
        mapped = remap_qasm(qasm, g)
        key = hashlib.sha256(
            (
                MOVE_SET_VERSION
                + "\0"
                + qasm
                + "\0"
                + mapped
                + "\0"
                + g.text
                + json.dumps(g.qubits)
                + self.compiler.binary_sha256
                + json.dumps(asdict(self.compiler.config), sort_keys=True)
            ).encode()
        ).hexdigest()
        with self.lock:
            self.requests += 1
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            dest = self.root / key
            record = dest / "geometry-score.json"
            if record.exists():
                with self.lock:
                    self.hits += 1
                return json.loads(record.read_text())
            result = self.compiler.run(
                mapped,
                Layout.parse(g.text),
                dest,
                "qasm",
                retain_slices=False,
                preprocess_layout="edpc",
            )
            with self.lock:
                self.native_calls += 1
            result.update(
                geometry_sha256=state_digest(g),
                cache_key=key,
                geometry_scope=geometry_scope("compiler measurement; objective assigned by search"),
                **g.identity_record(),
            )
            (dest / "original.qasm").write_text(qasm)
            (dest / "identity.json").write_text(json.dumps(g.identity_record(), indent=2))
            record.write_text(json.dumps(result, indent=2))
            return result

    def accounting(self):
        """Return cumulative score requests, new native calls and cache hits.

        Returns:
            dict[str, int]: Cumulative score_requests, new_native_compilations and cache_hits.
        """
        return {
            "score_requests": self.requests,
            "new_native_compilations": self.native_calls,
            "cache_hits": self.hits,
        }


def spatial_observation(grid, features, adjacency, current, baseline, best, progress):
    """Encode geometry and circuit features for the variable-size grid policy.

    Return float32 planes (7,H,W), flattened circuit/context features, ordered
    action tuples and action features (A,10). The final context scalars are
    current/baseline, best/baseline and progress; scores use consistent units.
    Candidate presence is not a promise of successful native compilation.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        features (numpy.ndarray): Float32 per-qubit circuit feature matrix (N, F).
        adjacency (numpy.ndarray): Float32 circuit interaction matrix (N, N), aligned to logical
            identities.
        current (float): Current objective value in the same units as baseline and best.
        baseline (float): Positive initial objective used to normalize observations.
        best (float): Best observed objective value, in baseline units.
        progress (float): Fraction of proposal budget consumed, normally in [0, 1].

    Returns:
        dict[str, Any]: planes (7,H,W), context vector, ordered actions and action_features
            (A,10), all numerical arrays float32.
    """
    g = as_state(grid)
    h, w = g.shape
    planes = np.zeros((7, h, w), np.float32)
    for q, (r, c) in enumerate(g.qubits):
        planes[0, r, c] = 1
        planes[1, r, c] = (q + 1) / len(g.qubits)
    for r, row in enumerate(g.rows):
        for c, x in enumerate(row):
            planes[2, r, c] = x == "M"
            planes[3, r, c] = x == "r"
            planes[4, r, c] = x not in "QrM"
            planes[5, r, c] = r / max(1, h - 1)
            planes[6, r, c] = c / max(1, w - 1)
    actions = proposal_actions(g)
    af = np.zeros((len(actions), 10), np.float32)
    for i, a in enumerate(actions):
        af[i, PROPOSAL_KINDS.index(a[0])] = 1
        if a[0] == "single":
            af[i, 6:9] = [(a[1] + 1) / len(g.qubits), a[2] / max(1, h - 1), a[3] / max(1, w - 1)]
        elif a[0] == "compound":
            af[i, 9] = a[1] / len(g.qubits)
    context = np.concatenate(
        [
            features.flatten(),
            adjacency.flatten(),
            np.array([current / max(1, baseline), best / max(1, baseline), progress], np.float32),
        ]
    ).astype(np.float32)
    return {"planes": planes, "context": context, "actions": actions, "action_features": af}


def write_layout_output(directory, grid, qasm, title="Optimized grid"):
    """Write a matched layout, identity, original/mapped QASM and resource bundle.

    directory is created if needed; named files are overwritten. The SVG shows
    static tiles. This performs no compilation or replay, so consumers must
    verify the exported pair separately and retain identity.json with the grid.

    Args:
        directory (str | os.PathLike[str]): Destination directory for the artifacts described
            above.
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        qasm (str): Already synthesized, single-register OpenQASM 2 source matching the grid
            data-qubit count.
        title (str): Display title, escaped before inclusion in SVG markup.
    """
    g = as_state(grid)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layout = Layout.parse(g.text)
    (directory / "layout.txt").write_text(g.text)
    (directory / "layout.svg").write_text(layout.svg(title))
    (directory / "identity.json").write_text(json.dumps(g.identity_record(), indent=2))
    (directory / "mapped-circuit.qasm").write_text(remap_qasm(qasm, g))
    (directory / "original-circuit.qasm").write_text(qasm)
    (directory / "resources.json").write_text(json.dumps(layout.resources(), indent=2))
