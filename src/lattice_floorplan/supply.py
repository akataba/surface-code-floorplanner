"""Explicit-count abstract M supply placement. No physical factory cost claim."""

import math
import random
from functools import lru_cache
from .geometry import GridState, as_state, valid_geometry, Proposal, propose


def supply_count(grid):
    """Count abstract M respawn cells in an ASCII grid or GridState.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.

    Returns:
        int: Number of abstract M cells.
    """
    return sum(row.count("M") for row in as_state(grid).rows)


def set_supply(grid, cells):
    """Replace all M cells with the requested (row, column) positions.

    Q identities and dimensions stay unchanged. Raises ValueError for empty,
    out-of-bounds or occupied placements; returns None if geometry is invalid.
    Counts refer to abstract sources, not physical distillation factories.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        cells (Iterable[tuple[int, int]]): Requested distinct M coordinates as zero-based (row,
            column) pairs.

    Returns:
        GridState | None: New supply placement, or None if routing geometry fails validation.

    Raises:
        ValueError: Empty supply placement, out-of-bounds coordinate, or placement on a non-routing/non-M cell.
    """
    g = as_state(grid)
    cells = set(cells)
    if not cells:
        raise ValueError("These T-consuming circuits need at least one supply cell")
    rows = [list(row) for row in g.rows]
    for r, row in enumerate(rows):
        for c, x in enumerate(row):
            if x == "M":
                rows[r][c] = "r"
    for r, c in cells:
        if not (0 <= r < len(rows) and 0 <= c < len(rows[0])) or rows[r][c] != "r":
            raise ValueError("Supply must occupy routing or former supply cells")
        rows[r][c] = "M"
    out = GridState(tuple("".join(row) for row in rows), g.qubits)
    return out if valid_geometry(out) else None


@lru_cache(maxsize=256)
def supply_moves(grid):
    """Return cached ((source, target), state) pairs for valid single-M relocations.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.

    Returns:
        tuple[tuple[tuple[tuple[int, int], tuple[int, int]], GridState], ...]:
            ((source,target),state) pairs for valid moves.
    """
    g = as_state(grid)
    ms = {(r, c) for r, row in enumerate(g.rows) for c, x in enumerate(row) if x == "M"}
    out = []
    for old in sorted(ms):
        for r, row in enumerate(g.rows):
            for c, x in enumerate(row):
                if x != "r":
                    continue
                new = set_supply(g, (ms - {old}) | {(r, c)})
                if new is not None:
                    out.append(((old, (r, c)), new))
    return tuple(out)


def propose_with_supply(grid, rng):
    # Seven equally likely families: six original geometry families plus M relocation.
    """Sample seven equal move families, including one M relocation family.

    Relocation endpoints are uniform among valid moves and carry the reverse/
    forward degree correction. An empty move family yields a counted self-loop.
    The passed random.Random supplies all sampling; supply count is unchanged.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        rng (random.Random): Caller-owned pseudorandom generator; sampling advances its state.

    Returns:
        Proposal: Geometry or M-relocation endpoint with acceptance metadata.
    """
    if rng.randrange(7):
        return propose(grid, rng)
    g = as_state(grid)
    moves = supply_moves(g)
    if not moves:
        return Proposal(g, "move_supply", ("move_supply",), null_reason="no_supply_endpoint")
    action, new = rng.choice(moves)
    return Proposal(
        new,
        "move_supply",
        ("move_supply", *action),
        log_reverse_over_forward=math.log(len(moves)) - math.log(len(supply_moves(new))),
    )


def initialize_supply(grid, count, seed):
    """Set tested count by removal/addition; RNG seed and greedy construction explicit.

    No compilation or hidden score selection in construction. Failure is reported.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        count (int): Positive target number of abstract M respawn cells, not physical factories.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.

    Returns:
        GridState | None: Seeded placement with requested count, or None if construction cannot
            continue.

    Raises:
        ValueError: Count must be positive.
    """
    if count < 1:
        raise ValueError("Count must be positive")
    g = as_state(grid)
    rng = random.Random(seed)
    ms = [(r, c) for r, row in enumerate(g.rows) for c, x in enumerate(row) if x == "M"]
    rng.shuffle(ms)
    if count <= len(ms):
        return set_supply(g, ms[:count])
    while len(ms) < count:
        candidates = []
        for r, row in enumerate(g.rows):
            for c, x in enumerate(row):
                if x == "r":
                    new = set_supply(g, ms + [(r, c)])
                    if new is not None:
                        candidates.append(((r, c), new))
        if not candidates:
            return None
        cell, g = rng.choice(candidates)
        ms.append(cell)
    return g


def initialize_padded_supply(grid, count, seed):
    """Additional supplies in spaced outer rows separated by routing corridors.

    This is an initialization heuristic, not a constraint on subsequent positions.

    Args:
        grid (GridState | str): Explicit geometry/identity state or ASCII text; text assigns
            identities in row-major Q order.
        count (int): Positive target number of abstract M respawn cells, not physical factories.
        seed (int): Seed for reproducible sampling; global RNG effects, if any, are described
            above.

    Returns:
        GridState | None: Padded placement, or None from unpadded construction when reducing the
            count.

    Raises:
        ValueError: Count is nonpositive or padded construction disconnects routing.
    """
    from .geometry import resize_at

    g = as_state(grid)
    if count <= supply_count(g):
        return initialize_supply(g, count, seed)
    remaining = count - supply_count(g)
    rng = random.Random(seed)
    while remaining:
        # From top: sparse supply row, clear routing row, previous grid.
        g = resize_at(g, "insert_row", 0)
        g = resize_at(g, "insert_row", 0)
        choices = list(range(0, g.shape[1], 2))
        rng.shuffle(choices)
        take = choices[: min(remaining, len(choices))]
        ms = [(r, c) for r, row in enumerate(g.rows) for c, x in enumerate(row) if x == "M"]
        g = set_supply(g, ms + [(0, c) for c in take])
        if g is None:
            raise ValueError("Padded supply construction disconnected routing")
        remaining -= len(take)
    return g
