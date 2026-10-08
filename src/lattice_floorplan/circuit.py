"""Mapping changes declaration order only; operation IDs stay unchanged."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Circuit:
    """Immutable LLI declaration IDs and operation lines for mapping experiments.

    Attributes:
        ids (tuple[int, ...]): Distinct logical IDs in allocation order.
        operations (tuple[str, ...]): LLI operation lines using those logical IDs.
    """

    ids: tuple[int, ...]
    operations: tuple[str, ...]

    @classmethod
    def parse(cls, text):
        """Parse one leading patch declaration; reject missing or duplicate logical IDs.

        Args:
            text (str): LLI beginning with one DeclareLogicalQubitPatches declaration.

        Returns:
            Circuit: Parsed declaration IDs and unchanged operation lines.

        Raises:
            ValueError: Missing/invalid leading declaration, duplicate logical IDs, or multiple declarations.
        """
        lines = tuple(line.strip() for line in text.splitlines() if line.strip())
        if not lines or not re.fullmatch(
            r"DeclareLogicalQubitPatches\s+\d+(?:\s*,\s*\d+)*", lines[0]
        ):
            raise ValueError("Expected LLI beginning with DeclareLogicalQubitPatches")
        ids = tuple(map(int, re.findall(r"\d+", lines[0])))
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate logical IDs")
        if any(line.startswith("DeclareLogicalQubitPatches") for line in lines[1:]):
            raise ValueError("Multiple declarations are unsupported")
        return cls(ids, lines[1:])

    def mapped(self, site_order=None):
        """Return LLI with declarations permuted into site_order and operations unchanged.

        site_order must contain every declared logical ID exactly once; None
        preserves declaration order. This changes initial allocation, not gates.
        Raises ValueError for an invalid permutation.

        Args:
            site_order (Sequence[int] | None): Permutation of declared IDs for initial
                allocation; None preserves their order.

        Returns:
            str: Newline-terminated LLI with the requested declaration order.

        Raises:
            ValueError: Mapping must contain each declared logical ID exactly once.
        """
        order = self.ids if site_order is None else tuple(site_order)
        if len(order) != len(self.ids) or set(order) != set(self.ids):
            raise ValueError("Mapping must contain each declared logical ID exactly once")
        return (
            "DeclareLogicalQubitPatches "
            + ",".join(map(str, order))
            + "\n"
            + "\n".join(self.operations)
            + "\n"
        )
