"""Conservative ASCII support for the pinned liblsqecc parser."""

from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    """Immutable rectangular ASCII tiles using the supported liblsqecc alphabet.

    Attributes:
        rows (tuple[str, ...]): Immutable ASCII rows; use parse/read for grammar validation.
    """

    rows: tuple[str, ...]

    @classmethod
    def parse(cls, text: str):
        """Parse ASCII text, ignoring empty lines, and validate supported tile types.

        Raises ValueError for a nonrectangular/empty grid, missing Q sites,
        unsupported symbols, or mixed M cells and digit-labeled factories.
        Parser validity alone does not establish routing or compiler feasibility.

        Args:
            text (str): Nonempty rectangular ASCII layout using supported tile symbols.

        Returns:
            Layout: Validated immutable ASCII rows.

        Raises:
            ValueError: Empty/nonrectangular grid, unsupported symbol, missing Q site, or mixed M/digit factories.
        """
        rows = tuple(line for line in text.splitlines() if line)
        if not rows or not rows[0] or len({len(r) for r in rows}) != 1:
            raise ValueError("Layout must be a nonempty rectangular grid")
        chars = set("".join(rows))
        unknown = chars - set("rQAMXY0123456789")
        if unknown:
            raise ValueError(
                f"Unsupported cells {sorted(unknown)}; T/D are recognized upstream but not materialized as Q sites"
            )
        if "M" in chars and chars & set("0123456789"):
            raise ValueError("Explicit factories and M respawn cells cannot coexist")
        if "Q" not in chars:
            raise ValueError("At least one Q site is required")
        return cls(rows)

    @classmethod
    def read(cls, path):
        """Read a text file and apply parse validation; propagate file-access errors.

        Args:
            path (str | os.PathLike[str]): Filesystem path of the input text file.

        Returns:
            Layout: Validated layout loaded from the file.

        Raises:
            ValueError: File content is not a supported layout.
            OSError: The file cannot be read.
        """
        return cls.parse(Path(path).read_text())

    @property
    def text(self):
        """Return the ASCII grid with one trailing newline.

        Returns:
            str: Newline-terminated ASCII grid.
        """
        return "\n".join(self.rows) + "\n"

    @property
    def sites(self):
        """Return Q coordinates as zero-based (row, column) pairs in row-major order.

        Returns:
            tuple[tuple[int, int], ...]: Row-major zero-based Q coordinates.
        """
        return tuple(
            (r, c) for r, row in enumerate(self.rows) for c, x in enumerate(row) if x == "Q"
        )

    def neighbors(self, cell):
        """Return in-bounds orthogonal neighbors of a (row, column) cell.

        Args:
            cell (tuple[int, int]): Zero-based (row, column) position in the grid.

        Returns:
            list[tuple[int, int]]: In-bounds orthogonal coordinates.
        """
        r, c = cell
        return [
            (a, b)
            for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= a < len(self.rows) and 0 <= b < len(self.rows[0])
        ]

    def factories(self):
        """Return connected equal-digit regions and adjacent routing output cells.

        M respawn cells are excluded. Disconnected regions with the same digit
        are separate entries; this is an ASCII resource description.

        Returns:
            list[dict[str, Any]]: Each region has label, sorted cells and sorted adjacent
                queue_cells. M cells are excluded.
        """
        seen, regions = set(), []
        for r, row in enumerate(self.rows):
            for c, x in enumerate(row):
                if x not in "0123456789" or (r, c) in seen:
                    continue
                pending, region = [(r, c)], set()
                while pending:
                    cell = pending.pop()
                    if cell in region:
                        continue
                    region.add(cell)
                    pending.extend(
                        n
                        for n in self.neighbors(cell)
                        if self.rows[n[0]][n[1]] == x and n not in region
                    )
                seen.update(region)
                queue = {
                    n
                    for cell in region
                    for n in self.neighbors(cell)
                    if self.rows[n[0]][n[1]] == "r"
                }
                regions.append({"label": x, "cells": sorted(region), "queue_cells": sorted(queue)})
        return regions

    def resources(self):
        """Return tile dimensions/counts, digit factory regions and access warnings.

        Returns:
            dict[str, Any]: height, width, area in tiles, symbol counts, data_sites,
                factory_count, factories and warnings.
        """
        counts = dict(sorted(Counter("".join(self.rows)).items()))
        factories = self.factories()
        return {
            "height": len(self.rows),
            "width": len(self.rows[0]),
            "area": len(self.rows) * len(self.rows[0]),
            "counts": counts,
            "data_sites": len(self.sites),
            "factory_count": len(factories),
            "factories": factories,
            "warnings": [
                "Factory has no adjacent routing output cell"
                for f in factories
                if not f["queue_cells"]
            ],
        }

    def move_data(self, source, target):
        """Swap Q/r while preserving every factory's output-cell set.

        Args:
            source (tuple[int, int]): In-bounds Q coordinate to vacate.
            target (tuple[int, int]): In-bounds routing coordinate to occupy.

        Returns:
            Layout: New grid with the Q/r swap applied; original is unchanged.

        Raises:
            ValueError: Source/target is not Q/r or the move changes factory output access.
        """
        if self.rows[source[0]][source[1]] != "Q" or self.rows[target[0]][target[1]] != "r":
            raise ValueError("Expected Q source and r destination")
        protected = {tuple(n) for f in self.factories() for n in f["queue_cells"]}
        if target in protected or any(
            self.rows[n[0]][n[1]].isdigit() for n in self.neighbors(source)
        ):
            raise ValueError("Move would change factory output access")
        rows = [list(r) for r in self.rows]
        rows[source[0]][source[1]], rows[target[0]][target[1]] = "r", "Q"
        return Layout.parse("\n".join("".join(r) for r in rows))

    def svg(self, title="Layout"):
        """Return SVG markup for the static tiles with an escaped title; write no files.

        Args:
            title (str): Display title, escaped before inclusion in SVG markup.

        Returns:
            str: Complete SVG markup; no file is written.
        """
        size, top = 36, 48
        w, h = len(self.rows[0]) * size, len(self.rows) * size
        palette = {
            "Q": "#2563eb",
            "r": "#edf2f7",
            "A": "#8b5cf6",
            "M": "#f59e0b",
            "X": "#334155",
            "Y": "#14b8a6",
        }
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{max(w, 420)}" height="{h + 100}" viewBox="0 0 {max(w, 420)} {h + 100}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="8" y="28" font-family="sans-serif" font-size="20">{escape(title)}</text>',
        ]
        for r, row in enumerate(self.rows):
            for c, x in enumerate(row):
                color = palette.get(x, "#fb923c")
                parts.append(
                    f'<rect x="{c * size}" y="{r * size + top}" width="36" height="36" fill="{color}" stroke="white"/>'
                )
                parts.append(
                    f'<text x="{c * size + 18}" y="{r * size + top + 24}" text-anchor="middle" font-family="monospace" font-size="16" fill="{"#334155" if x == "r" else "white"}">{x}</text>'
                )
        parts.append(
            f'<text x="8" y="{h + top + 25}" font-family="sans-serif" font-size="12">Q data · r routing · digits factories · M respawn</text></svg>'
        )
        return "\n".join(parts)
