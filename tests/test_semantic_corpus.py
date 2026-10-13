"""Ensure the actual compiler inputs match the gate lists checked semantically."""

import json
from pathlib import Path
import re
import unittest


class SemanticCorpusTests(unittest.TestCase):
    def test_all_twenty_qasm_inputs_match_their_gate_lists(self):
        root = Path(__file__).resolve().parents[1]
        entries = json.loads((root / "examples/corrected-v3/manifest.json").read_text())["entries"]
        self.assertEqual(len(entries), 20)
        self.assertEqual(
            {e["family"] for e in entries},
            {"line", "star", "matching", "dense", "QFT", "Haar", "QSP", "QSVT"},
        )
        for entry in entries:
            with self.subTest(circuit=entry["name"]):
                path = root / entry["path"]
                instructions = []
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith(("OPENQASM", "include", "qreg", "//")):
                        continue
                    match = re.fullmatch(
                        r"(h|x|z|s|sdg|t|tdg|cx)\s+q\[(\d+)\](?:,\s*q\[(\d+)\])?;", line
                    )
                    self.assertIsNotNone(match, line)
                    instructions.append(
                        [match[1], *[int(x) for x in match.groups()[1:] if x is not None]]
                    )
                self.assertEqual(instructions, json.loads((path.parent / "gates.json").read_text()))
