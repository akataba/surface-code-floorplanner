import json
import os
import tempfile
import unittest
from pathlib import Path
from lattice_floorplan.compiler import Compiler, CompilerConfig
from lattice_floorplan.circuit import Circuit
from lattice_floorplan.layout import Layout

BINARY = os.environ.get("LSQECC_SLICER")


@unittest.skipUnless(BINARY, "Set LSQECC_SLICER to run real compiler tests")
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.compiler = Compiler(BINARY, CompilerConfig(timeout=3))

    def run_case(self, text, layout="QrQ\nrrr", name="run"):
        return self.compiler.run(text, Layout.parse(layout), self.root / name)

    def test_known_single_measurement(self):
        result = self.run_case("DeclareLogicalQubitPatches 0,1\nMultiBodyMeasure 0:X,1:X\n")
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(
            (result["slices"], result["reported_slices"], result["instructions"]), (1, 2, 1)
        )

    def test_declaration_order_places_actual_patch_ids(self):
        result = self.run_case("DeclareLogicalQubitPatches 7,3\nLogicalPauli 7 X\n")
        self.assertEqual(result["status"], "success", result)
        slices = json.loads((self.root / "run/slices.json").read_text())
        self.assertEqual(slices[0][0][0]["text"], "Id: 7")
        self.assertEqual(slices[0][0][2]["text"], "Id: 3")

    def test_insufficient_sites(self):
        result = self.run_case("DeclareLogicalQubitPatches 0,1\nLogicalPauli 0 X\n", layout="Qr")
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["slices"])

    def test_late_bad_instruction_not_scored(self):
        result = self.run_case(
            "DeclareLogicalQubitPatches 0,1\nLogicalPauli 0 X\nLogicalPauli 1 X\nNotAnInstruction 0\n"
        )
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["slices"])

    def test_blocked_route_not_scored(self):
        result = self.run_case(
            "DeclareLogicalQubitPatches 0,1\nMultiBodyMeasure 0:X,1:X\n", layout="QXQ"
        )
        self.assertNotEqual(result["status"], "success")
        self.assertIsNone(result["slices"])

    def test_factory_wait_changes_emitted_frames(self):
        text = "DeclareLogicalQubitPatches 0\nRequestMagicState 1 0\n"
        layout = Layout.parse("Qrr\nrrr\n111")
        a = self.compiler.run(text, layout, self.root / "a")
        b = Compiler(BINARY, CompilerConfig(distillation_time=20, timeout=3)).run(
            text, layout, self.root / "b"
        )
        self.assertEqual(a["status"], "success", a)
        self.assertEqual(b["status"], "success", b)
        self.assertGreater(b["slices"], a["slices"])

    def test_repeated_compilations_are_identical(self):
        text = "DeclareLogicalQubitPatches 0,1\nMultiBodyMeasure 0:X,1:X\n"
        frames = []
        for i in range(3):
            result = self.run_case(text, name=f"repeat-{i}")
            self.assertEqual(result["status"], "success")
            frames.append((self.root / f"repeat-{i}" / "slices.json").read_text())
        self.assertEqual(len(set(frames)), 1)

    def test_summary_mode_matches_retained_frames(self):
        text = "DeclareLogicalQubitPatches 0,1\nMultiBodyMeasure 0:X,1:X\n"
        layout = Layout.parse("QrQ\nrrr")
        full = self.compiler.run(text, layout, self.root / "full")
        summary = self.compiler.run(text, layout, self.root / "summary", retain_slices=False)
        self.assertEqual(full["status"], "success")
        self.assertEqual(summary["status"], "success")
        self.assertEqual(full["slices"], summary["slices"])
        self.assertEqual(summary["count_source"], "pinned_wave_counter_minus_one")

    def test_summary_mode_rejects_blocked_routes(self):
        text = "DeclareLogicalQubitPatches 0,1\nMultiBodyMeasure 0:X,1:X\n"
        result = self.compiler.run(
            text, Layout.parse("QXQ"), self.root / "blocked-summary", retain_slices=False
        )
        self.assertNotEqual(result["status"], "success")
        self.assertIsNone(result["slices"])
