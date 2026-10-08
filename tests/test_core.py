import json
import tempfile
import unittest
from pathlib import Path
from lattice_floorplan.layout import Layout
from lattice_floorplan.circuit import Circuit
from lattice_floorplan.compiler import Compiler, CompilerConfig
from lattice_floorplan.search import MappingEnv, optimize


class CoreTests(unittest.TestCase):
    def test_layout_guards(self):
        for text in ("", "Qr\nr", "Q1M", "QT", "QD", "Qr "):
            with self.subTest(text=text), self.assertRaises(ValueError):
                Layout.parse(text)

    def test_components_and_protected_outputs(self):
        layout = Layout.parse("Qrrrr\n11r11")
        self.assertEqual(layout.resources()["factory_count"], 2)
        with self.assertRaises(ValueError):
            layout.move_data((0, 0), (0, 1))

    def test_mapping_changes_only_declaration(self):
        circuit = Circuit.parse("DeclareLogicalQubitPatches 7,3\nMultiBodyMeasure 7:X,3:X\n")
        self.assertEqual(
            circuit.mapped([3, 7]), "DeclareLogicalQubitPatches 3,7\nMultiBodyMeasure 7:X,3:X\n"
        )
        with self.assertRaises(ValueError):
            circuit.mapped([3, 3])

    def test_episode_masks_and_terminal_score(self):
        circuit = Circuit.parse("DeclareLogicalQubitPatches 0,1\n")
        calls = []

        def evaluate(order):
            calls.append(order)
            return {"status": "success", "slices": 4}

        env = MappingEnv(circuit, Layout.parse("QrQ"), evaluate, 4)
        self.assertEqual(env.step(1)[1:3], (0.0, False))
        self.assertEqual(env.observation()["action_mask"], [True, False])
        self.assertFalse(calls)
        with self.assertRaises(ValueError):
            env.step(1)
        self.assertEqual(env.step(0)[1:3], (0.5, True))
        self.assertEqual(calls, [(1, 0)])
        with self.assertRaises(ValueError):
            env.step(0)

    def test_failure_reward_below_any_success(self):
        c = Circuit.parse("DeclareLogicalQubitPatches 0\n")
        env = MappingEnv(c, Layout.parse("Q"), lambda _: {"status": "timeout", "slices": None}, 1)
        self.assertEqual(env.step(0)[1], 0)

    def test_policy_learns_in_small_controlled_problem(self):
        class ToyCompiler:
            def run(self, text, layout, directory):
                c = Circuit.parse(text)
                return {"status": "success", "slices": 1 if c.ids == (1, 0) else 100}

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "search"
            result = optimize(
                ToyCompiler(),
                "DeclareLogicalQubitPatches 0,1\n",
                Layout.parse("QrQ"),
                path,
                "reinforce",
                200,
                2,
            )
            policy = json.loads((path / "policy.json").read_text())
            self.assertGreater(policy["logits"][0][1], policy["logits"][0][0])
            self.assertEqual(result["compiler_calls"], 200)
            self.assertEqual(result["best_slices"], 1)


class AdapterTests(unittest.TestCase):
    def run_fake(self, body, timeout=2):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            binary = path / "fake"
            import sys

            binary.write_text(
                "#!" + str(Path(sys.executable).resolve()) + "\nimport sys,json,time\nfrom pathlib import Path\n" + body
            )
            binary.chmod(0o755)
            return Compiler(binary, CompilerConfig(timeout=timeout)).run(
                "DeclareLogicalQubitPatches 0\n", Layout.parse("Q"), path / "run"
            )

    def test_clean_wave_contract(self):
        r = self.run_fake(
            "Path(sys.argv[sys.argv.index('-o')+1]).write_text('[[[null]]]')\nprint('1,2,0.01')\n"
        )
        self.assertEqual((r["status"], r["slices"], r["reported_slices"]), ("success", 1, 2))

    def test_partial_output_not_success_even_with_zero_exit(self):
        r = self.run_fake(
            "Path(sys.argv[sys.argv.index('-o')+1]).write_text('[[[null]]]')\nprint('Encountered exception: stopped\\n1,2,0.01')\n"
        )
        self.assertEqual(r["status"], "error")
        self.assertIsNone(r["slices"])

    def test_truncated_json(self):
        r = self.run_fake(
            "Path(sys.argv[sys.argv.index('-o')+1]).write_text('[')\nprint('1,2,0.01')\n"
        )
        self.assertEqual(r["status"], "error")

    def test_summary_mismatch(self):
        r = self.run_fake(
            "Path(sys.argv[sys.argv.index('-o')+1]).write_text('[[[null]]]')\nprint('1,7,0.01')\n"
        )
        self.assertEqual(r["status"], "error")

    def test_timeout(self):
        r = self.run_fake("time.sleep(2)\n", timeout=0.02)
        self.assertEqual(r["status"], "timeout")
        self.assertIsNone(r["slices"])

    def test_nonzero_with_valid_json(self):
        r = self.run_fake(
            "Path(sys.argv[sys.argv.index('-o')+1]).write_text('[[[null]]]')\nprint('1,2,0.01')\nsys.exit(1)\n"
        )
        self.assertEqual(r["status"], "error")
