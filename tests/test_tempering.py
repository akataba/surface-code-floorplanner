import math, unittest
from lattice_floorplan.tempering import search, TemperingConfig, local_log_accept, swap_log_accept
from lattice_floorplan.geometry import GridState
from fixtures import GRID


class TemperingTests(unittest.TestCase):
    def test_local_detailed_balance_with_asymmetric_degrees(self):
        ea, eb, t, da, db = 10, 13, 2, 16, 27
        ab = min(1, math.exp(local_log_accept(ea, eb, t, da, db)))
        ba = min(1, math.exp(local_log_accept(eb, ea, t, db, da)))
        self.assertAlmostEqual(math.exp(-ea / t) / da * ab, math.exp(-eb / t) / db * ba)

    def test_swap_sign_and_balance(self):
        self.assertLess(swap_log_accept(10, 20, 1, 10), 0)
        self.assertGreater(swap_log_accept(20, 10, 1, 10), 0)
        x = swap_log_accept(10, 20, 1, 10)
        self.assertEqual(x, -swap_log_accept(20, 10, 1, 10))
        self.assertAlmostEqual(
            math.exp(-10 / 1 - 20 / 10) * math.exp(x), math.exp(-20 / 1 - 10 / 10)
        )

    def test_exact_budget_failure_rejection_and_shared_calibration(self):
        class Oracle:
            def __init__(self):
                self.calls = 0

            def evaluate(self, qasm, g):
                self.calls += 1
                return {
                    "status": "success" if g == GridState.parse(GRID) else "error",
                    "slices": 100 if g == GridState.parse(GRID) else None,
                }

        histories = []
        for method in ["sa", "restart_sa", "parallel_tempering"]:
            o = Oracle()
            r = search({"name": "x", "family": "line", "qasm": "q"}, o, GRID, method, 64, 7)
            self.assertEqual(o.calls, 64)
            self.assertEqual(r["best_slices"], 100)
            self.assertEqual(r["failed_candidates"] + r["null_proposals"], 63)
            self.assertEqual(r["best_grid"], GRID)
            histories.append([x["grid_sha256"] for x in r["history"][:9]])
            self.assertEqual(sum(r["swap_attempts"]), 9 if method == "parallel_tempering" else 0)
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(histories[0], histories[2])

    def test_temperature_roundtrips_on_constant_energy(self):
        class Oracle:
            def evaluate(self, qasm, g):
                return {"status": "success", "slices": 8100 / g.area}

        r = search(
            {"name": "x", "family": "line", "qasm": "q"},
            Oracle(),
            GRID,
            "parallel_tempering",
            256,
            1,
        )
        self.assertEqual(r["swap_attempts"], r["swap_accepts"])
        self.assertTrue(all(x > 0 for x in r["roundtrips"]))
        self.assertAlmostEqual(r["best_objective"], 8100)

    def test_bloat_has_cost_even_when_slices_improve(self):
        class Oracle:
            def evaluate(self, qasm, g):
                return {"status": "success", "slices": 100 if g.area <= 81 else 99}

        r = search(
            {"name": "x", "family": "line", "qasm": "q"},
            Oracle(),
            GRID,
            "parallel_tempering",
            64,
            11,
        )
        self.assertLessEqual(r["best_objective"], 8100)
        self.assertLessEqual(r["best_area"], 81)
        for h in r["history"]:
            self.assertEqual(h["objective"], h["slices"] * h["area"])

    def test_repeatability_and_actual_grid_output(self):
        class Oracle:
            def evaluate(self, qasm, g):
                return {"status": "success", "slices": 100 + sum(c * 2 for r, c in g.qubits)}

        task = {"name": "x", "family": "line", "qasm": "q"}
        a = search(task, Oracle(), GRID, "parallel_tempering", 64, 11)
        b = search(task, Oracle(), GRID, "parallel_tempering", 64, 11)
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["swaps"], b["swaps"])
        self.assertEqual(a["best_grid"], b["best_grid"])
        self.assertLessEqual(a["best_slices"], a["baseline"])
        self.assertEqual(len(a["logical_qubit_positions"]), 5)


if __name__ == "__main__":
    unittest.main()
