import unittest, random, math
from lattice_floorplan.geometry import GridState, valid_geometry
from lattice_floorplan.supply import (
    supply_count,
    set_supply,
    supply_moves,
    initialize_supply,
    initialize_padded_supply,
    propose_with_supply,
)
from lattice_floorplan.tempering import search, TemperingConfig

G = GridState.parse("rrrrr\nrMrMr\nrrQrr\nrrrrr\nrrrrr\n")


class SupplyTests(unittest.TestCase):
    def test_identity_count_and_routing_preserved(self):
        moves = supply_moves(G)
        self.assertTrue(moves)
        for _, g in moves:
            self.assertEqual(supply_count(g), 2)
            self.assertEqual(g.qubits, G.qubits)
            self.assertTrue(valid_geometry(g))
            self.assertIn(G, [x for _, x in supply_moves(g)])

    def test_reverse_degree_detailed_balance(self):
        for _, g in supply_moves(G):
            dx, dy = len(supply_moves(G)), len(supply_moves(g))
            ea, eb, temp = 2.0, 3.0, 2.0
            forward = math.exp(-ea / temp) / dx * min(1, math.exp(-(eb - ea) / temp) * dx / dy)
            reverse = math.exp(-eb / temp) / dy * min(1, math.exp(-(ea - eb) / temp) * dy / dx)
            self.assertAlmostEqual(forward, reverse)

    def test_count_initialization_repeatable_and_rejects_zero(self):
        for n in [1, 2, 3, 4]:
            g = initialize_supply(G, n, 61)
            self.assertIsNotNone(g)
            self.assertEqual(supply_count(g), n)
            self.assertEqual(g, initialize_supply(G, n, 61))
            self.assertEqual(g.qubits, G.qubits)
        with self.assertRaises(ValueError):
            initialize_supply(G, 0, 1)

    def test_padded_initialization_can_exceed_original_area(self):
        g = initialize_padded_supply(G, 32, 61)
        self.assertEqual(supply_count(g), 32)
        self.assertTrue(valid_geometry(g))
        self.assertGreater(g.area, G.area)
        self.assertEqual(g.logical_to_site, G.logical_to_site)

    def test_seven_family_search_keeps_count(self):
        class Oracle:
            def evaluate(self, qasm, g):
                assert supply_count(g) == 2
                return {"status": "success", "slices": 20}

        r = search(
            {"name": "x", "family": "test", "qasm": "q"},
            Oracle(),
            G,
            "sa",
            64,
            61,
            TemperingConfig(move_supply=True),
        )
        self.assertEqual(r["queries"], 64)
        self.assertTrue(any(h.get("proposal_kind") == "move_supply" for h in r["history"]))
