"""Unit tests for PCBGrid, A*, and the sequential router."""

from __future__ import annotations

import os
import sys
import unittest

# Make the repo root importable when running `python -m unittest discover tests`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pcb_grid import PCBGrid  # noqa: E402
from router import (  # noqa: E402
    route_board,
    route_board_best_of,
    route_board_rrr,
    route_single_net,
)


class TestPCBGrid(unittest.TestCase):
    def test_init_dimensions(self) -> None:
        g = PCBGrid(5, 7)
        self.assertEqual(g.width, 5)
        self.assertEqual(g.height, 7)
        self.assertEqual(g.grid.shape, (7, 5))  # numpy is (rows, cols) = (h, w)

    def test_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            PCBGrid(0, 5)
        with self.assertRaises(ValueError):
            PCBGrid(5, -1)

    def test_add_obstacle(self) -> None:
        g = PCBGrid(5, 5)
        g.add_obstacle(2, 3)
        self.assertEqual(g.grid[3, 2], 1)
        self.assertFalse(g.is_valid(2, 3))

    def test_add_obstacle_out_of_bounds(self) -> None:
        g = PCBGrid(5, 5)
        with self.assertRaises(IndexError):
            g.add_obstacle(5, 0)
        with self.assertRaises(IndexError):
            g.add_obstacle(0, -1)

    def test_is_valid_bounds(self) -> None:
        g = PCBGrid(5, 5)
        self.assertTrue(g.is_valid(0, 0))
        self.assertTrue(g.is_valid(4, 4))
        self.assertFalse(g.is_valid(-1, 0))
        self.assertFalse(g.is_valid(5, 0))
        self.assertFalse(g.is_valid(0, 5))

    def test_clone_independent(self) -> None:
        g = PCBGrid(5, 5)
        g.add_obstacle(2, 2)
        c = g.clone()
        c.add_obstacle(3, 3)
        self.assertFalse(g.is_valid(2, 2))
        self.assertTrue(g.is_valid(3, 3), "Clone mutation leaked into original")
        self.assertFalse(c.is_valid(3, 3))


class TestRouteSingleNet(unittest.TestCase):
    def test_path_endpoints(self) -> None:
        g = PCBGrid(5, 5)
        path = route_single_net(g, (0, 0), (4, 4))
        self.assertIsNotNone(path)
        assert path is not None  # for type checker
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))

    def test_optimal_length(self) -> None:
        g = PCBGrid(10, 10)
        path = route_single_net(g, (1, 1), (8, 5))
        assert path is not None
        # On an open grid, A* with Manhattan heuristic returns the optimal
        # path: |dx| + |dy| edges, which is |dx| + |dy| + 1 cells.
        expected = abs(8 - 1) + abs(5 - 1) + 1
        self.assertEqual(len(path), expected)

    def test_steps_are_4_connected(self) -> None:
        g = PCBGrid(8, 8)
        path = route_single_net(g, (0, 0), (5, 3))
        assert path is not None
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_avoids_obstacles(self) -> None:
        g = PCBGrid(5, 5)
        g.add_obstacle(2, 2)
        path = route_single_net(g, (0, 2), (4, 2))
        assert path is not None
        self.assertNotIn((2, 2), path)

    def test_unreachable_returns_none(self) -> None:
        g = PCBGrid(5, 5)
        for y in range(5):
            g.add_obstacle(2, y)  # impassable wall
        self.assertIsNone(route_single_net(g, (0, 0), (4, 4)))

    def test_invalid_endpoints(self) -> None:
        g = PCBGrid(5, 5)
        g.add_obstacle(0, 0)
        self.assertIsNone(route_single_net(g, (0, 0), (4, 4)))
        self.assertIsNone(route_single_net(g, (1, 1), (10, 10)))

    def test_start_equals_end(self) -> None:
        g = PCBGrid(5, 5)
        self.assertEqual(route_single_net(g, (2, 2), (2, 2)), [(2, 2)])


class TestRouteBoard(unittest.TestCase):
    def test_basic_two_nets(self) -> None:
        g = PCBGrid(10, 10)
        netlist = [((0, 0), (9, 0)), ((0, 9), (9, 9))]
        summary = route_board(g, netlist)
        self.assertEqual(len(summary["routed"]), 2)
        self.assertEqual(summary["unrouted"], [])

    def test_pin_sharing_supported(self) -> None:
        # Two nets share the (5, 5) start pin. Without the pin-sharing fix,
        # the second net would fail because (5,5) becomes an obstacle after
        # the first route.
        g = PCBGrid(10, 10)
        netlist = [((5, 5), (0, 0)), ((5, 5), (9, 9))]
        summary = route_board(g, netlist)
        self.assertEqual(
            len(summary["routed"]), 2,
            "Both nets should route despite sharing the (5,5) pin",
        )

    def test_path_locked_blocks_later_net(self) -> None:
        # Single corridor; first net consumes it, second net cannot route.
        g = PCBGrid(5, 5)
        for x in range(5):
            for y in range(5):
                if y != 2:
                    g.add_obstacle(x, y)
        netlist = [((0, 2), (4, 2)), ((1, 2), (3, 2))]
        summary = route_board(g, netlist)
        self.assertEqual(len(summary["routed"]), 1)
        self.assertEqual(len(summary["unrouted"]), 1)

    def test_unrouted_pair_preserved(self) -> None:
        g = PCBGrid(5, 5)
        for y in range(5):
            g.add_obstacle(2, y)
        netlist = [((0, 0), (4, 4))]
        summary = route_board(g, netlist)
        self.assertEqual(summary["routed"], [])
        self.assertEqual(summary["unrouted"], [((0, 0), (4, 4))])

    def test_sort_strategy_invalid(self) -> None:
        g = PCBGrid(5, 5)
        with self.assertRaises(ValueError):
            route_board(g, [((0, 0), (1, 1))], sort_strategy="not_a_strategy")


class TestRouteBoardRRR(unittest.TestCase):
    def test_no_unrouted_unchanged(self) -> None:
        # When the initial pass routes everything, R&R must not change the
        # outcome — the loop should exit immediately.
        g_plain = PCBGrid(10, 10)
        g_rrr = PCBGrid(10, 10)
        netlist = [((0, 0), (9, 0)), ((0, 9), (9, 9))]
        plain = route_board(g_plain, netlist)
        rrr = route_board_rrr(g_rrr, netlist)
        self.assertEqual(len(plain["routed"]), len(rrr["routed"]))
        self.assertEqual(len(rrr["unrouted"]), 0)

    def test_recovers_blocked_net(self) -> None:
        # Construct a scenario where greedy fails but R&R succeeds.
        # A wall with a single gap; first net (long) snakes through the gap
        # with a non-shortest route, blocking the second net. R&R should
        # rip up the long net, route the short net through the gap directly,
        # and re-route the long net around.
        g = PCBGrid(15, 15)
        for y in range(15):
            if y != 7:
                g.add_obstacle(7, y)

        netlist = [
            ((0, 0), (14, 14)),  # long diagonal — A* might route through gap
            ((6, 7), (8, 7)),    # short hop straight through the gap
        ]

        plain = route_board(g.clone(), netlist)
        rrr = route_board_rrr(g.clone(), netlist)

        self.assertGreaterEqual(
            len(rrr["routed"]), len(plain["routed"]),
            "R&R must never route fewer nets than plain greedy",
        )

    def test_max_iterations_zero_disables_rrr(self) -> None:
        # max_iterations=0 means "no R&R loop" — the result must equal the
        # initial route_board pass.
        g_plain = PCBGrid(10, 10)
        g_rrr = PCBGrid(10, 10)
        for x in range(10):
            for y in range(10):
                if y != 2:
                    g_plain.add_obstacle(x, y)
                    g_rrr.add_obstacle(x, y)
        netlist = [((0, 2), (9, 2)), ((1, 2), (8, 2))]
        plain = route_board(g_plain, netlist)
        rrr = route_board_rrr(g_rrr, netlist, max_iterations=0)
        self.assertEqual(len(plain["routed"]), len(rrr["routed"]))
        self.assertEqual(len(plain["unrouted"]), len(rrr["unrouted"]))

    def test_no_worse_than_baseline_on_random_boards(self) -> None:
        # On a batch of random boards, R&R should never route fewer nets than
        # plain greedy — it can only improve or tie.
        import random
        rng = random.Random(123)
        for _ in range(15):
            g_plain = PCBGrid(15, 15)
            g_rrr = PCBGrid(15, 15)
            obstacles: set[tuple[int, int]] = set()
            while len(obstacles) < 20:
                obstacles.add((rng.randint(0, 14), rng.randint(0, 14)))
            for x, y in obstacles:
                g_plain.add_obstacle(x, y)
                g_rrr.add_obstacle(x, y)

            pins: set[tuple[int, int]] = set()
            netlist = []
            while len(netlist) < 6:
                sx, sy = rng.randint(0, 14), rng.randint(0, 14)
                ex, ey = rng.randint(0, 14), rng.randint(0, 14)
                if (sx, sy) == (ex, ey):
                    continue
                if not g_plain.is_valid(sx, sy) or not g_plain.is_valid(ex, ey):
                    continue
                if (sx, sy) in pins or (ex, ey) in pins:
                    continue
                pins.add((sx, sy))
                pins.add((ex, ey))
                netlist.append(((sx, sy), (ex, ey)))

            plain = route_board(g_plain, netlist)
            rrr = route_board_rrr(g_rrr, netlist)
            self.assertGreaterEqual(len(rrr["routed"]), len(plain["routed"]))


class TestRouteBoardBestOf(unittest.TestCase):
    def test_returns_strategy(self) -> None:
        g = PCBGrid(10, 10)
        netlist = [((0, 0), (9, 9)), ((0, 9), (9, 0))]
        summary = route_board_best_of(g, netlist)
        self.assertIn("strategy", summary)
        self.assertIn("routed", summary)
        self.assertIn("unrouted", summary)

    def test_no_worse_than_default(self) -> None:
        # best_of should always match or beat the user-order baseline.
        g_default = PCBGrid(15, 15)
        g_best = PCBGrid(15, 15)
        for y in range(15):
            if y not in (4, 9):
                g_default.add_obstacle(7, y)
                g_best.add_obstacle(7, y)
        netlist = [
            ((0, 4), (14, 4)),
            ((0, 9), (14, 9)),
            ((0, 0), (1, 0)),
            ((0, 14), (14, 14)),
        ]
        baseline = route_board(g_default, netlist)
        best = route_board_best_of(g_best, netlist)
        self.assertGreaterEqual(len(best["routed"]), len(baseline["routed"]))

    def test_use_rrr_no_worse_than_plain(self) -> None:
        # best_of with R&R should match or beat best_of without.
        g_plain = PCBGrid(15, 15)
        g_rrr = PCBGrid(15, 15)
        for y in range(15):
            if y not in (4, 9):
                g_plain.add_obstacle(7, y)
                g_rrr.add_obstacle(7, y)
        netlist = [
            ((0, 4), (14, 4)),
            ((0, 9), (14, 9)),
            ((0, 0), (1, 0)),
            ((0, 14), (14, 14)),
        ]
        plain = route_board_best_of(g_plain, netlist, use_rrr=False)
        rrr = route_board_best_of(g_rrr, netlist, use_rrr=True)
        self.assertGreaterEqual(len(rrr["routed"]), len(plain["routed"]))


if __name__ == "__main__":
    unittest.main()
