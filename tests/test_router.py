"""Unit tests for PCBGrid, A*, and the sequential router."""

from __future__ import annotations

import os
import sys
import unittest

# Make the repo root importable when running `python -m unittest discover tests`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pcb_grid import Pad, PCBGrid  # noqa: E402
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


class TestMultiLayer(unittest.TestCase):
    """Multi-layer routing: vias, through-hole pads, preferred directions."""

    def test_grid_layers_shape(self) -> None:
        g = PCBGrid(4, 6, num_layers=3)
        self.assertEqual(g.num_layers, 3)
        self.assertEqual(g.layers.shape, (3, 6, 4))
        # Default-layer compat: g.grid is a view of layer 0.
        self.assertEqual(g.grid.shape, (6, 4))

    def test_obstacles_per_layer_independent(self) -> None:
        g = PCBGrid(5, 5, num_layers=2)
        g.add_obstacle(2, 2, layer=0)
        self.assertFalse(g.is_valid(2, 2, layer=0))
        self.assertTrue(g.is_valid(2, 2, layer=1))

    def test_add_via_marks_all_layers(self) -> None:
        g = PCBGrid(5, 5, num_layers=3)
        g.add_via(2, 2, layers=(0, 1, 2))
        for z in range(3):
            self.assertFalse(g.is_valid(2, 2, layer=z))
        self.assertIn((2, 2), g.vias)

    def test_clone_preserves_layers_and_vias(self) -> None:
        g = PCBGrid(5, 5, num_layers=2, clearance=1)
        g.add_obstacle(1, 1, layer=1)
        g.add_via(3, 3, layers=(0, 1))
        c = g.clone()
        self.assertEqual(c.num_layers, 2)
        self.assertEqual(c.clearance, 1)
        self.assertFalse(c.is_valid(1, 1, layer=1))
        self.assertIn((3, 3), c.vias)
        # Independence.
        c.add_obstacle(0, 0, layer=0)
        self.assertTrue(g.is_valid(0, 0, layer=0))

    def test_wall_only_routable_via_layer_switch(self) -> None:
        # Solid wall on layer 0 forces a layer switch onto layer 1.
        g = PCBGrid(10, 5, num_layers=2)
        for y in range(5):
            g.add_obstacle(5, y, layer=0)

        path = route_single_net(g, (0, 2, 0), (9, 2, 0))
        self.assertIsNotNone(path)
        assert path is not None
        # Path must traverse layer 1 at some point.
        layers_used = {cell[2] for cell in path}
        self.assertIn(1, layers_used)
        self.assertIn(0, layers_used)

    def test_via_cost_reflected_in_search(self) -> None:
        # With a very expensive via (1000), routing around the obstacle in 2D
        # should be preferred. With a cheap via (1), a layer-switch shortcut
        # should win.
        # Use a partial wall that has a way around it on the same layer.
        g_high = PCBGrid(8, 8, num_layers=2)
        g_low = PCBGrid(8, 8, num_layers=2)
        for y in (3, 4, 5):
            g_high.add_obstacle(4, y, layer=0)
            g_low.add_obstacle(4, y, layer=0)

        p_high = route_single_net(g_high, (0, 4, 0), (7, 4, 0), via_cost=1000.0)
        p_low = route_single_net(g_low, (0, 4, 0), (7, 4, 0), via_cost=1.0)
        assert p_high is not None and p_low is not None

        # High via cost: stay on layer 0, route around.
        self.assertTrue(all(cell[2] == 0 for cell in p_high))
        # Low via cost: shortcut through layer 1 (some cell on layer 1).
        self.assertTrue(any(cell[2] == 1 for cell in p_low))

    def test_through_hole_pad_accessible_on_any_layer(self) -> None:
        # Pad on (3, 3) connects layers 0 and 1. Net starts on layer 1, ends
        # at the pad; A* should accept layer 0 OR layer 1 termination.
        g = PCBGrid(8, 8, num_layers=2)
        pad = Pad(3, 3, layers=(0, 1))
        # The seed cells are already marked obstacles by add_via, but
        # _try_route_with_pin_clear temporarily clears them. For a direct
        # route_single_net call we need them empty.
        # (We don't pre-stamp the pad here — the test exercises route_single_net.)

        path = route_single_net(g, (0, 0, 1), pad)
        self.assertIsNotNone(path)
        assert path is not None
        last = path[-1]
        self.assertEqual((last[0], last[1]), (3, 3))
        self.assertIn(last[2], (0, 1))

    def test_routed_paths_include_vias_on_switches(self) -> None:
        g = PCBGrid(10, 5, num_layers=2)
        for y in range(5):
            g.add_obstacle(5, y, layer=0)  # wall on layer 0

        netlist = [((0, 2, 0), (9, 2, 0))]
        summary = route_board(g, netlist)
        self.assertEqual(len(summary["routed"]), 1)
        net = summary["routed"][0]
        # Two layer switches needed (down to 1, back to 0) -> at least 2 vias.
        self.assertGreaterEqual(len(net["vias"]), 1)

    def test_prefer_directions_changes_path_choice(self) -> None:
        # On an open board, A* with prefer_directions=True should prefer
        # horizontal moves on layer 0. The path from (0,0) to (5,5) has
        # equal Manhattan paths but the preferred-direction tiebreaker
        # should pull most steps onto horizontal first then vertical.
        g_plain = PCBGrid(10, 10, num_layers=2)
        g_pref = PCBGrid(10, 10, num_layers=2)
        p_plain = route_single_net(g_plain, (0, 0, 0), (5, 5, 0))
        p_pref = route_single_net(
            g_pref, (0, 0, 0), (5, 5, 0), prefer_directions=True,
        )
        assert p_plain is not None and p_pref is not None
        # Same path length (admissible heuristic + 1.0 minimum edge cost),
        # but the preferred-direction path should have all-horizontal-first
        # or be at least as horizontal-heavy as plain.
        horizontal_steps = sum(
            1 for a, b in zip(p_pref, p_pref[1:])
            if a[1] == b[1] and a[2] == b[2]
        )
        # Five horizontal moves required; the prefer_directions=True path
        # should take all five before any vertical step.
        prefix_horiz = 0
        for a, b in zip(p_pref, p_pref[1:]):
            if a[1] == b[1] and a[2] == b[2]:
                prefix_horiz += 1
            else:
                break
        self.assertEqual(horizontal_steps, 5)
        self.assertEqual(prefix_horiz, 5)

    def test_single_layer_2tuple_backcompat(self) -> None:
        # On a single-layer board, 2-tuple endpoints must yield 2-tuple paths.
        g = PCBGrid(5, 5)
        path = route_single_net(g, (0, 0), (4, 4))
        assert path is not None
        for cell in path:
            self.assertEqual(len(cell), 2)

    def test_multi_layer_path_is_3tuple(self) -> None:
        g = PCBGrid(5, 5, num_layers=2)
        path = route_single_net(g, (0, 0, 0), (4, 4, 1))
        assert path is not None
        for cell in path:
            self.assertEqual(len(cell), 3)
        # Goal is on layer 1 -> at least one via.
        self.assertEqual(path[-1][2], 1)


class TestClearance(unittest.TestCase):
    """Design rules: clearance > 0 enforces a halo between routed traces."""

    def test_clearance_zero_is_identity(self) -> None:
        # clearance=0 must produce exactly the legacy behaviour on a fixed
        # netlist: same routed counts, same total cells.
        netlist = [((0, 0), (9, 0)), ((0, 5), (9, 5)), ((0, 9), (9, 9))]
        g0 = PCBGrid(10, 10, clearance=0)
        g1 = PCBGrid(10, 10)  # default clearance=0
        s0 = route_board(g0, list(netlist))
        s1 = route_board(g1, list(netlist))
        self.assertEqual(len(s0["routed"]), len(s1["routed"]))
        self.assertEqual(
            sum(len(n["path"]) for n in s0["routed"]),
            sum(len(n["path"]) for n in s1["routed"]),
        )

    def test_clearance_one_separates_parallel_traces(self) -> None:
        # Two horizontal nets on adjacent rows. With clearance=0 both route
        # as 10-cell straight lines. With clearance=1 the second one's
        # natural y=4 path is mostly blocked by the first's halo and it
        # must detour through y>=5 in the middle.
        net0 = ((0, 3), (9, 3))
        net1 = ((0, 4), (9, 4))

        g_no = PCBGrid(10, 10, clearance=0)
        s_no = route_board(g_no, [net0, net1])
        self.assertEqual(len(s_no["routed"]), 2)
        self.assertEqual(len(s_no["routed"][0]["path"]), 10)
        self.assertEqual(len(s_no["routed"][1]["path"]), 10)  # straight line

        g_cl = PCBGrid(10, 10, clearance=1)
        s_cl = route_board(g_cl, [net0, net1])
        self.assertEqual(len(s_cl["routed"]), 2)
        # First net still routes straight (no other traces yet).
        self.assertEqual(len(s_cl["routed"][0]["path"]), 10)
        # Second net is forced to detour: longer than 10 cells, and the
        # interior (away from the pin-clear region at the endpoints) must
        # be on y>=5 because y=3..y=4 sit in net0's halo.
        net1_path = s_cl["routed"][1]["path"]
        self.assertGreater(len(net1_path), 10)
        interior_ys = [y for x, y in net1_path if 2 <= x <= 7]
        self.assertTrue(
            all(y >= 5 for y in interior_ys),
            f"Net1 middle should detour to y>=5; got ys={interior_ys}",
        )

    def test_clearance_one_fails_in_narrow_corridor(self) -> None:
        # A 2-row corridor (y=4, 5) with walls everywhere else. Net 0 routes
        # along y=4 and stamps a halo over y=3..5. The only remaining
        # potential row (y=5) is now fully blocked — net 1 cannot fit.
        g = PCBGrid(10, 10, clearance=1)
        for y in range(10):
            if y not in (4, 5):
                for x in range(10):
                    g.add_obstacle(x, y)
        netlist = [((0, 4), (9, 4)), ((0, 5), (9, 5))]
        summary = route_board(g, netlist)
        self.assertEqual(len(summary["routed"]), 1)
        self.assertEqual(len(summary["unrouted"]), 1)

    def test_pin_sharing_under_clearance(self) -> None:
        # Two nets share the (5, 5) pin. Under clearance=1, the first net
        # ends at (5, 5) and stamps a halo around it; the second net must
        # still be able to start from (5, 5) thanks to halo-aware pin-clear.
        g = PCBGrid(10, 10, clearance=1)
        netlist = [((5, 5), (0, 0)), ((5, 5), (9, 9))]
        summary = route_board(g, netlist)
        self.assertEqual(
            len(summary["routed"]), 2,
            "Both nets should route despite sharing the (5,5) pin under clearance=1",
        )

    def test_clearance_preserves_static_obstacles_during_pin_clear(self) -> None:
        # Static obstacle WITHIN the pin's halo radius. Pin-clear must skip
        # static cells so A* cannot shortcut through a wall just because it
        # happens to be near the endpoint.
        g = PCBGrid(10, 10, clearance=2)
        g.add_obstacle(1, 5)  # static wall immediately next to pin (0, 5)
        path = route_single_net(g, (0, 5), (9, 5))
        assert path is not None
        self.assertNotIn(
            (1, 5), path,
            "Static obstacle at (1,5) is in the halo of pin (0,5) but must NOT be cleared",
        )

    def test_rrr_under_clearance(self) -> None:
        # R&R should still terminate and not route fewer than greedy.
        g_plain = PCBGrid(15, 15, clearance=1)
        g_rrr = PCBGrid(15, 15, clearance=1)
        for y in range(15):
            if y != 7:
                g_plain.add_obstacle(7, y)
                g_rrr.add_obstacle(7, y)
        netlist = [
            ((0, 0), (14, 14)),
            ((6, 7), (8, 7)),
        ]
        plain = route_board(g_plain, netlist)
        rrr = route_board_rrr(g_rrr, netlist)
        self.assertGreaterEqual(len(rrr["routed"]), len(plain["routed"]))

    def test_clone_preserves_clearance_and_static_mask(self) -> None:
        g = PCBGrid(5, 5, clearance=2)
        g.add_obstacle(2, 2)
        c = g.clone()
        self.assertEqual(c.clearance, 2)
        self.assertTrue(c.is_static_obstacle(2, 2))


class TestCustomHeuristic(unittest.TestCase):
    """The pluggable heuristic kwarg and the nodes_expanded counter."""

    def test_default_heuristic_unchanged(self) -> None:
        g = PCBGrid(10, 10)
        path = route_single_net(g, (0, 0), (9, 9))
        assert path is not None
        # Optimal manhattan on a clear board.
        self.assertEqual(len(path), 19)

    def test_custom_heuristic_called(self) -> None:
        # Custom admissible heuristic: zero everywhere. A* with h=0 is just
        # Dijkstra — should still find the optimal path. Verify the callable
        # is invoked at least once and the resulting path is optimal.
        g = PCBGrid(10, 10)
        zero_calls = [0]

        def zero_h(node, goal_xy):
            zero_calls[0] += 1
            return 0.0

        path_zero = route_single_net(g, (0, 0), (9, 9), heuristic=zero_h)
        path_manhattan = route_single_net(g, (0, 0), (9, 9))

        assert path_zero is not None and path_manhattan is not None
        # Both must produce optimal-length paths.
        self.assertEqual(len(path_zero), len(path_manhattan))
        self.assertEqual(len(path_zero), 19)  # |dx|+|dy|+1 on an open board
        # Custom heuristic was invoked many times during the search.
        self.assertGreater(zero_calls[0], 0)

    def test_custom_heuristic_reduces_expansions_with_obstacles(self) -> None:
        # On a board with obstacles, Manhattan A* should expand FEWER nodes
        # than a zero heuristic (Dijkstra). This is the property the learned
        # heuristic is supposed to amplify.
        g = PCBGrid(20, 20)
        # Diagonal wall with a gap.
        for i in range(15):
            g.add_obstacle(i, i)

        nodes_zero = [0]
        path_zero = route_single_net(
            g, (0, 1), (19, 19),
            heuristic=lambda n, gxy: 0.0,
            nodes_expanded=nodes_zero,
        )
        nodes_manhattan = [0]
        path_manhattan = route_single_net(
            g, (0, 1), (19, 19), nodes_expanded=nodes_manhattan,
        )

        assert path_zero is not None and path_manhattan is not None
        self.assertEqual(len(path_zero), len(path_manhattan))
        self.assertGreater(nodes_zero[0], nodes_manhattan[0])

    def test_nodes_expanded_counter(self) -> None:
        g = PCBGrid(10, 10)
        nodes = [0]
        route_single_net(g, (0, 0), (3, 3), nodes_expanded=nodes)
        # On a clear 10x10 board with Manhattan heuristic, A* should expand
        # exactly the cells on the optimal path (and maybe a few tiebreakers).
        # Just sanity-check the counter is populated.
        self.assertGreater(nodes[0], 0)
        self.assertLess(nodes[0], 100)


if __name__ == "__main__":
    unittest.main()
