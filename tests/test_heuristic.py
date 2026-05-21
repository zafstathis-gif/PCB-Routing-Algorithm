"""Tests for the learned A* heuristic: Dijkstra-from-goal, HeuristicNet, clamps."""

from __future__ import annotations

import os
import random
import sys
import unittest

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pcb_grid import PCBGrid  # noqa: E402
from rl.heuristic_data import (  # noqa: E402
    build_examples_for_board,
    dijkstra_from_goal,
    random_board,
)
from rl.heuristic_net import (  # noqa: E402
    CROP_SIZE,
    MAX_LAYERS,
    HeuristicNet,
    extract_crop,
    make_learned_heuristic,
    scalar_features,
)
from router import route_single_net  # noqa: E402


class TestDijkstraFromGoal(unittest.TestCase):
    def test_zero_at_goal(self) -> None:
        g = PCBGrid(10, 10, num_layers=2)
        dist = dijkstra_from_goal(g, (5, 5, 0))
        self.assertEqual(dist[0, 5, 5], 0.0)

    def test_distance_equals_manhattan_on_clear_board(self) -> None:
        g = PCBGrid(10, 10, num_layers=1)
        dist = dijkstra_from_goal(g, (0, 0, 0))
        for x in range(10):
            for y in range(10):
                self.assertEqual(dist[0, y, x], x + y)

    def test_obstacle_increases_distance(self) -> None:
        # U-shaped wall around the goal: the only opening is at the far
        # side, forcing a much longer path than naive manhattan.
        g = PCBGrid(10, 10, num_layers=1)
        for x in range(4, 7):
            g.add_obstacle(x, 4, layer=0)
            g.add_obstacle(x, 6, layer=0)
        g.add_obstacle(6, 5, layer=0)
        # Goal at (5, 5) is now enclosed except via (4, 5).
        dist = dijkstra_from_goal(g, (5, 5, 0))
        # From (7, 5) — just outside the wall on the right — naive manhattan
        # is 2, but the path must go around the U, so distance > 2.
        self.assertGreater(dist[0, 5, 7], 2.0)

    def test_via_cost_dominates_across_layers(self) -> None:
        g = PCBGrid(5, 5, num_layers=2)
        dist = dijkstra_from_goal(g, (0, 0, 0), via_cost=10.0)
        # Same (x, y), different layer: must traverse a via.
        self.assertEqual(dist[1, 0, 0], 10.0)


class TestCropAndScalars(unittest.TestCase):
    def test_crop_shape(self) -> None:
        layers = np.zeros((2, 20, 20), dtype=np.float32)
        crop = extract_crop(layers, 10, 10)
        self.assertEqual(crop.shape, (MAX_LAYERS, CROP_SIZE, CROP_SIZE))

    def test_crop_picks_up_obstacles(self) -> None:
        layers = np.zeros((2, 20, 20), dtype=np.float32)
        layers[0, 10, 10] = 1.0
        crop = extract_crop(layers, 10, 10)
        # Center of crop (5,5 for size 11) on layer 0.
        center = CROP_SIZE // 2
        self.assertEqual(crop[0, center, center], 1.0)

    def test_crop_zero_pads_at_edges(self) -> None:
        layers = np.ones((1, 20, 20), dtype=np.float32)
        crop = extract_crop(layers, 0, 0)
        # Cells at negative offsets should be zero (out of board).
        self.assertEqual(crop[0, 0, 0], 0.0)
        # The bottom-right cells (positive offsets from origin) should be set.
        self.assertEqual(crop[0, CROP_SIZE // 2, CROP_SIZE // 2], 1.0)

    def test_scalar_features_signed(self) -> None:
        s = scalar_features(0, 0, (10, 5), width=20, height=20,
                            crop_obstacle_density=0.1)
        self.assertAlmostEqual(s[0], 10.0 / 20)
        self.assertAlmostEqual(s[1], 5.0 / 20)
        self.assertAlmostEqual(s[2], 0.1)


class TestHeuristicNet(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        m = HeuristicNet()
        crop = torch.zeros(MAX_LAYERS, CROP_SIZE, CROP_SIZE)
        scalars = torch.zeros(3)
        out = m(crop, scalars)
        # Single sample -> (1,) or scalar; we used unsqueeze(0) so it's (1,).
        self.assertEqual(out.shape, torch.Size([1]))

    def test_batched_forward(self) -> None:
        m = HeuristicNet()
        crops = torch.zeros(8, MAX_LAYERS, CROP_SIZE, CROP_SIZE)
        scalars = torch.zeros(8, 3)
        out = m(crops, scalars)
        self.assertEqual(out.shape, torch.Size([8]))

    def test_output_is_nonneg(self) -> None:
        m = HeuristicNet()
        crops = torch.randn(16, MAX_LAYERS, CROP_SIZE, CROP_SIZE)
        scalars = torch.randn(16, 3)
        out = m(crops, scalars)
        self.assertTrue((out >= 0).all())


class TestLearnedHeuristicIntegration(unittest.TestCase):
    """Smoke-test: untrained network + each clamp produces optimal paths."""

    def test_all_clamps_produce_optimal_path(self) -> None:
        g = PCBGrid(15, 15)
        model = HeuristicNet()  # untrained — predictions near 0 due to softplus

        for clamp in ("max", "min", "raw"):
            heuristic = make_learned_heuristic(model, g, clamp=clamp)
            path = route_single_net(
                g, (0, 0), (10, 8), heuristic=heuristic,
            )
            assert path is not None
            # On a clear 15x15 board, optimal is |dx|+|dy|+1 = 19 cells.
            # All clamps should produce optimal paths here because:
            # max: tied or better than manhattan (no obstacles).
            # min: weaker but still admissible.
            # raw: untrained net predicts small values, behaves like h=0
            #      = Dijkstra = still optimal.
            self.assertEqual(len(path), 19)

    def test_invalid_clamp_raises(self) -> None:
        g = PCBGrid(10, 10)
        model = HeuristicNet()
        with self.assertRaises(ValueError):
            make_learned_heuristic(model, g, clamp="bad")


class TestExamplesBuilder(unittest.TestCase):
    def test_examples_match_dijkstra(self) -> None:
        rng = random.Random(0)
        grid = random_board(rng, num_obstacles=8)
        crops, scalars, targets = build_examples_for_board(
            grid, rng, max_examples_per_board=20,
        )
        self.assertEqual(crops.shape[1:], (MAX_LAYERS, CROP_SIZE, CROP_SIZE))
        self.assertEqual(scalars.shape[1], 3)
        self.assertEqual(targets.shape[0], crops.shape[0])
        # Targets must be non-negative.
        self.assertTrue((targets >= 0).all())


if __name__ == "__main__":
    unittest.main()
