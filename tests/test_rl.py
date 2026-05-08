"""Tests for the RL components: envs and policy networks (forward shapes only)."""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pcb_grid import PCBGrid                                     # noqa: E402
from rl.env import RoutingEnv, RoutingEnvRipup                   # noqa: E402
from rl.policy import (                                          # noqa: E402
    CNNActorCritic,
    CNNActorCriticRipup,
    PolicyNet,
    action_to_features,
    net_to_features,
)


class TestRoutingEnv(unittest.TestCase):
    def test_episode_terminates(self) -> None:
        env = RoutingEnv(10, 10)
        grid = PCBGrid(10, 10)
        netlist = [((0, 0), (9, 9)), ((0, 9), (9, 0))]
        obs = env.reset(grid, netlist)
        self.assertEqual(len(obs["remaining"]), 2)

        steps = 0
        while obs["remaining"] and steps < 20:
            obs, _, done = env.step(0)
            steps += 1
            if done:
                break
        self.assertEqual(len(obs["remaining"]), 0)


class TestRoutingEnvRipup(unittest.TestCase):
    def test_basic_route_and_done(self) -> None:
        env = RoutingEnvRipup(8, 8)
        grid = PCBGrid(8, 8)
        netlist = [((0, 0), (7, 0)), ((0, 7), (7, 7))]
        obs = env.reset(grid, netlist)
        self.assertEqual(len(obs["remaining"]), 2)
        self.assertEqual(len(obs["routed"]), 0)
        self.assertEqual(env.num_actions(), 2)

        # Route both nets; episode should terminate naturally.
        obs, r1, done = env.step(0)
        self.assertEqual(r1, 1.0)
        self.assertEqual(len(env.routed), 1)
        self.assertFalse(done)

        obs, r2, done = env.step(0)
        self.assertEqual(r2, 1.0)
        self.assertEqual(len(env.routed), 2)
        self.assertTrue(done)

    def test_ripup_returns_pair_to_remaining(self) -> None:
        env = RoutingEnvRipup(8, 8)
        grid = PCBGrid(8, 8)
        netlist = [((0, 0), (7, 0))]
        env.reset(grid, netlist)
        env.step(0)  # route it
        self.assertEqual(len(env.routed), 1)
        self.assertEqual(len(env.remaining), 0)

        # Action 0 is now the rip-up of routed[0] (since R=0, K=1, action 0 is ripup).
        obs, reward, _ = env.step(0)
        self.assertEqual(reward, -1.0)
        self.assertEqual(len(env.routed), 0)
        self.assertEqual(len(env.remaining), 1)
        self.assertEqual(env.ripup_count, 1)

    def test_ripup_frees_cells(self) -> None:
        # After ripping up, the cells where the trace was should be free
        # (assuming they weren't static obstacles).
        env = RoutingEnvRipup(6, 6)
        grid = PCBGrid(6, 6)
        netlist = [((0, 3), (5, 3))]
        env.reset(grid, netlist)
        env.step(0)
        path_cells = set(env.routed[0]["path"])

        env.step(0)  # rip up
        for x, y in path_cells:
            self.assertEqual(env.grid.grid[y, x], 0,
                             f"Cell ({x},{y}) should be free after rip-up")

    def test_action_index_out_of_range(self) -> None:
        env = RoutingEnvRipup(6, 6)
        env.reset(PCBGrid(6, 6), [((0, 0), (5, 5))])
        with self.assertRaises(IndexError):
            env.step(99)

    def test_step_budget_terminates(self) -> None:
        # With max_steps=1, the episode ends after exactly one action.
        env = RoutingEnvRipup(8, 8, max_steps=1)
        env.reset(PCBGrid(8, 8), [((0, 0), (7, 7)), ((0, 7), (7, 0))])
        _, _, done = env.step(0)
        self.assertTrue(done)

    def test_reward_sum_equals_final_routed_count(self) -> None:
        # The dense reward (route +1, fail 0, ripup -1) telescopes so that the
        # cumulative sum equals the number of currently-routed nets at any time.
        env = RoutingEnvRipup(8, 8)
        env.reset(PCBGrid(8, 8), [((0, 0), (7, 0)), ((0, 7), (7, 7))])
        cumulative = 0.0
        for action_choice in (0, 0, 0):  # route, route, ripup
            _, r, done = env.step(action_choice)
            cumulative += r
            self.assertEqual(cumulative, len(env.routed))
            if done:
                break


class TestPolicyShapes(unittest.TestCase):
    def test_policynet_logits_shape(self) -> None:
        policy = PolicyNet(board_size=20)
        board = torch.zeros(400)
        feats = torch.tensor(
            [net_to_features(((1, 1), (5, 5)), 20),
             net_to_features(((2, 2), (8, 8)), 20),
             net_to_features(((3, 3), (10, 10)), 20)],
            dtype=torch.float32,
        )
        logits = policy(board, feats)
        self.assertEqual(logits.shape, torch.Size([3]))

    def test_cnn_actor_critic_shapes(self) -> None:
        policy = CNNActorCritic(board_size=20)
        board = torch.zeros(400)
        feats = torch.tensor(
            [net_to_features(((1, 1), (5, 5)), 20),
             net_to_features(((2, 2), (8, 8)), 20)],
            dtype=torch.float32,
        )
        logits, value = policy(board, feats)
        self.assertEqual(logits.shape, torch.Size([2]))
        self.assertEqual(value.shape, torch.Size([]))  # scalar

    def test_cnn_actor_critic_ripup_shapes(self) -> None:
        policy = CNNActorCriticRipup(board_size=20)
        board = torch.zeros(400)
        feats = torch.tensor(
            [action_to_features(((1, 1), (5, 5)), 20, is_ripup=False),
             action_to_features(((2, 2), (8, 8)), 20, is_ripup=False),
             action_to_features(((3, 3), (10, 10)), 20, is_ripup=True)],
            dtype=torch.float32,
        )
        logits, value = policy(board, feats)
        self.assertEqual(logits.shape, torch.Size([3]))
        self.assertEqual(value.shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()
