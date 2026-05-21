"""Evaluate the trained net-ordering policy against the heuristic baselines.

Generates a held-out set of random boards (different RNG seed than training)
and reports the mean number of nets each method routes per board.

Run from the project root:
    python -m rl.evaluate
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pcb_grid import PCBGrid  # noqa: E402
from rl.env import RoutingEnv, RoutingEnvRipup  # noqa: E402
from rl.policy import (  # noqa: E402
    CNNActorCritic,
    CNNActorCriticRipup,
    PolicyNet,
    action_to_features,
    net_to_features,
)
from rl.train import (  # noqa: E402
    BOARD_SIZE,
    N_NETS,
    random_board_and_netlist,
)
from rl.train import WEIGHTS_PATH as REINFORCE_WEIGHTS  # noqa: E402
from rl.train_ppo import WEIGHTS_PATH as PPO_WEIGHTS  # noqa: E402
from rl.train_ppo_ripup import WEIGHTS_PATH as PPO_RIPUP_WEIGHTS  # noqa: E402
from router import (  # noqa: E402
    ALL_STRATEGIES,
    NetPair,
    route_board,
    route_board_best_of,
    route_board_rrr,
)

N_TEST = 200
EVAL_SEED = 9999


def evaluate_strategy(
    strategy, boards: list[tuple[PCBGrid, list[NetPair]]],
    use_rrr: bool = False,
) -> tuple[float, float]:
    counts = []
    runner = route_board_rrr if use_rrr else route_board
    for grid, netlist in boards:
        g = grid.clone()
        s = runner(g, netlist, sort_strategy=strategy)
        counts.append(len(s["routed"]))
    return float(np.mean(counts)), float(np.std(counts))


def evaluate_best_of(
    boards: list[tuple[PCBGrid, list[NetPair]]],
    use_rrr: bool = False,
) -> tuple[float, float]:
    counts = []
    for grid, netlist in boards:
        g = grid.clone()
        s = route_board_best_of(g, netlist, use_rrr=use_rrr)
        counts.append(len(s["routed"]))
    return float(np.mean(counts)), float(np.std(counts))


def evaluate_policy_reinforce(
    policy: PolicyNet,
    env: RoutingEnv,
    boards: list[tuple[PCBGrid, list[NetPair]]],
) -> tuple[float, float]:
    counts = []
    with torch.no_grad():
        for grid, netlist in boards:
            obs = env.reset(grid, netlist)
            while obs["remaining"]:
                board = torch.tensor(obs["board"].flatten(), dtype=torch.float32)
                feats = torch.tensor(
                    [net_to_features(p, BOARD_SIZE) for p in obs["remaining"]],
                    dtype=torch.float32,
                )
                logits = policy(board, feats)
                action = int(torch.argmax(logits).item())
                obs, _, _ = env.step(action)
            counts.append(len(env.routed))
    return float(np.mean(counts)), float(np.std(counts))


def evaluate_policy_ppo(
    policy: CNNActorCritic,
    env: RoutingEnv,
    boards: list[tuple[PCBGrid, list[NetPair]]],
) -> tuple[float, float]:
    counts = []
    with torch.no_grad():
        for grid, netlist in boards:
            obs = env.reset(grid, netlist)
            while obs["remaining"]:
                board = torch.tensor(obs["board"].flatten(), dtype=torch.float32)
                feats = torch.tensor(
                    [net_to_features(p, BOARD_SIZE) for p in obs["remaining"]],
                    dtype=torch.float32,
                )
                logits, _ = policy(board, feats)
                action = int(torch.argmax(logits).item())
                obs, _, _ = env.step(action)
            counts.append(len(env.routed))
    return float(np.mean(counts)), float(np.std(counts))


def evaluate_policy_ppo_ripup(
    policy: CNNActorCriticRipup,
    env: RoutingEnvRipup,
    boards: list[tuple[PCBGrid, list[NetPair]]],
) -> tuple[float, float, float]:
    counts = []
    ripups = []
    with torch.no_grad():
        for grid, netlist in boards:
            obs = env.reset(grid, netlist)
            while True:
                feats: list[list[float]] = []
                for p in obs["remaining"]:
                    feats.append(action_to_features(p, BOARD_SIZE, is_ripup=False))
                for n in obs["routed"]:
                    feats.append(action_to_features(n["pair"], BOARD_SIZE, is_ripup=True))
                if not feats:
                    break
                board = torch.tensor(obs["board"].flatten(), dtype=torch.float32)
                feats_t = torch.tensor(feats, dtype=torch.float32)
                logits, _ = policy(board, feats_t)
                action = int(torch.argmax(logits).item())
                obs, _, done = env.step(action)
                if done:
                    break
            counts.append(len(env.routed))
            ripups.append(env.ripup_count)
    return float(np.mean(counts)), float(np.std(counts)), float(np.mean(ripups))


def main() -> None:
    rng = random.Random(EVAL_SEED)
    boards = [random_board_and_netlist(rng) for _ in range(N_TEST)]

    print(f"Evaluating on {N_TEST} held-out random boards "
          f"({BOARD_SIZE}x{BOARD_SIZE}, {N_NETS} nets each)\n")
    print(f"{'method':<32} {'mean routed':<14} {'std':<8}")
    print("-" * 56)

    print("--- greedy (no R&R) ---")
    for s in ALL_STRATEGIES:
        label = "user-order" if s is None else s
        mean, std = evaluate_strategy(s, boards, use_rrr=False)
        print(f"  {label:<30} {mean:<14.3f} {std:<8.3f}")
    mean, std = evaluate_best_of(boards, use_rrr=False)
    print(f"  {'best_of':<30} {mean:<14.3f} {std:<8.3f}")

    print("--- with rip-up-and-reroute ---")
    for s in ALL_STRATEGIES:
        label = "user-order" if s is None else s
        mean, std = evaluate_strategy(s, boards, use_rrr=True)
        print(f"  {label:<30} {mean:<14.3f} {std:<8.3f}")
    mean, std = evaluate_best_of(boards, use_rrr=True)
    print(f"  {'best_of (oracle)':<30} {mean:<14.3f} {std:<8.3f}")

    print("--- learned policies ---")

    if os.path.exists(REINFORCE_WEIGHTS):
        policy_r = PolicyNet(board_size=BOARD_SIZE)
        policy_r.load_state_dict(torch.load(REINFORCE_WEIGHTS, weights_only=True))
        policy_r.eval()
        env = RoutingEnv(BOARD_SIZE, BOARD_SIZE)
        mean, std = evaluate_policy_reinforce(policy_r, env, boards)
        print(f"  {'REINFORCE + flat MLP':<30} {mean:<14.3f} {std:<8.3f}")
    else:
        print(f"  REINFORCE weights not found at {REINFORCE_WEIGHTS}")

    if os.path.exists(PPO_WEIGHTS):
        policy_p = CNNActorCritic(board_size=BOARD_SIZE)
        policy_p.load_state_dict(torch.load(PPO_WEIGHTS, weights_only=True))
        policy_p.eval()
        env = RoutingEnv(BOARD_SIZE, BOARD_SIZE)
        mean, std = evaluate_policy_ppo(policy_p, env, boards)
        print(f"  {'PPO + CNN':<30} {mean:<14.3f} {std:<8.3f}")
    else:
        print(f"  PPO weights not found at {PPO_WEIGHTS}")

    if os.path.exists(PPO_RIPUP_WEIGHTS):
        policy_pr = CNNActorCriticRipup(board_size=BOARD_SIZE)
        policy_pr.load_state_dict(torch.load(PPO_RIPUP_WEIGHTS, weights_only=True))
        policy_pr.eval()
        env_r = RoutingEnvRipup(BOARD_SIZE, BOARD_SIZE)
        mean, std, mean_ripups = evaluate_policy_ppo_ripup(policy_pr, env_r, boards)
        print(f"  {'PPO + CNN + ripup':<30} {mean:<14.3f} {std:<8.3f}"
              f"   (avg ripups: {mean_ripups:.2f})")
    else:
        print(f"  PPO+ripup weights not found at {PPO_RIPUP_WEIGHTS}")


if __name__ == "__main__":
    main()
