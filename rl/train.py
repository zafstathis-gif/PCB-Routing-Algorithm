"""REINFORCE training for the net-ordering policy.

Trains a tiny MLP policy that, given the current board and the list of
remaining nets, picks which net to route next. Reward is +1 per successfully
routed net; the episode return equals the number of routed nets.

Variance reduction: a moving-average baseline is subtracted from the return
when computing the policy-gradient loss. An entropy bonus keeps exploration
alive in early training.

Run from the project root:
    python -m rl.train
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim

# Make the project root importable when run as `python rl/train.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pcb_grid import PCBGrid                                     # noqa: E402
from router import NetPair                                       # noqa: E402
from rl.env import RoutingEnv                                    # noqa: E402
from rl.policy import PolicyNet, net_to_features                 # noqa: E402


# --- Problem distribution -------------------------------------------------
BOARD_SIZE = 20
N_NETS = 8
N_OBSTACLES = 25

# --- Training hyperparameters --------------------------------------------
EPISODES = 4000
LR = 3e-4
ENTROPY_COEF = 0.01
BASELINE_ALPHA = 0.05
SEED = 42

WEIGHTS_PATH = os.path.join(_HERE, "policy.pt")


def random_board_and_netlist(rng: random.Random) -> tuple[PCBGrid, list[NetPair]]:
    """Sample a random board and a netlist of unique-pin pairs."""
    grid = PCBGrid(BOARD_SIZE, BOARD_SIZE)

    obstacles: set[tuple[int, int]] = set()
    while len(obstacles) < N_OBSTACLES:
        obstacles.add((rng.randint(0, BOARD_SIZE - 1),
                       rng.randint(0, BOARD_SIZE - 1)))
    for x, y in obstacles:
        grid.add_obstacle(x, y)

    pins: set[tuple[int, int]] = set()
    netlist: list[NetPair] = []
    attempts = 0
    while len(netlist) < N_NETS and attempts < 10_000:
        attempts += 1
        sx, sy = rng.randint(0, BOARD_SIZE - 1), rng.randint(0, BOARD_SIZE - 1)
        ex, ey = rng.randint(0, BOARD_SIZE - 1), rng.randint(0, BOARD_SIZE - 1)
        if (sx, sy) == (ex, ey):
            continue
        if not grid.is_valid(sx, sy) or not grid.is_valid(ex, ey):
            continue
        if (sx, sy) in pins or (ex, ey) in pins:
            continue
        pins.add((sx, sy))
        pins.add((ex, ey))
        netlist.append(((sx, sy), (ex, ey)))

    return grid, netlist


def run_episode(policy: PolicyNet, env: RoutingEnv,
                grid: PCBGrid, netlist: list[NetPair]
                ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
    """Roll out one episode under the current policy, sampling actions."""
    obs = env.reset(grid, netlist)
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[float] = []

    while obs["remaining"]:
        board = torch.tensor(obs["board"].flatten(), dtype=torch.float32)
        feats = torch.tensor(
            [net_to_features(p, BOARD_SIZE) for p in obs["remaining"]],
            dtype=torch.float32,
        )

        logits = policy(board, feats)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        obs, reward, _ = env.step(int(action.item()))
        rewards.append(reward)

    return log_probs, entropies, rewards


def train() -> None:
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    policy = PolicyNet(board_size=BOARD_SIZE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    env = RoutingEnv(BOARD_SIZE, BOARD_SIZE)

    baseline = 0.0
    rolling: list[float] = []

    for ep in range(EPISODES):
        grid, netlist = random_board_and_netlist(rng)
        log_probs, entropies, rewards = run_episode(policy, env, grid, netlist)

        total_reward = float(sum(rewards))
        advantage = total_reward - baseline
        baseline += BASELINE_ALPHA * (total_reward - baseline)

        # REINFORCE loss: -E[advantage * log pi(a|s)]  -  entropy_bonus.
        policy_loss = -advantage * torch.stack(log_probs).sum()
        entropy_bonus = ENTROPY_COEF * torch.stack(entropies).sum()
        loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rolling.append(total_reward)
        if (ep + 1) % 200 == 0:
            avg = float(np.mean(rolling[-200:]))
            print(f"ep {ep + 1:5d}  avg routed (last 200): {avg:5.2f} / {N_NETS}"
                  f"   baseline={baseline:5.2f}")

    torch.save(policy.state_dict(), WEIGHTS_PATH)
    print(f"\nSaved policy weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
