"""PPO training for the net-ordering policy with the CNN actor-critic.

Phase 2 upgrade over `train.py` (REINFORCE + flat MLP):
  * CNN board encoder (better inductive bias for spatial obstacle patterns).
  * Value head (critic) sharing the net-feature encoder.
  * Generalized Advantage Estimation (GAE).
  * Clipped surrogate objective with multiple epochs per batch.

Run from the project root:
    python -m rl.train_ppo
"""

from __future__ import annotations

import collections
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rl.env import RoutingEnv                                    # noqa: E402
from rl.policy import CNNActorCritic, net_to_features            # noqa: E402
from rl.train import (                                            # noqa: E402
    BOARD_SIZE,
    N_NETS,
    random_board_and_netlist,
)


# --- PPO hyperparameters --------------------------------------------------
N_ITERATIONS = 250
EPISODES_PER_BATCH = 32
EPOCHS_PER_BATCH = 4
MINIBATCH_SIZE = 64
LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
GAMMA = 1.0          # episodes are short and undiscounted
GAE_LAMBDA = 0.95
GRAD_CLIP = 0.5
SEED = 42

WEIGHTS_PATH = os.path.join(_HERE, "policy_ppo.pt")


def collect_episode(policy, env, grid, netlist):
    """Roll out one episode under the current policy, storing transitions."""
    obs = env.reset(grid, netlist)
    transitions = []

    while obs["remaining"]:
        board_t = torch.tensor(obs["board"].flatten(), dtype=torch.float32)
        feats_t = torch.tensor(
            [net_to_features(p, BOARD_SIZE) for p in obs["remaining"]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            logits, value = policy(board_t, feats_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        transitions.append({
            "board": obs["board"].copy(),
            "remaining": list(obs["remaining"]),
            "action": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        })

        obs, reward, _ = env.step(int(action.item()))
        transitions[-1]["reward"] = float(reward)

    return transitions


def compute_gae(transitions, gamma: float, lam: float):
    """Add 'advantage' and 'return' fields to each transition (in place)."""
    last_gae = 0.0
    for t in reversed(range(len(transitions))):
        next_value = 0.0 if t == len(transitions) - 1 else transitions[t + 1]["value"]
        delta = transitions[t]["reward"] + gamma * next_value - transitions[t]["value"]
        last_gae = delta + gamma * lam * last_gae
        transitions[t]["advantage"] = last_gae
        transitions[t]["return"] = last_gae + transitions[t]["value"]
    return transitions


def ppo_update(policy, optimizer, transitions, epochs, minibatch_size):
    """Run PPO update epochs over the given transitions."""
    n = len(transitions)

    # Normalize advantages (across the whole batch) — standard PPO trick.
    advs = np.array([t["advantage"] for t in transitions], dtype=np.float32)
    adv_mean, adv_std = advs.mean(), advs.std() + 1e-8
    for t in transitions:
        t["advantage_norm"] = (t["advantage"] - adv_mean) / adv_std

    indices = list(range(n))
    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            batch_idxs = indices[start:start + minibatch_size]

            policy_losses = []
            value_losses = []
            entropies = []

            for idx in batch_idxs:
                tr = transitions[idx]
                board_t = torch.tensor(tr["board"].flatten(), dtype=torch.float32)
                feats_t = torch.tensor(
                    [net_to_features(p, BOARD_SIZE) for p in tr["remaining"]],
                    dtype=torch.float32,
                )

                logits, value = policy(board_t, feats_t)
                dist = torch.distributions.Categorical(logits=logits)
                action_t = torch.tensor(tr["action"])
                new_log_prob = dist.log_prob(action_t)
                old_log_prob = torch.tensor(tr["log_prob"])

                ratio = torch.exp(new_log_prob - old_log_prob)
                adv = tr["advantage_norm"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2)

                ret_t = torch.tensor(tr["return"], dtype=torch.float32)
                value_loss = (value - ret_t) ** 2

                entropy = dist.entropy()

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy_loss = -torch.stack(entropies).mean()

            total = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()


def train() -> None:
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    policy = CNNActorCritic(board_size=BOARD_SIZE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    env = RoutingEnv(BOARD_SIZE, BOARD_SIZE)

    rolling_returns: collections.deque = collections.deque(maxlen=200)

    for it in range(N_ITERATIONS):
        all_transitions = []
        episode_returns = []

        for _ in range(EPISODES_PER_BATCH):
            grid, netlist = random_board_and_netlist(rng)
            transitions = collect_episode(policy, env, grid, netlist)
            compute_gae(transitions, GAMMA, GAE_LAMBDA)

            episode_returns.append(sum(t["reward"] for t in transitions))
            all_transitions.extend(transitions)

        ppo_update(policy, optimizer, all_transitions,
                   EPOCHS_PER_BATCH, MINIBATCH_SIZE)

        rolling_returns.extend(episode_returns)

        if (it + 1) % 10 == 0:
            avg = float(np.mean(rolling_returns))
            print(f"iter {it + 1:4d}  "
                  f"avg return (last {len(rolling_returns)} ep): "
                  f"{avg:5.2f} / {N_NETS}")

    torch.save(policy.state_dict(), WEIGHTS_PATH)
    print(f"\nSaved PPO+CNN policy to {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
