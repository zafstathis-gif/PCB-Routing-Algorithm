"""PPO training with the rip-up-aware action space (Phase 3).

Phase 3 upgrade over `train_ppo.py`:
  * Environment is `RoutingEnvRipup`. Each step the agent picks one of
    R + K legal actions:
        actions[0..R-1]    route remaining[i]
        actions[R..R+K-1]  rip up routed[j]
  * Policy is `CNNActorCriticRipup`, which scores variable-size action sets
    using a 7-d feature per action (the 7th dim is a route/rip-up indicator).
  * Rewards:  +1 per successful route, 0 per failed route, -1 per rip-up.
    Sum over an episode equals the final routed count, so maximizing return
    directly maximizes nets routed minus rip-up cost.

Run from the project root:
    python -m rl.train_ppo_ripup
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

from rl.env import RoutingEnvRipup  # noqa: E402
from rl.policy import CNNActorCriticRipup, action_to_features  # noqa: E402
from rl.train import (  # noqa: E402
    BOARD_SIZE,
    N_NETS,
    random_board_and_netlist,
)

N_ITERATIONS = 300
EPISODES_PER_BATCH = 32
EPOCHS_PER_BATCH = 4
MINIBATCH_SIZE = 64
LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02   # slightly higher than Phase 2: bigger action space → more exploration needed
GAMMA = 1.0
GAE_LAMBDA = 0.95
GRAD_CLIP = 0.5
SEED = 42

WEIGHTS_PATH = os.path.join(_HERE, "policy_ppo_ripup.pt")


def _action_features_for_obs(obs, board_size: int) -> list[list[float]]:
    """Build the 7-d feature row for each legal action this step."""
    feats: list[list[float]] = []
    for pair in obs["remaining"]:
        feats.append(action_to_features(pair, board_size, is_ripup=False))
    for net in obs["routed"]:
        feats.append(action_to_features(net["pair"], board_size, is_ripup=True))
    return feats


def collect_episode(policy, env, grid, netlist):
    obs = env.reset(grid, netlist)
    transitions = []

    while True:
        feats = _action_features_for_obs(obs, BOARD_SIZE)
        if not feats:
            break  # no legal action

        board_t = torch.tensor(obs["board"].flatten(), dtype=torch.float32)
        feats_t = torch.tensor(feats, dtype=torch.float32)

        with torch.no_grad():
            logits, value = policy(board_t, feats_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        transitions.append({
            "board": obs["board"].copy(),
            "action_feats": feats,           # list of 7-d rows
            "action": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        })

        obs, reward, done = env.step(int(action.item()))
        transitions[-1]["reward"] = float(reward)
        if done:
            break

    return transitions


def compute_gae(transitions, gamma: float, lam: float):
    last_gae = 0.0
    for t in reversed(range(len(transitions))):
        next_value = 0.0 if t == len(transitions) - 1 else transitions[t + 1]["value"]
        delta = transitions[t]["reward"] + gamma * next_value - transitions[t]["value"]
        last_gae = delta + gamma * lam * last_gae
        transitions[t]["advantage"] = last_gae
        transitions[t]["return"] = last_gae + transitions[t]["value"]
    return transitions


def ppo_update(policy, optimizer, transitions, epochs, minibatch_size):
    n = len(transitions)
    if n == 0:
        return

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
                feats_t = torch.tensor(tr["action_feats"], dtype=torch.float32)

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

    policy = CNNActorCriticRipup(board_size=BOARD_SIZE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    env = RoutingEnvRipup(BOARD_SIZE, BOARD_SIZE)

    rolling_routed: collections.deque = collections.deque(maxlen=200)
    rolling_ripups: collections.deque = collections.deque(maxlen=200)

    for it in range(N_ITERATIONS):
        all_transitions = []
        for _ in range(EPISODES_PER_BATCH):
            grid, netlist = random_board_and_netlist(rng)
            transitions = collect_episode(policy, env, grid, netlist)
            compute_gae(transitions, GAMMA, GAE_LAMBDA)

            rolling_routed.append(len(env.routed))
            rolling_ripups.append(env.ripup_count)
            all_transitions.extend(transitions)

        ppo_update(policy, optimizer, all_transitions,
                   EPOCHS_PER_BATCH, MINIBATCH_SIZE)

        if (it + 1) % 10 == 0:
            avg_routed = float(np.mean(rolling_routed))
            avg_ripups = float(np.mean(rolling_ripups))
            print(f"iter {it + 1:4d}  "
                  f"routed (last {len(rolling_routed)} ep): "
                  f"{avg_routed:5.2f} / {N_NETS}   "
                  f"ripups: {avg_ripups:4.2f}")

    torch.save(policy.state_dict(), WEIGHTS_PATH)
    print(f"\nSaved PPO+CNN+ripup policy to {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
