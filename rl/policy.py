"""Policy network for net-ordering.

For each remaining net, the policy produces a scalar logit conditioned on
the current board state. A softmax over those logits gives a categorical
distribution over which net to route next.

Architecture (intentionally small — the task is small):

    flat board (H*W,)  ->  MLP  ->  board_emb (32)
    per-net features (6,)  ->  MLP  ->  net_emb (32)
    [board_emb, net_emb] -> MLP -> scalar logit per net
"""

from __future__ import annotations

import torch
import torch.nn as nn

from router import NetPair, _endpoint_xy


def net_to_features(pair: NetPair, board_size: int) -> list[float]:
    """Hand-engineered features for a candidate net, normalized to [0, 1]."""
    sx, sy = _endpoint_xy(pair[0])
    ex, ey = _endpoint_xy(pair[1])
    s = float(board_size)
    manhattan = abs(sx - ex) + abs(sy - ey)
    bbox_w = abs(sx - ex) + 1
    bbox_h = abs(sy - ey) + 1
    return [
        sx / s, sy / s, ex / s, ey / s,
        manhattan / (2.0 * s),
        (bbox_w * bbox_h) / (s * s),
    ]


def action_to_features(pair: NetPair, board_size: int, is_ripup: bool) -> list[float]:
    """Augmented feature vector for the rip-up-aware action space.

    Adds a single binary indicator on top of `net_to_features`:
        0.0 → "route this remaining net"
        1.0 → "rip up this routed net"
    """
    return net_to_features(pair, board_size) + [1.0 if is_ripup else 0.0]


NET_FEATURE_DIM = 6
ACTION_FEATURE_DIM = 7


class PolicyNet(nn.Module):
    """Flat-MLP policy used by REINFORCE (Phase 1). Kept for backwards compatibility."""

    def __init__(
        self,
        board_size: int = 20,
        board_emb: int = 32,
        net_emb: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.num_layers = num_layers
        flat = num_layers * board_size * board_size

        self.board_encoder = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, board_emb),
            nn.ReLU(),
        )
        self.net_encoder = nn.Sequential(
            nn.Linear(NET_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, net_emb),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(board_emb + net_emb, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, board: torch.Tensor, net_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board:     (H*W,) flat board tensor, float32.
            net_feats: (N, 6) features for N candidate nets.

        Returns:
            (N,) logits over candidate nets.
        """
        b_emb = self.board_encoder(board)                       # (board_emb,)
        n_emb = self.net_encoder(net_feats)                     # (N, net_emb)
        b_emb_rep = b_emb.unsqueeze(0).expand(net_feats.size(0), -1)
        combined = torch.cat([b_emb_rep, n_emb], dim=-1)        # (N, b+n)
        return self.scorer(combined).squeeze(-1)                # (N,)


class CNNActorCritic(nn.Module):
    """CNN encoder + actor head + value (critic) head, used by PPO.

    Architecture:
        board (H, W) ─ Conv-Conv-MaxPool-Conv-MaxPool ─ flatten ─ Linear ─ board_emb
        net features (N, 6) ─ MLP ─ net_emb (N, e)

    Actor:
        For each net i:  [board_emb, net_emb_i]  ─ MLP ─ scalar logit
        softmax over the N logits gives the action distribution.

    Critic:
        Sum-pools net embeddings (sum, not mean, so the count is preserved as a signal).
        [board_emb, sum(net_emb)] ─ MLP ─ scalar V(s).
    """

    def __init__(
        self,
        board_size: int = 20,
        board_emb_dim: int = 64,
        net_emb_dim: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.board_emb_dim = board_emb_dim
        self.net_emb_dim = net_emb_dim
        self.num_layers = num_layers

        # Two MaxPool(2)s reduce 20x20 -> 5x5; assumes board_size is divisible by 4.
        assert board_size % 4 == 0, "board_size must be divisible by 4 for the CNN"
        self.cnn = nn.Sequential(
            nn.Conv2d(num_layers, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat_size = 32 * (board_size // 4) * (board_size // 4)
        self.board_proj = nn.Sequential(
            nn.Linear(flat_size, board_emb_dim),
            nn.ReLU(),
        )

        self.net_encoder = nn.Sequential(
            nn.Linear(NET_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, net_emb_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(board_emb_dim + net_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(board_emb_dim + net_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_board(self, board: torch.Tensor) -> torch.Tensor:
        """Encode a board into a fixed-size embedding.

        Accepted input shapes:
          * (H*W,)            -> single layer, flat (legacy)
          * (H, W)            -> single layer, 2D
          * (C, H, W)         -> multi-layer with C channels
        Returns (board_emb_dim,).
        """
        if board.dim() == 1:
            board = board.view(self.num_layers, self.board_size, self.board_size)
        elif board.dim() == 2:
            board = board.unsqueeze(0)  # (1, H, W)
        # board is now (C, H, W); add batch dim.
        x = board.unsqueeze(0)  # (1, C, H, W)
        x = self.cnn(x).flatten(start_dim=1)
        return self.board_proj(x).squeeze(0)

    def forward(
        self, board: torch.Tensor, net_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board:     (H, W) or (H*W,) tensor.
            net_feats: (N, 6) features for N candidate nets.

        Returns:
            logits: (N,) over candidate actions.
            value:  scalar V(s).
        """
        b_emb = self.encode_board(board)                          # (B,)
        n_emb = self.net_encoder(net_feats)                       # (N, E)

        b_rep = b_emb.unsqueeze(0).expand(net_feats.size(0), -1)  # (N, B)
        actor_in = torch.cat([b_rep, n_emb], dim=-1)              # (N, B+E)
        logits = self.actor(actor_in).squeeze(-1)                 # (N,)

        n_emb_pool = n_emb.sum(dim=0)                             # (E,) — encodes count
        critic_in = torch.cat([b_emb, n_emb_pool], dim=-1)        # (B+E,)
        value = self.critic(critic_in).squeeze(-1)                # scalar

        return logits, value


class CNNActorCriticRipup(nn.Module):
    """Phase 3 actor-critic: same CNN trunk as `CNNActorCritic`, but the actor
    scores a *variable* action set (route candidates + rip-up candidates).

    Each action carries a 7-d feature vector built by `action_to_features`,
    where the 7th feature is a binary route/rip-up indicator. The actor is
    shared across both action types — the rip-up indicator is the only thing
    that lets it specialize.
    """

    def __init__(
        self,
        board_size: int = 20,
        board_emb_dim: int = 64,
        action_emb_dim: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.board_emb_dim = board_emb_dim
        self.action_emb_dim = action_emb_dim
        self.num_layers = num_layers

        assert board_size % 4 == 0, "board_size must be divisible by 4 for the CNN"
        self.cnn = nn.Sequential(
            nn.Conv2d(num_layers, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat_size = 32 * (board_size // 4) * (board_size // 4)
        self.board_proj = nn.Sequential(
            nn.Linear(flat_size, board_emb_dim),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, action_emb_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(board_emb_dim + action_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(board_emb_dim + action_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_board(self, board: torch.Tensor) -> torch.Tensor:
        if board.dim() == 1:
            board = board.view(self.num_layers, self.board_size, self.board_size)
        elif board.dim() == 2:
            board = board.unsqueeze(0)
        x = board.unsqueeze(0)  # (1, C, H, W)
        x = self.cnn(x).flatten(start_dim=1)
        return self.board_proj(x).squeeze(0)

    def forward(
        self, board: torch.Tensor, action_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board:        (H, W) or (H*W,) tensor.
            action_feats: (M, 7) features for M candidate actions
                          (route candidates first, then rip-up candidates).

        Returns:
            logits: (M,) over candidate actions.
            value:  scalar V(s).
        """
        b_emb = self.encode_board(board)                          # (B,)
        a_emb = self.action_encoder(action_feats)                 # (M, E)

        b_rep = b_emb.unsqueeze(0).expand(action_feats.size(0), -1)
        actor_in = torch.cat([b_rep, a_emb], dim=-1)
        logits = self.actor(actor_in).squeeze(-1)

        a_emb_pool = a_emb.sum(dim=0)
        critic_in = torch.cat([b_emb, a_emb_pool], dim=-1)
        value = self.critic(critic_in).squeeze(-1)

        return logits, value
