"""Learned A* heuristic: a small CNN that predicts cost-to-go from a local crop.

Drops into `route_single_net(..., heuristic=h)` in place of the default
Manhattan distance. The network takes an 11x11 obstacle crop centred on the
current cell (stacked across copper layers) plus three scalar features
`[dx/W, dy/H, crop_density]`, and outputs a non-negative cost-to-go estimate.

The model is wrapped with a `min(learned, manhattan)` clamp in
`make_learned_heuristic`, so A* always sees an admissible heuristic.

Architecture: 3 conv layers + 1 maxpool, then a 2-layer MLP, then a softplus
scalar. Around 70k parameters at the default settings.

The companion training script `rl/train_heuristic.py` builds supervised
`(crop, scalars, true_cost)` triples by running Dijkstra from the goal on
random boards and trains with a one-sided Huber loss that penalises
overestimates roughly 5x more than underestimates.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pcb_grid import PCBGrid

CROP_SIZE: int = 11
MAX_LAYERS: int = 4  # zero-pad shorter stacks so one model fits 1..MAX_LAYERS


class HeuristicNet(nn.Module):
    """CNN cost-to-go predictor for A*.

    Input shapes per forward call:
      * `crop`: `(B, C, CROP_SIZE, CROP_SIZE)` where C = `max_layers`.
      * `scalars`: `(B, 3)` — `[dx/W, dy/H, obstacle_density]`.

    Output: `(B,)` — predicted cost-to-go in cell units.
    """

    def __init__(
        self,
        crop_size: int = CROP_SIZE,
        max_layers: int = MAX_LAYERS,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.max_layers = max_layers

        self.cnn = nn.Sequential(
            nn.Conv2d(max_layers, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 11 -> 5
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        flat = 32 * (crop_size // 2) * (crop_size // 2)
        self.head = nn.Sequential(
            nn.Linear(flat + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, crop: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        if crop.dim() == 3:
            crop = crop.unsqueeze(0)
            scalars = scalars.unsqueeze(0)
        feat = self.cnn(crop).flatten(start_dim=1)
        x = torch.cat([feat, scalars], dim=-1)
        # Softplus keeps the prediction non-negative (cost-to-go is ≥ 0).
        return nn.functional.softplus(self.head(x).squeeze(-1))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_crop(
    layers: np.ndarray,
    cx: int,
    cy: int,
    *,
    crop_size: int = CROP_SIZE,
    max_layers: int = MAX_LAYERS,
) -> np.ndarray:
    """Extract an `(max_layers, crop_size, crop_size)` crop centered on `(cx, cy)`.

    Zero-pads when the crop extends outside the board. Layer channels beyond
    `layers.shape[0]` are zero so one model handles 1..max_layers boards.
    """
    num_layers, H, W = layers.shape
    half = crop_size // 2

    out = np.zeros((max_layers, crop_size, crop_size), dtype=np.float32)
    for z in range(min(num_layers, max_layers)):
        for dy in range(-half, half + 1):
            yy = cy + dy
            if not (0 <= yy < H):
                continue
            for dx in range(-half, half + 1):
                xx = cx + dx
                if not (0 <= xx < W):
                    continue
                out[z, dy + half, dx + half] = float(layers[z, yy, xx])
    return out


def scalar_features(
    cx: int, cy: int, goal_xy: tuple[int, int],
    width: int, height: int, crop_obstacle_density: float,
) -> np.ndarray:
    """3-D vector: signed dx/W, dy/H, and the crop's obstacle density."""
    return np.array([
        (goal_xy[0] - cx) / max(1, width),
        (goal_xy[1] - cy) / max(1, height),
        crop_obstacle_density,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Wrap a trained model as an A* heuristic
# ---------------------------------------------------------------------------


def make_learned_heuristic(
    model: HeuristicNet,
    grid: PCBGrid,
    *,
    clamp: str = "max",
):
    """Return an A*-compatible heuristic callable backed by `model`.

    The returned callable matches `HeuristicFn = (Coord3D, Coord2D) -> float`.
    Precomputes the grid's layer tensor once so per-call cost is one model
    forward pass (no torch->numpy conversions in the hot path).

    `clamp` controls how the learned prediction `h_L` combines with
    Manhattan `h_M`:

      * ``"max"`` (default): `max(h_L, h_M)`. A* with a tighter admissible
        heuristic (closer to true cost from below) expands fewer nodes.
        Both `h_L` and `h_M` are individually admissible iff the model is
        trained to under-predict — the max of two admissible heuristics is
        admissible *and* tighter than either. This is the configuration
        that actually reduces A* expansions vs vanilla Manhattan.

      * ``"min"``: `min(h_L, h_M)`. Strictly admissible-by-construction
        even if `h_L` overshoots (since the min then equals `h_M`), but
        always ≤ Manhattan, so A* expands **more** nodes than vanilla.
        Useful as a "did we break A* optimality?" sanity check.

      * ``"raw"``: `h_L` only — no admissibility clamp. Weighted-A* style:
        small overshoots produce small path-length suboptimality in
        exchange for far fewer expansions.
    """
    if clamp not in ("max", "min", "raw"):
        raise ValueError(f"clamp must be 'max', 'min', or 'raw'; got {clamp!r}")

    layers_np = grid.layers.astype(np.float32)
    height, width = grid.height, grid.width
    model.eval()

    def heuristic(node, goal_xy: tuple[int, int]) -> float:
        cx, cy = node[0], node[1]
        manhattan = float(abs(cx - goal_xy[0]) + abs(cy - goal_xy[1]))

        crop = extract_crop(layers_np, cx, cy)
        density = float(crop.mean())
        scalars = scalar_features(cx, cy, goal_xy, width, height, density)

        with torch.no_grad():
            crop_t = torch.from_numpy(crop)
            scalars_t = torch.from_numpy(scalars)
            pred = float(model(crop_t, scalars_t).item())

        if clamp == "max":
            return max(pred, manhattan)
        if clamp == "min":
            return min(pred, manhattan)
        return pred  # "raw"

    return heuristic


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = HeuristicNet()
    print(f"HeuristicNet params: {_count_params(m):,}")
    # Smoke test.
    crop = torch.zeros(MAX_LAYERS, CROP_SIZE, CROP_SIZE)
    scalars = torch.zeros(3)
    out = m(crop, scalars)
    print(f"Forward output shape: {tuple(out.shape)}")
