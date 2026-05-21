"""Generate supervised training data for the learned A* heuristic.

For each random board we pick a random goal cell, run one Dijkstra pass
**from the goal** (single-source shortest paths), and harvest the resulting
distance map. Every reachable cell `c` then provides a training example:

    (crop_around_c, scalar_features_for_c, true_cost_to_go = dist[c])

One Dijkstra pass per board yields up to `H*W` examples — far more
efficient than running A* once per source-goal pair.

Run standalone to materialize a `.npz` cache:

    python -m rl.heuristic_data --boards 5000 --out rl/heuristic_data.npz
"""

from __future__ import annotations

import argparse
import heapq
import os
import random
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pcb_grid import PCBGrid  # noqa: E402
from rl.heuristic_net import CROP_SIZE, MAX_LAYERS, extract_crop, scalar_features  # noqa: E402

# Same defaults as the existing RL training pipeline.
BOARD_SIZE = 20
NUM_LAYERS = 2
NUM_OBSTACLES = 18
VIA_COST = 10.0


# ---------------------------------------------------------------------------
# Random board generator
# ---------------------------------------------------------------------------


def random_board(
    rng: random.Random,
    num_obstacles: int = NUM_OBSTACLES,
) -> PCBGrid:
    """Random 2-layer board with `num_obstacles` static obstacles on layer 0.

    Default (`NUM_OBSTACLES=18`) matches the training distribution; raising
    it produces denser, harder boards where the learned heuristic has more
    room to help.
    """
    grid = PCBGrid(BOARD_SIZE, BOARD_SIZE, num_layers=NUM_LAYERS)
    placed = 0
    while placed < num_obstacles:
        x = rng.randint(0, BOARD_SIZE - 1)
        y = rng.randint(0, BOARD_SIZE - 1)
        if grid.is_valid(x, y, 0):
            grid.add_obstacle(x, y, layer=0)
            placed += 1
    return grid


# ---------------------------------------------------------------------------
# Dijkstra from a goal cell (3D, with via cost)
# ---------------------------------------------------------------------------


def dijkstra_from_goal(
    grid: PCBGrid,
    goal: tuple[int, int, int],
    *,
    via_cost: float = VIA_COST,
) -> np.ndarray:
    """Return a 3D float array `dist[layer, y, x]` of shortest-path cost
    from any reachable `(x, y, layer)` to `goal` (so `dist[goal] = 0`).
    Unreachable cells stay at `np.inf`.

    Uses the same edge model as `_astar_3d`: planar cost 1, via cost
    `via_cost`. No `prefer_directions` (we want a clean baseline).
    """
    L, H, W = grid.layers.shape
    dist = np.full((L, H, W), np.inf, dtype=np.float32)
    gx, gy, gz = goal
    if not grid.is_valid(gx, gy, gz):
        return dist
    dist[gz, gy, gx] = 0.0

    heap: list[tuple[float, int, int, int]] = [(0.0, gx, gy, gz)]
    while heap:
        d, x, y, z = heapq.heappop(heap)
        if d > dist[z, y, x]:
            continue

        # Planar neighbors.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not grid.is_valid(nx, ny, z):
                continue
            nd = d + 1.0
            if nd < dist[z, ny, nx]:
                dist[z, ny, nx] = nd
                heapq.heappush(heap, (nd, nx, ny, z))

        # Via neighbors.
        if L > 1:
            for dz in (-1, 1):
                nz = z + dz
                if not 0 <= nz < L:
                    continue
                if not grid.is_valid(x, y, nz):
                    continue
                nd = d + via_cost
                if nd < dist[nz, y, x]:
                    dist[nz, y, x] = nd
                    heapq.heappush(heap, (nd, x, y, nz))

    return dist


# ---------------------------------------------------------------------------
# Build supervised triples
# ---------------------------------------------------------------------------


def build_examples_for_board(
    grid: PCBGrid, rng: random.Random,
    *, max_examples_per_board: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three arrays for one board:
        crops:    (N, MAX_LAYERS, CROP_SIZE, CROP_SIZE) float32
        scalars:  (N, 3) float32
        targets:  (N,) float32 — true cost-to-go in cell units

    Picks a random goal on layer 0, runs Dijkstra, then samples up to
    `max_examples_per_board` reachable cells uniformly.
    """
    L, H, W = grid.layers.shape

    # Goal lives on layer 0 (matches the SMD-pad convention).
    while True:
        gx, gy = rng.randint(0, W - 1), rng.randint(0, H - 1)
        if grid.is_valid(gx, gy, 0):
            break
    goal = (gx, gy, 0)

    dist = dijkstra_from_goal(grid, goal)

    reachable_idxs = np.argwhere(np.isfinite(dist))  # rows: (z, y, x)
    if len(reachable_idxs) == 0:
        return (
            np.zeros((0, MAX_LAYERS, CROP_SIZE, CROP_SIZE), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    if len(reachable_idxs) > max_examples_per_board:
        sel = rng.sample(range(len(reachable_idxs)), max_examples_per_board)
        reachable_idxs = reachable_idxs[sel]

    layers_np = grid.layers.astype(np.float32)
    crops = np.zeros(
        (len(reachable_idxs), MAX_LAYERS, CROP_SIZE, CROP_SIZE),
        dtype=np.float32,
    )
    scalars = np.zeros((len(reachable_idxs), 3), dtype=np.float32)
    targets = np.zeros((len(reachable_idxs),), dtype=np.float32)

    for i, (z, y, x) in enumerate(reachable_idxs):
        crop = extract_crop(layers_np, int(x), int(y))
        crops[i] = crop
        density = float(crop.mean())
        scalars[i] = scalar_features(
            int(x), int(y), (gx, gy), W, H, density,
        )
        targets[i] = float(dist[z, y, x])

    return crops, scalars, targets


def build_dataset(
    n_boards: int = 5000,
    seed: int = 0,
    *,
    max_examples_per_board: int = 200,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate `build_examples_for_board` across `n_boards`."""
    rng = random.Random(seed)
    all_crops: list[np.ndarray] = []
    all_scalars: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for b in range(n_boards):
        grid = random_board(rng)
        crops, scalars, targets = build_examples_for_board(
            grid, rng, max_examples_per_board=max_examples_per_board,
        )
        all_crops.append(crops)
        all_scalars.append(scalars)
        all_targets.append(targets)

        if verbose and (b + 1) % 500 == 0:
            n_far = sum(len(c) for c in all_crops)
            print(f"  board {b + 1}/{n_boards}  examples={n_far}")

    crops_arr = np.concatenate(all_crops, axis=0)
    scalars_arr = np.concatenate(all_scalars, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    return crops_arr, scalars_arr, targets_arr


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-board", type=int, default=200)
    parser.add_argument(
        "--out", default=os.path.join(_HERE, "heuristic_data.npz"),
        help="Output .npz path",
    )
    args = parser.parse_args(argv)

    print(f"Generating dataset: {args.boards} boards, seed={args.seed}")
    crops, scalars, targets = build_dataset(
        n_boards=args.boards, seed=args.seed,
        max_examples_per_board=args.max_per_board,
    )
    print(f"\nDataset: crops={crops.shape}, scalars={scalars.shape}, "
          f"targets={targets.shape}")
    print(f"  target mean={targets.mean():.2f}  max={targets.max():.2f}")

    np.savez_compressed(args.out, crops=crops, scalars=scalars, targets=targets)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
