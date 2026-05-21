"""Evaluate the learned A* heuristic vs vanilla Manhattan A* on held-out boards.

Metrics reported:
  * Mean A* nodes expanded per route (the headline number — the learned
    heuristic exists to shrink this).
  * Mean path-length ratio (learned_path_len / manhattan_path_len). With the
    `min(learned, manhattan)` clamp this must be 1.000 by construction:
    sanity check that A* optimality wasn't broken.
  * Mean wall-clock per route. The honesty check — model inference must not
    erase the savings from fewer nodes.
  * Breakdown by obstacle density.

Run from the project root:

    python -m rl.evaluate_heuristic                  # uses rl/heuristic_net.pt
    python -m rl.evaluate_heuristic --boards 100     # fewer boards for a quick check
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pcb_grid import PCBGrid  # noqa: E402
from rl.heuristic_data import random_board  # noqa: E402
from rl.heuristic_net import HeuristicNet, make_learned_heuristic  # noqa: E402
from router import route_single_net  # noqa: E402

N_BOARDS = 200
PAIRS_PER_BOARD = 5
EVAL_SEED = 9999


def random_valid_pair(grid: PCBGrid, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    W, H = grid.width, grid.height
    while True:
        sx, sy = rng.randint(0, W - 1), rng.randint(0, H - 1)
        ex, ey = rng.randint(0, W - 1), rng.randint(0, H - 1)
        if (sx, sy) == (ex, ey):
            continue
        if grid.is_valid(sx, sy, 0) and grid.is_valid(ex, ey, 0):
            return (sx, sy), (ex, ey)


def _run_one(grid, start, end, heuristic=None):
    """Route once and return (nodes_expanded, wall_clock_s, path_length)."""
    nodes = [0]
    t0 = time.perf_counter()
    if heuristic is not None:
        path = route_single_net(
            grid, start, end,
            heuristic=heuristic,
            nodes_expanded=nodes,
        )
    else:
        path = route_single_net(grid, start, end, nodes_expanded=nodes)
    dt = time.perf_counter() - t0
    if path is None:
        return None
    return nodes[0], dt, len(path)


def evaluate(
    weights_path: str,
    n_boards: int = N_BOARDS,
    num_obstacles: int = 18,
) -> None:
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        print("Run `python -m rl.train_heuristic` first.")
        return

    model = HeuristicNet()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    print(f"Loaded heuristic from {weights_path}")
    print(f"Evaluating on {n_boards} held-out boards "
          f"× {PAIRS_PER_BOARD} routes each "
          f"(obstacles/board = {num_obstacles})")
    print("Comparing all four heuristic options:")
    print("  Manhattan        — vanilla A*, always admissible, always optimal")
    print("  Learned (max)    — max(learned, manhattan), tighter where the")
    print("                      net predicts above manhattan (the win case)")
    print("  Learned (min)    — min(learned, manhattan), strictly admissible")
    print("                      but always weaker than manhattan (sanity)")
    print("  Learned (raw)    — net only; weighted-A* style, no optimality\n")

    rng = random.Random(EVAL_SEED)

    rows = []
    for b in range(n_boards):
        grid = random_board(rng, num_obstacles=num_obstacles)
        density = float(grid.layers.mean())

        for _ in range(PAIRS_PER_BOARD):
            start, end = random_valid_pair(grid, rng)

            r_m = _run_one(grid, start, end, None)
            r_max = _run_one(grid, start, end, make_learned_heuristic(model, grid, clamp="max"))
            r_min = _run_one(grid, start, end, make_learned_heuristic(model, grid, clamp="min"))
            r_raw = _run_one(grid, start, end, make_learned_heuristic(model, grid, clamp="raw"))

            if any(r is None for r in (r_m, r_max, r_min, r_raw)):
                continue

            rows.append((density, *r_m, *r_max, *r_min, *r_raw))

    if not rows:
        print("No successful routes; cannot summarize.")
        return

    arr = np.array(rows)
    cols = ["density",
            "m_nodes", "m_time", "m_len",
            "max_nodes", "max_time", "max_len",
            "min_nodes", "min_time", "min_len",
            "raw_nodes", "raw_time", "raw_len"]

    def stats(prefix):
        return (arr[:, cols.index(f"{prefix}_nodes")].mean(),
                arr[:, cols.index(f"{prefix}_time")].mean(),
                arr[:, cols.index(f"{prefix}_len")].mean())

    m_nodes, m_time, m_len = stats("m")
    max_nodes, max_time, max_len = stats("max")
    min_nodes, min_time, min_len = stats("min")
    raw_nodes, raw_time, raw_len = stats("raw")

    print(f"{'method':<22} {'nodes':>10} {'vs M':>8} "
          f"{'wall (s)':>10} {'path len':>10} {'len ratio':>10}")
    print("-" * 78)
    for name, n, t, ln in [
        ("Manhattan", m_nodes, m_time, m_len),
        ("Learned (max)", max_nodes, max_time, max_len),
        ("Learned (min)", min_nodes, min_time, min_len),
        ("Learned (raw)", raw_nodes, raw_time, raw_len),
    ]:
        ratio_nodes = n / m_nodes
        ratio_len = ln / m_len
        print(f"{name:<22} {n:>10.1f} {ratio_nodes:>8.3f} "
              f"{t:>10.5f} {ln:>10.2f} {ratio_len:>10.4f}")

    # The bottom-line claim for the CV: learned-max admissibility violation rate.
    # If learned-max produced longer paths than Manhattan, the network overshot.
    overshoots = (arr[:, cols.index("max_len")] > arr[:, cols.index("m_len")] + 1e-6).mean()
    print(f"\nLearned (max) admissibility violations "
          f"(path longer than Manhattan-A*): {100*overshoots:.2f}%")
    print("(0% = network is admissible everywhere it was queried)")

    # Density breakdown for the "max" clamp (the headline configuration).
    print("\nLearned (max) vs Manhattan by obstacle density:")
    bins = [(0.0, 0.030), (0.030, 0.050), (0.050, 1.0)]
    densities = arr[:, 0]
    print(f"  {'density bucket':<18} {'n':>4} "
          f"{'M nodes':>10} {'L nodes':>10} {'ratio':>8}")
    print("  " + "-" * 56)
    for lo, hi in bins:
        mask = (densities >= lo) & (densities < hi)
        if not mask.any():
            continue
        sub_m = arr[mask, cols.index("m_nodes")].mean()
        sub_l = arr[mask, cols.index("max_nodes")].mean()
        print(f"  [{lo:.3f}, {hi:.3f})    {int(mask.sum()):>4} "
              f"{sub_m:>10.1f} {sub_l:>10.1f} {sub_l / sub_m:>8.3f}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default=os.path.join(_HERE, "heuristic_net.pt"),
        help="Path to the trained HeuristicNet weights",
    )
    parser.add_argument(
        "--boards", type=int, default=N_BOARDS,
        help=f"Number of held-out boards to evaluate on (default {N_BOARDS})",
    )
    parser.add_argument(
        "--obstacles", type=int, default=18,
        help="Static obstacles per board (default 18 = training distribution; "
             "raise to 60-100 to stress the heuristic on denser boards).",
    )
    args = parser.parse_args(argv)

    evaluate(args.weights, n_boards=args.boards, num_obstacles=args.obstacles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
