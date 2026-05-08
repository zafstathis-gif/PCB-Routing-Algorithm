"""Matplotlib visualization for PCB routing results."""

from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from pcb_grid import PCBGrid
from router import BestOfSummary, RouteSummary, route_board_best_of


def visualize_board(
    grid: PCBGrid,
    successful_paths: Union[RouteSummary, BestOfSummary],
    title: str = "PCB Routing Result",
) -> None:
    """Render the grid, obstacles, and routed traces with matplotlib.

    The grid passed in has typically already been mutated by the router so
    that routed traces are also marked as obstacles. We tease the two apart
    by subtracting the cells contained in `successful_paths["routed"]`, so
    only the *static* obstacles are drawn in black; traces are drawn as
    colored polylines on top. Any unrouted pin pairs are drawn as faded
    crosses connected by a dashed red line.

    Args:
        grid:             PCBGrid after routing.
        successful_paths: Summary dict from `route_board` or
                          `route_board_best_of`.
        title:            Plot title.
    """
    routed = successful_paths["routed"]
    unrouted = successful_paths.get("unrouted", [])

    # Rebuild the static-obstacle map by removing trace cells from grid.grid.
    static_obs = grid.grid.copy()
    for net in routed:
        for x, y in net["path"]:
            static_obs[y, x] = 0

    fig, ax = plt.subplots(figsize=(9, 9))

    # Greyscale heatmap: 0 = empty (light), 1 = obstacle (black).
    ax.imshow(static_obs, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")

    # Cell-boundary minor gridlines for a checkerboard look.
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    step = max(1, grid.width // 10)
    ax.set_xticks(np.arange(0, grid.width, step))
    ax.set_yticks(np.arange(0, grid.height, step))

    # Distinct bright colors: tab10 (<=10 nets) or tab20 (more).
    cmap = plt.get_cmap("tab10" if len(routed) <= 10 else "tab20")

    for i, net in enumerate(routed):
        color = cmap(i % cmap.N)
        xs = [c[0] for c in net["path"]]
        ys = [c[1] for c in net["path"]]

        ax.plot(
            xs, ys,
            color=color,
            linewidth=2.8,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=f"Net {i}: {net['pair'][0]} -> {net['pair'][1]}",
        )

        # Start = circle, end = star.
        (sx, sy), (ex, ey) = net["pair"]
        ax.plot(sx, sy, marker="o", color=color,
                markersize=12, markeredgecolor="black", markeredgewidth=1.2)
        ax.plot(ex, ey, marker="*", color=color,
                markersize=18, markeredgecolor="black", markeredgewidth=1.0)

    # Faded crosses + dashed line for nets the router could not connect.
    for j, pair in enumerate(unrouted):
        (sx, sy), (ex, ey) = pair
        label = "Unrouted" if j == 0 else None
        ax.plot([sx, ex], [sy, ey],
                color="red", linestyle=":", linewidth=1.2, alpha=0.55,
                label=label)
        ax.plot(sx, sy, marker="x", color="red", markersize=11,
                markeredgewidth=2.0, alpha=0.7)
        ax.plot(ex, ey, marker="x", color="red", markersize=11,
                markeredgewidth=2.0, alpha=0.7)

    # Top-left origin matches the PCBGrid convention.
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(grid.height - 0.5, -0.5)
    ax.set_aspect("equal")

    if "strategy" in successful_paths:
        title = f"{title}  [strategy: {successful_paths['strategy']}]"
    ax.set_title(title)

    if routed or unrouted:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=True,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    grid = PCBGrid(20, 20)

    # Static obstacles forming small barriers around the board.
    static_obstacles = [
        (5, 5), (5, 6), (5, 7), (6, 5), (7, 5),
        (10, 8), (10, 9), (10, 10), (10, 11),
        (14, 14), (15, 14), (16, 14),
        (3, 13), (3, 14), (3, 15),
        (12, 3), (13, 3), (8, 16),
    ]
    for x, y in static_obstacles:
        grid.add_obstacle(x, y)

    netlist = [
        ((1, 1), (18, 18)),
        ((1, 18), (18, 1)),
        ((6, 6), (16, 6)),
        ((11, 1), (18, 12)),
        ((1, 10), (9, 16)),
    ]

    summary = route_board_best_of(grid, netlist)

    print(f"Strategy chosen: {summary['strategy']}")
    print(f"Routed:   {len(summary['routed'])} / {len(netlist)}")
    for net in summary["routed"]:
        print(f"  {net['pair']}  ->  {len(net['path'])} cells")
    if summary["unrouted"]:
        print(f"Unrouted: {summary['unrouted']}")

    visualize_board(grid, summary)
