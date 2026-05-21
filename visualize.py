"""Matplotlib visualization for PCB routing results.

Single-layer boards render as a single subplot — identical to the original
rendering. Multi-layer boards render as a grid of subplots, one per layer,
with vias marked as black-outlined circles bridging stacked subplots.
"""

from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from pcb_grid import Pad, PCBGrid
from router import (
    BestOfSummary,
    NetPair,
    RouteSummary,
    _unpack_cell,
    route_board_best_of,
)


def _endpoint_xy_layer(ep) -> tuple[int, int, tuple[int, ...]]:
    """Pull out (x, y, layers_tuple) from any endpoint form."""
    if isinstance(ep, Pad):
        return ep.x, ep.y, ep.layers
    if len(ep) == 2:
        return ep[0], ep[1], (0,)
    return ep[0], ep[1], (ep[2],)


def _draw_layer(
    ax,
    static_obs_2d: np.ndarray,
    halo_2d: np.ndarray,
    routed_on_layer,
    vias,
    unrouted,
    grid: PCBGrid,
    layer: int,
    show_unrouted: bool,
) -> None:
    """Render one layer's subplot.

    Static obstacles (walls, pads, vias) are drawn dark; trace-clearance
    halos are drawn light gray. Traces themselves are drawn as colored
    polylines on top.
    """
    # Combined background: 0 = empty, 1 = halo (light), 2 = static (dark).
    background = np.where(static_obs_2d > 0, 2, np.where(halo_2d > 0, 1, 0))
    ax.imshow(
        background,
        cmap=plt.matplotlib.colors.ListedColormap(
            ["white", "#dcdcdc", "#222222"]
        ),
        vmin=0, vmax=2, interpolation="nearest",
    )

    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    step = max(1, grid.width // 10)
    ax.set_xticks(np.arange(0, grid.width, step))
    ax.set_yticks(np.arange(0, grid.height, step))

    cmap = plt.get_cmap("tab10" if len(routed_on_layer) <= 10 else "tab20")

    for net_idx, segments in routed_on_layer:
        color = cmap(net_idx % cmap.N)
        for seg in segments:
            xs = [c[0] for c in seg]
            ys = [c[1] for c in seg]
            ax.plot(xs, ys, color=color, linewidth=2.8,
                    solid_capstyle="round", solid_joinstyle="round")

    # Vias: small black-outlined circles, drawn on every layer they touch.
    for vx, vy in vias:
        ax.plot(vx, vy, marker="o", color="white",
                markersize=10, markeredgecolor="black", markeredgewidth=1.6,
                zorder=10)

    if show_unrouted:
        for j, pair in enumerate(unrouted):
            (sx, sy, _slayers) = _endpoint_xy_layer(pair[0])
            (ex, ey, _elayers) = _endpoint_xy_layer(pair[1])
            label = "Unrouted" if j == 0 else None
            ax.plot([sx, ex], [sy, ey],
                    color="red", linestyle=":", linewidth=1.2, alpha=0.55,
                    label=label)
            ax.plot(sx, sy, marker="x", color="red", markersize=11,
                    markeredgewidth=2.0, alpha=0.7)
            ax.plot(ex, ey, marker="x", color="red", markersize=11,
                    markeredgewidth=2.0, alpha=0.7)

    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(grid.height - 0.5, -0.5)
    ax.set_aspect("equal")
    if grid.num_layers > 1:
        ax.set_title(f"Layer {layer}")


def visualize_board(
    grid: PCBGrid,
    successful_paths: Union[RouteSummary, BestOfSummary],
    title: str = "PCB Routing Result",
) -> None:
    """Render the grid, obstacles, and routed traces with matplotlib.

    For multi-layer boards, draws one subplot per layer. Vias appear on every
    layer they bridge.
    """
    routed = successful_paths["routed"]
    unrouted = successful_paths.get("unrouted", [])

    # Separate static obstacles (walls/pads) from trace halos. Static cells
    # render black; halo cells render light gray so the viewer can see how
    # clearance shapes the routable region.
    static_mask = grid.static_mask.astype(np.int8)
    halo_mask = grid.layers.astype(np.int8) - static_mask
    # Trace cells themselves are also in halo_mask — peel them out so the
    # colored polyline is drawn on top of a clean background.
    for net in routed:
        for cell in net["path"]:
            x, y, z = _unpack_cell(cell)
            halo_mask[z, y, x] = 0
    # Clamp to {0, 1} (halo_mask could be negative if the user mutated layers
    # directly; we don't worry about that case).
    halo_mask = np.clip(halo_mask, 0, 1)

    # Group each net's path cells by layer so we can draw each layer's polyline.
    # A net may visit multiple layers via vias; we draw each contiguous run
    # of same-layer cells as its own polyline segment.
    per_layer_routed: dict[int, list] = {z: [] for z in range(grid.num_layers)}
    for net_idx, net in enumerate(routed):
        segments_by_layer: dict[int, list[list[tuple[int, int]]]] = {
            z: [] for z in range(grid.num_layers)
        }
        current_layer = None
        current_seg: list[tuple[int, int]] = []
        for cell in net["path"]:
            x, y, z = _unpack_cell(cell)
            if z != current_layer and current_seg:
                segments_by_layer[current_layer].append(current_seg)  # type: ignore[index]
                current_seg = []
            current_layer = z
            current_seg.append((x, y))
        if current_seg and current_layer is not None:
            segments_by_layer[current_layer].append(current_seg)

        for z, segments in segments_by_layer.items():
            if segments:
                per_layer_routed[z].append((net_idx, segments))

    via_set: set[tuple[int, int]] = set()
    for net in routed:
        for via_xy in net.get("vias", []):
            via_set.add(via_xy)
    via_set |= grid.vias

    n = grid.num_layers
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(min(9, 5 * cols), min(9, 5 * rows)) if n > 1 else (9, 9),
        squeeze=False,
    )

    cmap = plt.get_cmap("tab10" if len(routed) <= 10 else "tab20")

    for z in range(n):
        ax = axes[z // cols][z % cols]
        _draw_layer(
            ax,
            static_obs_2d=static_mask[z],
            halo_2d=halo_mask[z],
            routed_on_layer=per_layer_routed[z],
            vias=via_set,
            unrouted=unrouted,
            grid=grid,
            layer=z,
            show_unrouted=(z == 0),
        )

        # Pin markers (circles for start, stars for end) on the layers where
        # the pin actually lives.
        for net_idx, net in enumerate(routed):
            color = cmap(net_idx % cmap.N)
            (sx, sy, slayers) = _endpoint_xy_layer(net["pair"][0])
            (ex, ey, elayers) = _endpoint_xy_layer(net["pair"][1])
            if z in slayers:
                ax.plot(sx, sy, marker="o", color=color,
                        markersize=12, markeredgecolor="black", markeredgewidth=1.2)
            if z in elayers:
                ax.plot(ex, ey, marker="*", color=color,
                        markersize=18, markeredgecolor="black", markeredgewidth=1.0)

    # Hide any unused subplots in the grid (e.g. 3 layers in a 2x2 figure).
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    # `successful_paths` may be a RouteSummary (no strategy key) or a
    # BestOfSummary (has it). cast to dict so .get() type-checks cleanly.
    summary_any = dict(successful_paths)
    strategy = summary_any.get("strategy")
    if strategy is not None:
        title = f"{title}  [strategy: {strategy}]"
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def animate_board(
    grid: PCBGrid,
    summary: Union[RouteSummary, BestOfSummary],
    *,
    output_path: str,
    fps: int = 4,
    cells_per_frame: int = 4,
) -> None:
    """Render the routing of `summary` as an animated GIF.

    Each frame extends the currently-drawn paths by `cells_per_frame` cells
    until every net is fully drawn, then holds for a few frames so the
    final state is readable when the GIF loops.

    Single-layer boards: one subplot. Multi-layer: per-layer subplots, with
    vias appearing as the path crosses them.

    Saving a GIF requires Pillow (already a transitive dep of matplotlib).
    """
    import matplotlib.animation as anim  # local import keeps top-level fast

    routed = summary["routed"]
    # animate_board only draws successful nets — unrouted pairs would just
    # flicker in place and add no information to the animation.

    # Static obstacle / halo masks (same as the static visualizer).
    static_mask = grid.static_mask.astype(np.int8)
    halo_mask = grid.layers.astype(np.int8) - static_mask
    for net in routed:
        for cell in net["path"]:
            x, y, z = _unpack_cell(cell)
            halo_mask[z, y, x] = 0
    halo_mask = np.clip(halo_mask, 0, 1)

    # Pre-compute the full ordered cell list per (net, layer) segment.
    # Drawing order: net by net, cell by cell within each net.
    per_layer_segments: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {
        z: [] for z in range(grid.num_layers)
    }
    via_appearances: list[tuple[int, tuple[int, int]]] = []  # (cell-index, (x,y))
    cumulative_cells = 0
    for net_idx, net in enumerate(routed):
        current_layer = None
        current_seg: list[tuple[int, int]] = []
        for cell in net["path"]:
            x, y, z = _unpack_cell(cell)
            if z != current_layer:
                if current_seg and current_layer is not None:
                    per_layer_segments[current_layer].append((net_idx, current_seg))
                    current_seg = []
                if current_layer is not None:
                    via_appearances.append((cumulative_cells, (x, y)))
                current_layer = z
            current_seg.append((x, y))
            cumulative_cells += 1
        if current_seg and current_layer is not None:
            per_layer_segments[current_layer].append((net_idx, current_seg))

    total_cells = sum(
        len(seg) for layer in per_layer_segments.values() for _, seg in layer
    )
    cells_per_frame = max(1, cells_per_frame)
    n_anim_frames = (total_cells + cells_per_frame - 1) // cells_per_frame
    hold_frames = max(1, fps)  # hold final frame for ~1 second
    total_frames = n_anim_frames + hold_frames

    n = grid.num_layers
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(min(9, 5 * cols), min(9, 5 * rows)) if n > 1 else (8, 8),
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10" if len(routed) <= 10 else "tab20")

    # Per-layer state: list of Line2D handles, one per net's segment on this layer.
    per_layer_lines: dict[int, list] = {z: [] for z in range(n)}
    via_artists: list = []

    def init_axes():
        for z in range(n):
            ax = axes[z // cols][z % cols]
            ax.clear()
            background = np.where(
                static_mask[z] > 0, 2,
                np.where(halo_mask[z] > 0, 1, 0),
            )
            ax.imshow(
                background,
                cmap=plt.matplotlib.colors.ListedColormap(
                    ["white", "#dcdcdc", "#222222"]),
                vmin=0, vmax=2, interpolation="nearest",
            )
            ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
            ax.grid(which="minor", color="lightgray", linewidth=0.4)
            ax.tick_params(which="minor", length=0)
            ax.set_xlim(-0.5, grid.width - 0.5)
            ax.set_ylim(grid.height - 0.5, -0.5)
            ax.set_aspect("equal")
            if n > 1:
                ax.set_title(f"Layer {z}")
            # Pre-create one Line2D per net per layer (initially empty).
            per_layer_lines[z] = []
            for net_idx, _ in per_layer_segments[z]:
                color = cmap(net_idx % cmap.N)
                line, = ax.plot(
                    [], [], color=color, linewidth=2.8,
                    solid_capstyle="round", solid_joinstyle="round",
                )
                per_layer_lines[z].append(line)

        # Pin markers (always visible from frame 0).
        for net_idx, net in enumerate(routed):
            color = cmap(net_idx % cmap.N)
            (sx, sy, slayers) = _endpoint_xy_layer(net["pair"][0])
            (ex, ey, elayers) = _endpoint_xy_layer(net["pair"][1])
            for z in slayers:
                if 0 <= z < n:
                    axes[z // cols][z % cols].plot(
                        sx, sy, marker="o", color=color,
                        markersize=10, markeredgecolor="black",
                        markeredgewidth=1.2,
                    )
            for z in elayers:
                if 0 <= z < n:
                    axes[z // cols][z % cols].plot(
                        ex, ey, marker="*", color=color,
                        markersize=14, markeredgecolor="black",
                        markeredgewidth=1.0,
                    )

        # Hide unused subplots.
        for k in range(n, rows * cols):
            axes[k // cols][k % cols].axis("off")

    def update(frame_idx: int):
        cells_drawn = min(total_cells, (frame_idx + 1) * cells_per_frame)

        # For each layer, extend each segment up to its share of cells_drawn.
        consumed = 0
        for z in range(n):
            for line_idx, (net_idx, seg) in enumerate(per_layer_segments[z]):
                seg_len = len(seg)
                if consumed >= cells_drawn:
                    per_layer_lines[z][line_idx].set_data([], [])
                else:
                    take = min(seg_len, cells_drawn - consumed)
                    xs = [c[0] for c in seg[:take]]
                    ys = [c[1] for c in seg[:take]]
                    per_layer_lines[z][line_idx].set_data(xs, ys)
                consumed += seg_len

        # Vias: drawn once the underlying cell is part of `cells_drawn`.
        for va in via_artists:
            va.remove()
        via_artists.clear()
        for via_at_cell, (vx, vy) in via_appearances:
            if cells_drawn > via_at_cell:
                for z in range(n):
                    ax = axes[z // cols][z % cols]
                    art, = ax.plot(
                        vx, vy, marker="o", color="white",
                        markersize=8, markeredgecolor="black",
                        markeredgewidth=1.4, zorder=10,
                    )
                    via_artists.append(art)
        return []

    init_axes()
    animation = anim.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // fps, blit=False,
    )
    animation.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)


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

    netlist: list[NetPair] = [
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
