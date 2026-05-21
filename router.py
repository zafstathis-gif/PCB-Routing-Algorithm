"""A* pathfinding and sequential netlist routing for PCBGrid.

Internally, every coordinate is the 3-tuple `(x, y, layer)`. Endpoints accept
2-tuples (single-layer back-compat), 3-tuples, or `Pad` instances (through-hole
or SMD pads); paths returned by `route_single_net` match the dimensionality of
the input — a 2-tuple in/out for the single-layer case keeps existing callers
working unchanged.

Multi-layer routing uses the same A* core. Planar moves cost 1.0; layer
switches (vias) cost `via_cost` (default `VIA_COST = 10.0`). The Manhattan
heuristic |dx|+|dy| remains admissible because it underestimates the via cost
as 0. When `prefer_directions=True`, even layers favor horizontal moves
(cost 1.0) over vertical (cost 1.2) and odd layers reverse — the heuristic
stays admissible since all edges cost ≥ 1.0.
"""

from __future__ import annotations

import heapq
from typing import Iterable, Optional, TypedDict, Union

import numpy as np

from pcb_grid import Pad, PCBGrid


Coord2D = tuple[int, int]
Coord3D = tuple[int, int, int]
Coord = Union[Coord2D, Coord3D]
PathCell = Union[Coord2D, Coord3D]
Endpoint = Union[Coord2D, Coord3D, Pad]
NetPair = tuple[Endpoint, Endpoint]

SortStrategy = Optional[str]
ALL_STRATEGIES: tuple[SortStrategy, ...] = (
    None,
    "manhattan_asc",
    "manhattan_desc",
    "bbox_area_asc",
    "bbox_area_desc",
)


# Default cost of a single via (layer-switch) edge, in planar-cell units.
VIA_COST: float = 10.0


class RoutedNet(TypedDict, total=False):
    pair: NetPair
    path: list[PathCell]
    vias: list[Coord2D]  # (x, y) of every via along the path; multi-layer only


class RouteSummary(TypedDict):
    routed: list[RoutedNet]
    unrouted: list[NetPair]


class BestOfSummary(TypedDict):
    routed: list[RoutedNet]
    unrouted: list[NetPair]
    strategy: SortStrategy


# Planar 4-connected neighborhood: PCB traces run horizontally/vertically only.
_PLANAR_OFFSETS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ---------------------------------------------------------------------------
# Endpoint / cell helpers
# ---------------------------------------------------------------------------


def _endpoint_layers(ep: Endpoint) -> tuple[int, ...]:
    """Return the layer indices an endpoint occupies.

    2-tuple `(x, y)`            -> (0,)        (back-compat single-layer)
    3-tuple `(x, y, z)`         -> (z,)
    `Pad(x, y, layers=(...))`   -> the pad's layers verbatim
    """
    if isinstance(ep, Pad):
        return ep.layers
    if len(ep) == 2:
        return (0,)
    return (ep[2],)


def _endpoint_xy(ep: Endpoint) -> Coord2D:
    if isinstance(ep, Pad):
        return (ep.x, ep.y)
    return (ep[0], ep[1])


def _endpoint_seed_cells(ep: Endpoint) -> tuple[Coord3D, ...]:
    """Every 3D cell a multi-source A* should seed (or terminate) on."""
    x, y = _endpoint_xy(ep)
    return tuple((x, y, z) for z in _endpoint_layers(ep))


def _is_2d_endpoint(ep: Endpoint) -> bool:
    """True if the endpoint omits explicit layer info (back-compat shape)."""
    return not isinstance(ep, Pad) and len(ep) == 2


def _unpack_cell(cell: PathCell, default_layer: int = 0) -> Coord3D:
    """Promote a 2- or 3-tuple path cell to canonical `(x, y, layer)`."""
    if len(cell) == 2:
        return (cell[0], cell[1], default_layer)
    return cell  # type: ignore[return-value]


def _manhattan_xy(a_xy: Coord2D, b_xy: Coord2D) -> int:
    """Planar Manhattan distance — used for net ordering and bbox stats."""
    return abs(a_xy[0] - b_xy[0]) + abs(a_xy[1] - b_xy[1])


def _bbox_area(pair: NetPair) -> int:
    (x1, y1) = _endpoint_xy(pair[0])
    (x2, y2) = _endpoint_xy(pair[1])
    return (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)


# ---------------------------------------------------------------------------
# A* core (multi-layer)
# ---------------------------------------------------------------------------


def _admissible_h(node: Coord3D, goal_xy: Coord2D) -> int:
    """Manhattan distance to the goal's (x, y) — admissible across all layers."""
    return abs(node[0] - goal_xy[0]) + abs(node[1] - goal_xy[1])


def _reconstruct_path(came_from: dict[Coord3D, Coord3D], end: Coord3D) -> list[Coord3D]:
    path: list[Coord3D] = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


def _planar_edge_cost(z: int, dx: int, dy: int, prefer_directions: bool) -> float:
    if not prefer_directions:
        return 1.0
    horizontal = dy == 0
    even_layer = z % 2 == 0
    preferred = (horizontal and even_layer) or (not horizontal and not even_layer)
    return 1.0 if preferred else 1.2


def _astar_3d(
    grid: PCBGrid,
    starts: Iterable[Coord3D],
    goals: set[Coord3D],
    goals_xy: Coord2D,
    *,
    via_cost: float = VIA_COST,
    prefer_directions: bool = False,
) -> Optional[list[Coord3D]]:
    """Multi-source / multi-goal A* on a 3D `PCBGrid`.

    Edges:
      * planar (4-connected within a layer) cost 1.0 (or 1.0/1.2 with
        `prefer_directions=True`).
      * vias  (between adjacent layers at the same (x, y)) cost `via_cost`.

    Returns the lowest-cost path as a list of `(x, y, layer)`, or `None`
    if no goal is reachable.
    """
    open_heap: list[tuple[float, int, Coord3D]] = []
    g_score: dict[Coord3D, float] = {}
    came_from: dict[Coord3D, Coord3D] = {}
    counter = 0

    for s in starts:
        if not grid.is_valid(s[0], s[1], s[2]):
            continue
        g_score[s] = 0.0
        heapq.heappush(open_heap, (_admissible_h(s, goals_xy), counter, s))
        counter += 1

    if not open_heap:
        return None

    num_layers = grid.num_layers

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in goals:
            return _reconstruct_path(came_from, current)

        current_g = g_score[current]
        cx, cy, cz = current

        # Planar neighbors.
        for dx, dy in _PLANAR_OFFSETS:
            nx, ny = cx + dx, cy + dy
            if not grid.is_valid(nx, ny, cz):
                continue
            step = _planar_edge_cost(cz, dx, dy, prefer_directions)
            tentative_g = current_g + step
            neighbor: Coord3D = (nx, ny, cz)
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _admissible_h(neighbor, goals_xy)
                heapq.heappush(open_heap, (f_score, counter, neighbor))
                counter += 1

        # Via neighbors (up / down).
        if num_layers > 1:
            for dz in (-1, 1):
                nz = cz + dz
                if not 0 <= nz < num_layers:
                    continue
                if not grid.is_valid(cx, cy, nz):
                    continue
                tentative_g = current_g + via_cost
                neighbor = (cx, cy, nz)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + _admissible_h(neighbor, goals_xy)
                    heapq.heappush(open_heap, (f_score, counter, neighbor))
                    counter += 1

    return None


def route_single_net(
    grid: PCBGrid,
    start: Endpoint,
    end: Endpoint,
    *,
    via_cost: float = VIA_COST,
    prefer_directions: bool = False,
) -> Optional[list[PathCell]]:
    """Find a lowest-cost 4-connected (+via) path from `start` to `end`.

    Endpoints may be 2-tuples (single-layer back-compat), 3-tuples
    `(x, y, layer)`, or `Pad(x, y, layers)`. The returned path mirrors the
    input dimensionality: with single-layer 2-tuple inputs the path is a
    list of 2-tuples (existing callers don't break); otherwise it's a list
    of `(x, y, layer)` 3-tuples.

    Returns `None` if either endpoint is invalid or no path exists.
    """
    if start == end:
        # Trivially a single-cell path.
        if _is_2d_endpoint(start) and _is_2d_endpoint(end) and grid.num_layers == 1:
            return [(_endpoint_xy(start))]
        x, y = _endpoint_xy(start)
        z = _endpoint_layers(start)[0]
        return [(x, y, z)]

    starts = _endpoint_seed_cells(start)
    goals = set(_endpoint_seed_cells(end))
    goal_xy = _endpoint_xy(end)

    # Bail early if every seed cell is out of bounds / blocked.
    if not any(grid.is_valid(s[0], s[1], s[2]) for s in starts):
        return None
    if not any(grid.is_valid(g[0], g[1], g[2]) for g in goals):
        return None

    path_3d = _astar_3d(
        grid, starts, goals, goal_xy,
        via_cost=via_cost, prefer_directions=prefer_directions,
    )
    if path_3d is None:
        return None

    # Back-compat: strip the layer if both endpoints were 2-tuples on a
    # single-layer board.
    if grid.num_layers == 1 and _is_2d_endpoint(start) and _is_2d_endpoint(end):
        return [(x, y) for x, y, _ in path_3d]
    return list(path_3d)


# ---------------------------------------------------------------------------
# Sequential routing
# ---------------------------------------------------------------------------


def _pin_clear_cells(grid: PCBGrid, endpoint: Endpoint) -> list[Coord3D]:
    """Cells that should be temporarily freed so A* can route through `endpoint`.

    Always includes the endpoint's own seed cells (one per pad layer). Also
    includes every cell within `grid.clearance` Chebyshev distance that is
    NOT a static obstacle — those represent halo cells stamped by previously
    routed traces and must be temporarily lifted so a new net sharing this
    pad can route out of it. Static obstacles (walls, other pads, the board
    outline) are *never* cleared.
    """
    x, y = _endpoint_xy(endpoint)
    ep_layers = _endpoint_layers(endpoint)
    halo = grid.clearance

    cells: list[Coord3D] = []
    seen: set[Coord3D] = set()
    for z in ep_layers:
        # Pin cell itself — always include (even if static, so A* can start here).
        seed: Coord3D = (x, y, z)
        if seed not in seen:
            cells.append(seed)
            seen.add(seed)

        if halo == 0:
            continue

        for dy in range(-halo, halo + 1):
            for dx in range(-halo, halo + 1):
                if dx == 0 and dy == 0:
                    continue
                cx, cy = x + dx, y + dy
                if not (0 <= cx < grid.width and 0 <= cy < grid.height):
                    continue
                if grid.static_mask[z, cy, cx]:
                    continue  # never lift static obstacles
                cell: Coord3D = (cx, cy, z)
                if cell in seen:
                    continue
                cells.append(cell)
                seen.add(cell)
    return cells


def _try_route_with_pin_clear(
    grid: PCBGrid,
    pair: NetPair,
    *,
    via_cost: float = VIA_COST,
    prefer_directions: bool = False,
) -> Optional[list[PathCell]]:
    """Route `pair`, temporarily clearing endpoint cells + their non-static halo.

    On success: returns the path. The halo region is restored before returning
    so the caller can re-stamp the new path's own halo via `_stamp_path_on_grid`
    without losing the original trace halos of unrelated nets.
    On failure: same restoration; returns None.
    """
    region = _pin_clear_cells(grid, pair[0]) + _pin_clear_cells(grid, pair[1])

    saved: list[tuple[int, int, int, int]] = []
    seen: set[Coord3D] = set()
    for cell in region:
        if cell in seen:
            continue
        seen.add(cell)
        x, y, z = cell
        saved.append((x, y, z, int(grid.layers[z, y, x])))
        grid.layers[z, y, x] = 0

    path = route_single_net(
        grid, pair[0], pair[1],
        via_cost=via_cost, prefer_directions=prefer_directions,
    )

    # Always restore the halo region. If `path` is not None the caller will
    # call `_stamp_path_on_grid(grid, path)` to mark the new trace + its halo,
    # which is independent of what the *prior* halo state was.
    for x, y, z, val in saved:
        grid.layers[z, y, x] = val

    return path


def _detect_vias(path: list[PathCell]) -> list[Coord2D]:
    """Return the `(x, y)` of each layer transition in a 3-tuple path."""
    vias: list[Coord2D] = []
    for a, b in zip(path, path[1:]):
        if len(a) == 3 and len(b) == 3 and a[2] != b[2]:
            vias.append((a[0], a[1]))
    return vias


def _stamp_path_on_grid(grid: PCBGrid, path: list[PathCell]) -> None:
    """Mark every cell of `path` (and its `grid.clearance` halo) as OBSTACLE.

    Trace stamping does NOT update `static_mask` — the static layer is
    reserved for pads / walls placed via `add_obstacle` / `add_via`.
    """
    grid.stamp_path(path)  # uses grid.clearance by default
    for via_xy in _detect_vias(path):
        grid.vias.add(via_xy)


def _rebuild_grid_from_routed(
    static_grid: PCBGrid,
    routed_nets: Iterable[RoutedNet],
) -> np.ndarray:
    """Build a 3D layer array from static obstacles + routed-path halos.

    Returns just the `.layers` array (the caller is expected to slice-assign
    it into a working grid). `static_grid.clearance` controls the halo width.
    """
    out = static_grid.layers.copy()
    halo = static_grid.clearance
    H, W = static_grid.height, static_grid.width
    for net in routed_nets:
        for cell in net["path"]:
            x, y, z = _unpack_cell(cell)
            x0 = max(0, x - halo)
            x1 = min(W, x + halo + 1)
            y0 = max(0, y - halo)
            y1 = min(H, y + halo + 1)
            out[z, y0:y1, x0:x1] = 1
    return out


def _sorted_netlist(
    netlist: Iterable[NetPair], strategy: SortStrategy
) -> list[NetPair]:
    """Return the netlist reordered according to `strategy`."""
    pairs = list(netlist)
    if strategy is None:
        return pairs
    if strategy == "manhattan_asc":
        return sorted(pairs, key=lambda p: _manhattan_xy(_endpoint_xy(p[0]), _endpoint_xy(p[1])))
    if strategy == "manhattan_desc":
        return sorted(pairs, key=lambda p: -_manhattan_xy(_endpoint_xy(p[0]), _endpoint_xy(p[1])))
    if strategy == "bbox_area_asc":
        return sorted(pairs, key=_bbox_area)
    if strategy == "bbox_area_desc":
        return sorted(pairs, key=lambda p: -_bbox_area(p))
    raise ValueError(f"Unknown sort strategy: {strategy!r}")


def route_board(
    grid: PCBGrid,
    netlist: list[NetPair],
    sort_strategy: SortStrategy = None,
    *,
    via_cost: float = VIA_COST,
    prefer_directions: bool = False,
) -> RouteSummary:
    """Route a netlist sequentially, turning each new path into an obstacle.

    Mutates `grid`. See `route_single_net` for endpoint forms — 2-tuples,
    3-tuples, and `Pad` instances are all accepted, and the resulting
    `RoutedNet["path"]` mirrors the input dimensionality.
    """
    routed: list[RoutedNet] = []
    unrouted: list[NetPair] = []

    for pair in _sorted_netlist(netlist, sort_strategy):
        path = _try_route_with_pin_clear(
            grid, pair, via_cost=via_cost, prefer_directions=prefer_directions,
        )
        if path is None:
            unrouted.append(pair)
            continue

        net: RoutedNet = {"pair": pair, "path": path, "vias": _detect_vias(path)}
        routed.append(net)
        _stamp_path_on_grid(grid, path)

    return {"routed": routed, "unrouted": unrouted}


# ---------------------------------------------------------------------------
# Rip-up and reroute
# ---------------------------------------------------------------------------


def route_board_rrr(
    grid: PCBGrid,
    netlist: list[NetPair],
    sort_strategy: SortStrategy = None,
    max_iterations: int = 10,
    max_ripups_per_net: int = 3,
    *,
    via_cost: float = VIA_COST,
    prefer_directions: bool = False,
) -> RouteSummary:
    """Sequential routing with rip-up-and-reroute (R&R).

    Same algorithm as before; generalized to multi-layer grids via the
    3D `_rebuild_grid_from_routed` helper. Halo stamping is automatic when
    `grid.clearance > 0`.
    """
    static = grid.clone()

    initial = route_board(
        grid, netlist, sort_strategy=sort_strategy,
        via_cost=via_cost, prefer_directions=prefer_directions,
    )
    routed: list[RoutedNet] = list(initial["routed"])
    unrouted: list[NetPair] = list(initial["unrouted"])

    if not unrouted:
        return {"routed": routed, "unrouted": unrouted}

    # NetPair keys may not be hashable when endpoints are `Pad` namedtuples
    # — but `Pad` *is* a NamedTuple so tuples-of-Pad are still hashable.
    ripup_counts: dict[NetPair, int] = {pair: 0 for pair in netlist}

    for _ in range(max_iterations):
        if not unrouted:
            break

        progressed = False
        next_unrouted: list[NetPair] = []

        for failed_pair in unrouted:
            # 1. Ideal path on a static-only grid identifies blockers.
            ideal_grid = static.clone()
            ideal_path = _try_route_with_pin_clear(
                ideal_grid, failed_pair,
                via_cost=via_cost, prefer_directions=prefer_directions,
            )
            if ideal_path is None:
                next_unrouted.append(failed_pair)
                continue

            ideal_set: set[Coord3D] = {_unpack_cell(c) for c in ideal_path}
            blocker_idxs = [
                i for i, net in enumerate(routed)
                if ripup_counts[net["pair"]] < max_ripups_per_net
                and any(_unpack_cell(cell) in ideal_set for cell in net["path"])
            ]

            if not blocker_idxs:
                next_unrouted.append(failed_pair)
                continue

            # 2. Speculatively rip up blockers and try the swap.
            saved_routed = list(routed)
            kept = [net for i, net in enumerate(routed) if i not in blocker_idxs]
            ripped_pairs = [routed[i]["pair"] for i in blocker_idxs]

            grid.layers[:] = _rebuild_grid_from_routed(static, kept)

            new_path = _try_route_with_pin_clear(
                grid, failed_pair,
                via_cost=via_cost, prefer_directions=prefer_directions,
            )
            if new_path is None:
                grid.layers[:] = _rebuild_grid_from_routed(static, saved_routed)
                next_unrouted.append(failed_pair)
                continue

            _stamp_path_on_grid(grid, new_path)
            kept_with_new: list[RoutedNet] = list(kept)
            kept_with_new.append({
                "pair": failed_pair,
                "path": new_path,
                "vias": _detect_vias(new_path),
            })

            # 3. Re-route the ripped nets, shortest first.
            re_routed: list[RoutedNet] = []
            re_failed: list[NetPair] = []
            for rp in _sorted_netlist(ripped_pairs, "manhattan_asc"):
                rp_path = _try_route_with_pin_clear(
                    grid, rp,
                    via_cost=via_cost, prefer_directions=prefer_directions,
                )
                if rp_path is None:
                    re_failed.append(rp)
                else:
                    re_routed.append({
                        "pair": rp, "path": rp_path, "vias": _detect_vias(rp_path),
                    })
                    _stamp_path_on_grid(grid, rp_path)

            # 4. Accept iff total routed strictly increased.
            new_total = len(kept_with_new) + len(re_routed)
            old_total = len(saved_routed)

            if new_total > old_total:
                routed = kept_with_new + re_routed
                next_unrouted.extend(re_failed)
                progressed = True
            else:
                grid.layers[:] = _rebuild_grid_from_routed(static, saved_routed)
                next_unrouted.append(failed_pair)

            for rp in ripped_pairs:
                ripup_counts[rp] += 1

        unrouted = next_unrouted
        if not progressed:
            break

    return {"routed": routed, "unrouted": unrouted}


# ---------------------------------------------------------------------------
# Best-of-strategies wrapper
# ---------------------------------------------------------------------------


def route_board_best_of(
    grid: PCBGrid,
    netlist: list[NetPair],
    strategies: Iterable[SortStrategy] = ALL_STRATEGIES,
    use_rrr: bool = False,
    *,
    via_cost: float = VIA_COST,
    prefer_directions: bool = False,
) -> BestOfSummary:
    """Run sequential routing under multiple ordering strategies, keep the best.

    "Best" = most nets routed; total trace length breaks ties (shorter wins).
    With ``use_rrr=True``, each strategy is followed by R&R.
    """
    static = grid.clone()
    runner = route_board_rrr if use_rrr else route_board

    best_summary: Optional[RouteSummary] = None
    best_strategy: SortStrategy = None
    best_grid: Optional[PCBGrid] = None
    best_score: tuple[int, int] = (-1, 0)

    for strategy in strategies:
        trial = static.clone()
        summary = runner(
            trial, netlist, sort_strategy=strategy,
            via_cost=via_cost, prefer_directions=prefer_directions,
        )

        total_cells = sum(len(net["path"]) for net in summary["routed"])
        score = (len(summary["routed"]), -total_cells)

        if score > best_score:
            best_score = score
            best_summary = summary
            best_strategy = strategy
            best_grid = trial

    assert best_summary is not None and best_grid is not None

    grid.layers[:] = best_grid.layers
    grid.vias = set(best_grid.vias)

    return {
        "routed": best_summary["routed"],
        "unrouted": best_summary["unrouted"],
        "strategy": best_strategy,
    }


if __name__ == "__main__":
    grid = PCBGrid(20, 20)

    # Build a vertical wall with a gap at y=10.
    for y in range(20):
        if y != 10:
            grid.add_obstacle(10, y)

    netlist: list[NetPair] = [
        ((2, 5), (17, 15)),
        ((2, 2), (18, 5)),
        ((1, 18), (18, 18)),
        ((5, 0), (5, 19)),
    ]

    summary = route_board_best_of(grid, netlist)
    print(f"Strategy chosen: {summary['strategy']}")
    print(f"Routed: {len(summary['routed'])} / {len(netlist)}")
    if summary["unrouted"]:
        print(f"Unrouted: {summary['unrouted']}")
