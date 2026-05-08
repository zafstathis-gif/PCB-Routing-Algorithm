"""A* pathfinding and sequential netlist routing for PCBGrid."""

from __future__ import annotations

import heapq
from typing import Iterable, Optional, TypedDict

import numpy as np

from pcb_grid import PCBGrid


Coord = tuple[int, int]
NetPair = tuple[Coord, Coord]

SortStrategy = Optional[str]
ALL_STRATEGIES: tuple[SortStrategy, ...] = (
    None,
    "manhattan_asc",
    "manhattan_desc",
    "bbox_area_asc",
    "bbox_area_desc",
)


class RoutedNet(TypedDict):
    pair: NetPair
    path: list[Coord]


class RouteSummary(TypedDict):
    routed: list[RoutedNet]
    unrouted: list[NetPair]


class BestOfSummary(TypedDict):
    routed: list[RoutedNet]
    unrouted: list[NetPair]
    strategy: SortStrategy


# 4-connected neighborhood: PCB traces run horizontally/vertically only.
_NEIGHBOR_OFFSETS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _manhattan(a: Coord, b: Coord) -> int:
    """Manhattan distance — admissible heuristic for 4-connected unit-cost grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bbox_area(pair: NetPair) -> int:
    (x1, y1), (x2, y2) = pair
    return (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)


def _reconstruct_path(came_from: dict[Coord, Coord], end: Coord) -> list[Coord]:
    """Walk the predecessor map backwards from end to start, then reverse."""
    path: list[Coord] = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


def route_single_net(
    grid: PCBGrid, start: Coord, end: Coord
) -> Optional[list[Coord]]:
    """Find a shortest 4-connected path from `start` to `end` on `grid`.

    A* search:
      * g_score[n]  = known cheapest cost from start to n.
      * h(n)        = Manhattan distance from n to end (admissible & consistent
                      on a 4-connected unit-cost grid, so A* is optimal).
      * f_score[n]  = g_score[n] + h(n) — the open set is ordered by this.

    The open set is a binary heap (`heapq`) keyed by (f, counter, node). The
    counter breaks f-score ties deterministically and avoids comparing tuples
    that contain the node itself. Rather than decrease-key (not supported by
    `heapq`), we push duplicate entries when a better g-score is found; stale
    pops are filtered implicitly by the `tentative_g < g_score[...]` guard.

    Args:
        grid:  The PCBGrid to route on. Obstacles are respected via
               `grid.is_valid`.
        start: (x, y) start coordinate.
        end:   (x, y) end coordinate.

    Returns:
        A list of (x, y) tuples from start to end inclusive, or None if no
        path exists (or if either endpoint is invalid).
    """
    if not grid.is_valid(*start) or not grid.is_valid(*end):
        return None

    if start == end:
        return [start]

    # g_score[n]: cheapest known cost from start to n. Missing key => +inf.
    g_score: dict[Coord, int] = {start: 0}

    # came_from[n]: predecessor of n on the best path found so far.
    came_from: dict[Coord, Coord] = {}

    # Open set ordered by f = g + h. The counter is a tiebreaker so heapq
    # never has to compare the Coord tuples directly.
    counter = 0
    open_heap: list[tuple[int, int, Coord]] = [(_manhattan(start, end), counter, start)]

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current == end:
            return _reconstruct_path(came_from, end)

        current_g = g_score[current]

        for dx, dy in _NEIGHBOR_OFFSETS:
            neighbor = (current[0] + dx, current[1] + dy)

            if not grid.is_valid(*neighbor):
                continue

            tentative_g = current_g + 1  # Unit edge cost.

            if tentative_g < g_score.get(neighbor, 1 << 30):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _manhattan(neighbor, end)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))

    return None


def _try_route_with_pin_clear(
    grid: PCBGrid, pair: NetPair
) -> Optional[list[Coord]]:
    """Route `pair`, temporarily clearing endpoints so pins can be shared.

    On success: returns the path (caller is responsible for marking cells on
    the grid). On failure: returns None and restores the original endpoint
    values.
    """
    (sx, sy), (ex, ey) = pair
    saved_s = grid.grid[sy, sx]
    saved_e = grid.grid[ey, ex]
    grid.grid[sy, sx] = 0
    grid.grid[ey, ex] = 0
    path = route_single_net(grid, pair[0], pair[1])
    if path is None:
        grid.grid[sy, sx] = saved_s
        grid.grid[ey, ex] = saved_e
    return path


def _grid_from_routed(static_grid: np.ndarray,
                      routed_nets: Iterable[RoutedNet]) -> np.ndarray:
    """Rebuild a grid array from static obstacles plus a list of routed paths."""
    g = static_grid.copy()
    for net in routed_nets:
        for x, y in net["path"]:
            g[y, x] = 1
    return g


def _sorted_netlist(
    netlist: Iterable[NetPair], strategy: SortStrategy
) -> list[NetPair]:
    """Return the netlist reordered according to `strategy`."""
    pairs = list(netlist)
    if strategy is None:
        return pairs
    if strategy == "manhattan_asc":
        return sorted(pairs, key=lambda p: _manhattan(p[0], p[1]))
    if strategy == "manhattan_desc":
        return sorted(pairs, key=lambda p: -_manhattan(p[0], p[1]))
    if strategy == "bbox_area_asc":
        return sorted(pairs, key=_bbox_area)
    if strategy == "bbox_area_desc":
        return sorted(pairs, key=lambda p: -_bbox_area(p))
    raise ValueError(f"Unknown sort strategy: {strategy!r}")


def route_board(
    grid: PCBGrid,
    netlist: list[NetPair],
    sort_strategy: SortStrategy = None,
) -> RouteSummary:
    """Route a netlist sequentially, turning each new path into an obstacle.

    For each (start, end) pair in `netlist` we call `route_single_net`. On
    success, every cell of the returned path is marked as an obstacle on
    `grid` so later nets cannot cross or share it. This mutates `grid`.

    Pin sharing: a pin can be the endpoint of multiple nets (e.g. a power
    rail). To support this, each net's endpoints are temporarily un-marked
    before its A* call and re-locked afterward — so a previously-routed
    trace ending at a pin does not block a new net starting from that pin.

    Net ordering: a sequential router's outcome depends heavily on net order.
    `sort_strategy` reorders the netlist before routing:
      * None              — preserve user order.
      * "manhattan_asc"   — shortest pin-to-pin distance first.
      * "manhattan_desc"  — longest first.
      * "bbox_area_asc"   — smallest bounding box first.
      * "bbox_area_desc"  — largest first.

    Args:
        grid:          PCBGrid to route on. Mutated in place.
        netlist:       List of ((sx, sy), (ex, ey)) pin pairs.
        sort_strategy: Net-ordering heuristic; see above.

    Returns:
        ``{"routed": [{"pair", "path"}, ...], "unrouted": [pair, ...]}``
    """
    routed: list[RoutedNet] = []
    unrouted: list[NetPair] = []

    for pair in _sorted_netlist(netlist, sort_strategy):
        path = _try_route_with_pin_clear(grid, pair)
        if path is None:
            unrouted.append(pair)
            continue

        routed.append({"pair": pair, "path": path})
        for x, y in path:
            grid.grid[y, x] = 1

    return {"routed": routed, "unrouted": unrouted}


def route_board_rrr(
    grid: PCBGrid,
    netlist: list[NetPair],
    sort_strategy: SortStrategy = None,
    max_iterations: int = 10,
    max_ripups_per_net: int = 3,
) -> RouteSummary:
    """Sequential routing with rip-up-and-reroute (R&R).

    Algorithm:
      1. Run an initial sequential pass with the given `sort_strategy`.
      2. For each net that failed, compute its *ideal path* on a static-only
         grid (i.e. as if no traces existed). Any already-routed net whose
         cells intersect that ideal path is a "blocker" — a candidate for
         rip-up.
      3. Tentatively rip up all (rippable) blockers, route the failed net on
         the cleared grid, then attempt to re-route the ripped-up nets in
         shortest-Manhattan order. Accept the swap iff the total number of
         routed nets strictly increased; otherwise revert.
      4. Repeat until no progress is made or `max_iterations` is reached.

    Oscillation control:
      * Each net has a per-board rip-up count, capped at `max_ripups_per_net`.
        Once exhausted, that net is no longer a rip-up candidate.
      * The outer loop bails on iterations that produce no net gain.

    Args:
        grid:                PCBGrid to route on. Mutated in place.
        netlist:             List of pin-pair connections.
        sort_strategy:       Ordering for the initial pass; see `route_board`.
        max_iterations:      Cap on R&R iterations after the initial pass.
        max_ripups_per_net:  Cap on how many times any single net may be
                             ripped up before being treated as immovable.

    Returns:
        Same shape as `route_board`: ``{"routed": [...], "unrouted": [...]}``.
    """
    static = grid.clone()

    initial = route_board(grid, netlist, sort_strategy=sort_strategy)
    routed: list[RoutedNet] = list(initial["routed"])
    unrouted: list[NetPair] = list(initial["unrouted"])

    if not unrouted:
        return {"routed": routed, "unrouted": unrouted}

    ripup_counts: dict[NetPair, int] = {pair: 0 for pair in netlist}

    for _ in range(max_iterations):
        if not unrouted:
            break

        progressed = False
        next_unrouted: list[NetPair] = []

        for failed_pair in unrouted:
            # 1. Find the ideal path the failed net would take on a static-only
            #    grid. Any routed net whose cells intersect this path is a
            #    candidate blocker.
            ideal_grid = static.clone()
            ideal_path = _try_route_with_pin_clear(ideal_grid, failed_pair)
            if ideal_path is None:
                # Truly impossible (endpoint geometry blocks even with no
                # traces) — give up on this net.
                next_unrouted.append(failed_pair)
                continue

            ideal_set = set(ideal_path)
            blocker_idxs = [
                i for i, net in enumerate(routed)
                if ripup_counts[net["pair"]] < max_ripups_per_net
                and any(cell in ideal_set for cell in net["path"])
            ]

            if not blocker_idxs:
                next_unrouted.append(failed_pair)
                continue

            # 2. Speculatively rip up the blockers and attempt the swap.
            saved_routed = list(routed)
            kept = [net for i, net in enumerate(routed) if i not in blocker_idxs]
            ripped_pairs = [routed[i]["pair"] for i in blocker_idxs]

            grid.grid[:] = _grid_from_routed(static.grid, kept)

            new_path = _try_route_with_pin_clear(grid, failed_pair)
            if new_path is None:
                # Swap is dead on arrival — restore and skip.
                grid.grid[:] = _grid_from_routed(static.grid, saved_routed)
                next_unrouted.append(failed_pair)
                continue

            for x, y in new_path:
                grid.grid[y, x] = 1

            kept_with_new: list[RoutedNet] = list(kept)
            kept_with_new.append({"pair": failed_pair, "path": new_path})

            # 3. Re-route the ripped nets, shortest first.
            re_routed: list[RoutedNet] = []
            re_failed: list[NetPair] = []
            for rp in _sorted_netlist(ripped_pairs, "manhattan_asc"):
                rp_path = _try_route_with_pin_clear(grid, rp)
                if rp_path is None:
                    re_failed.append(rp)
                else:
                    re_routed.append({"pair": rp, "path": rp_path})
                    for x, y in rp_path:
                        grid.grid[y, x] = 1

            # 4. Accept iff total routed strictly increased.
            new_total = len(kept_with_new) + len(re_routed)
            old_total = len(saved_routed)

            if new_total > old_total:
                routed = kept_with_new + re_routed
                next_unrouted.extend(re_failed)
                progressed = True
            else:
                # Net wash or regression — revert.
                grid.grid[:] = _grid_from_routed(static.grid, saved_routed)
                next_unrouted.append(failed_pair)

            # Count rip-up attempts whether accepted or reverted: this prevents
            # the loop from re-attempting the same futile swap forever.
            for rp in ripped_pairs:
                ripup_counts[rp] += 1

        unrouted = next_unrouted
        if not progressed:
            break

    return {"routed": routed, "unrouted": unrouted}


def route_board_best_of(
    grid: PCBGrid,
    netlist: list[NetPair],
    strategies: Iterable[SortStrategy] = ALL_STRATEGIES,
    use_rrr: bool = False,
) -> BestOfSummary:
    """Run sequential routing under multiple ordering strategies, keep the best.

    "Best" = most nets routed; total trace length breaks ties (shorter wins).

    With ``use_rrr=True``, each strategy is followed by rip-up-and-reroute
    (`route_board_rrr`); otherwise the plain greedy `route_board` is used.

    The original `grid` is mutated to reflect the winning attempt's routed
    traces; intermediate attempts use independent clones and are discarded.

    Returns:
        A `BestOfSummary` with the usual `routed`/`unrouted` plus a
        `strategy` field naming the winning ordering.
    """
    static = grid.clone()
    runner = route_board_rrr if use_rrr else route_board

    best_summary: Optional[RouteSummary] = None
    best_strategy: SortStrategy = None
    best_grid: Optional[PCBGrid] = None
    best_score: tuple[int, int] = (-1, 0)

    for strategy in strategies:
        trial = static.clone()
        summary = runner(trial, netlist, sort_strategy=strategy)

        # Score: more routed nets wins; on ties, shorter total trace wins.
        total_cells = sum(len(net["path"]) for net in summary["routed"])
        score = (len(summary["routed"]), -total_cells)

        if score > best_score:
            best_score = score
            best_summary = summary
            best_strategy = strategy
            best_grid = trial

    assert best_summary is not None and best_grid is not None

    # Replay the winning grid state onto the user's grid.
    grid.grid[:] = best_grid.grid

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
