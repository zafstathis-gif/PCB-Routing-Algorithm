"""Benchmark net-ordering strategies on a fixed netlist, with and without R&R.

Run:  python bench.py                  # legacy single-layer, clearance=0
      python bench.py --clearance 1    # enforce 1-cell trace-to-trace clearance
      python bench.py --layers 2       # 2-layer routing with vias
      python bench.py --layers 2 --clearance 1

Reports, for each strategy, how many nets were successfully routed (and the
total trace length) under both plain greedy routing and rip-up-and-reroute.
Quantifies the effect of (a) net ordering, (b) R&R, and (c) design rules.
"""

from __future__ import annotations

import argparse

from pcb_grid import PCBGrid
from router import ALL_STRATEGIES, NetPair, route_board, route_board_rrr

STATIC_OBSTACLES: list[tuple[int, int]] = [
    (5, 5), (5, 6), (5, 7), (6, 5), (7, 5),
    (10, 8), (10, 9), (10, 10), (10, 11),
    (14, 14), (15, 14), (16, 14),
    (3, 13), (3, 14), (3, 15),
    (12, 3), (13, 3), (8, 16),
]

NETLIST: list[NetPair] = [
    ((1, 1), (18, 18)),
    ((1, 18), (18, 1)),
    ((1, 10), (18, 10)),
    ((10, 1), (10, 18)),
    ((6, 6), (16, 6)),
    ((11, 1), (18, 12)),
    ((1, 10), (9, 16)),
    ((4, 2), (12, 17)),
]


def make_grid(num_layers: int, clearance: int) -> PCBGrid:
    g = PCBGrid(20, 20, num_layers=num_layers, clearance=clearance)
    # Static obstacles live on layer 0 only — components & wall geometry.
    for x, y in STATIC_OBSTACLES:
        g.add_obstacle(x, y, layer=0)
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of routing layers (default 1).")
    parser.add_argument("--clearance", type=int, default=0,
                        help="Trace-to-trace clearance in cells (default 0).")
    args = parser.parse_args()

    print(f"Netlist size: {len(NETLIST)}")
    print(f"Board: 20x20  layers={args.layers}  clearance={args.clearance}")
    print(f"Static obstacles: {len(STATIC_OBSTACLES)}")
    print()
    print(f"{'strategy':<18} {'greedy':<14} {'+ R&R':<14}")
    print("-" * 50)
    for strategy in ALL_STRATEGIES:
        plain = route_board(
            make_grid(args.layers, args.clearance),
            NETLIST, sort_strategy=strategy,
        )
        rrr = route_board_rrr(
            make_grid(args.layers, args.clearance),
            NETLIST, sort_strategy=strategy,
        )

        plain_cells = sum(len(n["path"]) for n in plain["routed"])
        rrr_cells = sum(len(n["path"]) for n in rrr["routed"])

        label = "user-order" if strategy is None else strategy
        print(f"{label:<18}"
              f" {len(plain['routed'])}/{len(NETLIST)}  ({plain_cells:>3} cells) "
              f" {len(rrr['routed'])}/{len(NETLIST)}  ({rrr_cells:>3} cells)")


if __name__ == "__main__":
    main()
