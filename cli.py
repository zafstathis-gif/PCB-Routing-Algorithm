"""Command-line entry point for the PCB autorouter.

Usage (after `pip install -e .`, the command is registered as `pcb-route`):

    pcb-route input.kicad_pcb -o output.kicad_pcb
    pcb-route input.kicad_pcb -o output.kicad_pcb --clearance 0.2 --grid 0.25
    pcb-route input.kicad_pcb -o output.kicad_pcb --strategy bbox_area_asc --rrr

Reads pads and any existing traces from `input.kicad_pcb`, routes every net,
and writes the result to `output.kicad_pcb`. Unrouted nets are reported on
stderr and left as un-traced pin connections in the output, so you can open
the file in KiCad and see exactly which connections still need attention.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence, Union

from kicad_io import load_board, save_routed_board
from router import (
    ALL_STRATEGIES,
    BestOfSummary,
    RouteSummary,
    route_board,
    route_board_best_of,
    route_board_rrr,
)


def _parse_strategy(s: str) -> object:
    if s == "user":
        return None
    valid = [v for v in ALL_STRATEGIES if v is not None] + ["user", "best_of"]
    if s not in valid:
        raise argparse.ArgumentTypeError(
            f"Unknown strategy {s!r}; choose one of {sorted(valid)}"
        )
    return s


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Input .kicad_pcb file")
    p.add_argument(
        "-o", "--output", required=True,
        help="Output .kicad_pcb file (input is not modified)",
    )
    p.add_argument(
        "--grid", type=float, default=0.5,
        help="Routing grid resolution in mm (default 0.5)",
    )
    p.add_argument(
        "--clearance", type=float, default=0.2,
        help="Trace-to-trace clearance in mm (default 0.2)",
    )
    p.add_argument(
        "--trace-width", type=float, default=0.25,
        help="Routed trace width in mm (default 0.25)",
    )
    p.add_argument(
        "--via-size", type=float, default=0.6,
        help="Via outer diameter in mm (default 0.6)",
    )
    p.add_argument(
        "--via-drill", type=float, default=0.3,
        help="Via drill diameter in mm (default 0.3)",
    )
    p.add_argument(
        "--strategy", type=_parse_strategy, default="best_of",
        help="Net-ordering strategy: user, manhattan_asc, manhattan_desc, "
             "bbox_area_asc, bbox_area_desc, or best_of (default).",
    )
    p.add_argument(
        "--rrr", action="store_true",
        help="Enable rip-up-and-reroute after the initial pass.",
    )
    p.add_argument(
        "--prefer-directions", action="store_true",
        help="Bias planar moves toward horizontal on even layers / vertical "
             "on odd layers (classical EDA heuristic).",
    )

    args = p.parse_args(argv)

    print(f"Reading {args.input} (grid={args.grid}mm, clearance={args.clearance}mm)")
    grid, netlist, ctx = load_board(
        args.input,
        grid_mm=args.grid,
        clearance_mm=args.clearance,
        trace_width_mm=args.trace_width,
        via_size_mm=args.via_size,
        via_drill_mm=args.via_drill,
    )
    print(
        f"  board: {grid.num_layers} copper layers, "
        f"{grid.width}x{grid.height} cells, clearance={grid.clearance} cells"
    )
    print(f"  netlist: {len(netlist)} pairs from "
          f"{len(set(ctx.pair_to_net_name.values()))} nets")

    summary: Union[RouteSummary, BestOfSummary]
    if args.strategy == "best_of":
        summary = route_board_best_of(
            grid, netlist, use_rrr=args.rrr,
            prefer_directions=args.prefer_directions,
        )
        print(f"Strategy: best_of (winner={summary['strategy']!r})")
    else:
        runner = route_board_rrr if args.rrr else route_board
        summary = runner(
            grid, netlist, sort_strategy=args.strategy,
            prefer_directions=args.prefer_directions,
        )
        print(f"Strategy: {args.strategy} {'+ R&R' if args.rrr else ''}")

    routed = len(summary["routed"])
    total = len(netlist)
    print(f"Routed {routed}/{total} ({100*routed/total:.1f}%)")

    if summary["unrouted"]:
        print(
            f"\n  {len(summary['unrouted'])} unrouted nets — left un-traced:",
            file=sys.stderr,
        )
        for pair in summary["unrouted"]:
            net_name = ctx.pair_to_net_name.get(pair, "?")
            print(f"    {net_name}: {pair}", file=sys.stderr)

    save_routed_board(ctx, summary, args.output)
    print(f"Wrote {args.output}")
    return 0 if not summary["unrouted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
