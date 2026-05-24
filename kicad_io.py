"""Read and write `.kicad_pcb` files so the router can run on real boards.

`load_board(path)` parses a `.kicad_pcb` via `kiutils`, builds a `PCBGrid`,
collects pads grouped by net, and returns a netlist plus a `KiCadContext`
that remembers how to write the routed result back. `save_routed_board(ctx,
summary, path)` takes a `RouteSummary` and appends `(segment ...)` and
`(via ...)` items to the kiutils Board, then writes it out.

A few conversions to keep in mind:

- **Coordinates.** KiCad uses millimetres with origin at the top-left of
  the page. The router uses integer cells with the same orientation. The
  conversion is a uniform mm-per-cell scale (default 0.5 mm/cell), so a
  50 x 30 mm board becomes a 100 x 60 grid.
- **Layers.** KiCad copper layers are mapped to router layer indices in
  stack-up order: ``F.Cu`` -> 0, ``In1.Cu`` -> 1, ..., ``B.Cu`` -> N-1.
- **Pads.** SMD pads sit on a single copper layer. Through-hole pads
  (``thru_hole``, layers include ``*.Cu``) span every copper layer and
  are stamped as vias.
- **Multi-pad nets.** Nets with more than two pads are connected via a
  Kruskal MST over pad positions: N pads -> N-1 routed pairs.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pcb_grid import Pad as RPad
from pcb_grid import PCBGrid

# kiutils is imported lazily so the module loads cleanly without the dependency.
try:
    import kiutils.board
    from kiutils.items.brditems import Segment, Via
    from kiutils.items.common import Position
    _HAVE_KIUTILS = True
except ImportError:  # pragma: no cover
    _HAVE_KIUTILS = False


DEFAULT_GRID_MM: float = 0.5      # one router cell = 0.5 mm by default
DEFAULT_TRACE_WIDTH_MM: float = 0.25
DEFAULT_VIA_SIZE_MM: float = 0.6
DEFAULT_VIA_DRILL_MM: float = 0.3
DEFAULT_BOARD_MARGIN_MM: float = 2.0


@dataclass
class KiCadContext:
    """All the state needed to translate a routed result back into a KiCad board.

    Held opaque by the caller — pass it through unchanged from `load_board`
    into `save_routed_board`.
    """
    board: Any  # kiutils.board.Board (typed as Any so kiutils stays lazy)
    grid_mm: float
    origin_x_mm: float
    origin_y_mm: float
    copper_layers: list[str]      # ordered: ["F.Cu", "In1.Cu", ..., "B.Cu"]
    trace_width_mm: float
    via_size_mm: float
    via_drill_mm: float
    net_name_to_number: dict[str, int] = field(default_factory=dict)
    pair_to_net_name: dict[tuple, str] = field(default_factory=dict)  # NetPair -> name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_board(
    path: str,
    *,
    grid_mm: float = DEFAULT_GRID_MM,
    clearance_mm: float = 0.0,
    margin_mm: float = DEFAULT_BOARD_MARGIN_MM,
    trace_width_mm: float = DEFAULT_TRACE_WIDTH_MM,
    via_size_mm: float = DEFAULT_VIA_SIZE_MM,
    via_drill_mm: float = DEFAULT_VIA_DRILL_MM,
) -> tuple[PCBGrid, list[tuple], KiCadContext]:
    """Parse `.kicad_pcb` and build a router-ready `(PCBGrid, netlist, context)`.

    Args:
        path: Path to the `.kicad_pcb` file.
        grid_mm: Routing grid resolution in mm. Smaller = finer routing but
                 quadratically more cells.
        clearance_mm: Trace-to-trace clearance in mm. Converted to cells via
                      ``ceil(clearance_mm / grid_mm)``.
        margin_mm: Padding to add around the pad bounding box when no
                   Edge.Cuts outline is present.
        trace_width_mm: Width of routed traces in the output file.
        via_size_mm, via_drill_mm: Via geometry for the output file.

    Returns:
        A 3-tuple ``(grid, netlist, ctx)``.
        * ``grid`` is the populated `PCBGrid` (pads/existing traces marked
          as static obstacles; through-hole pads stamped as vias).
        * ``netlist`` is a list of `(start_pad, end_pad)` pairs (MST-derived
          for multi-pad nets).
        * ``ctx`` is the opaque `KiCadContext` to pass to `save_routed_board`.
    """
    if not _HAVE_KIUTILS:
        raise ImportError(
            "kicad_io requires `kiutils`. Install with `pip install kiutils`."
        )

    board = kiutils.board.Board.from_file(path)
    return _load(
        board,
        grid_mm=grid_mm,
        clearance_mm=clearance_mm,
        margin_mm=margin_mm,
        trace_width_mm=trace_width_mm,
        via_size_mm=via_size_mm,
        via_drill_mm=via_drill_mm,
    )


def save_routed_board(
    ctx: KiCadContext,
    summary: Any,  # RouteSummary or BestOfSummary — both are TypedDicts
    output_path: str,
) -> None:
    """Write the routed result back to `output_path` as a `.kicad_pcb` file.

    Each routed net becomes one or more `(segment ...)` items per layer plus
    a `(via ...)` item for every layer transition. Existing pre-routed traces
    in the input board are preserved. Unrouted nets are silently omitted —
    the caller is expected to report them.
    """
    if not _HAVE_KIUTILS:
        raise ImportError(
            "kicad_io requires `kiutils`. Install with `pip install kiutils`."
        )

    board = ctx.board

    for net in summary["routed"]:
        net_name = ctx.pair_to_net_name.get(net["pair"], "")
        net_number = ctx.net_name_to_number.get(net_name, 0)
        _emit_path_items(
            board, net["path"], ctx, net_number,
        )

    board.to_file(output_path)


# ---------------------------------------------------------------------------
# Loading internals
# ---------------------------------------------------------------------------


def _load(
    board,
    *,
    grid_mm: float,
    clearance_mm: float,
    margin_mm: float,
    trace_width_mm: float,
    via_size_mm: float,
    via_drill_mm: float,
) -> tuple[PCBGrid, list[tuple], KiCadContext]:
    copper_layers = _copper_layer_names(board)
    num_layers = len(copper_layers)
    if num_layers == 0:
        raise ValueError("No copper layers found in board")

    pads_by_net = _collect_pads_by_net(board, copper_layers)
    all_pad_positions_mm = [
        (px, py) for pads in pads_by_net.values() for (px, py, _) in pads
    ]

    if not all_pad_positions_mm:
        raise ValueError("Board has no pads to route")

    min_x = min(p[0] for p in all_pad_positions_mm) - margin_mm
    min_y = min(p[1] for p in all_pad_positions_mm) - margin_mm
    max_x = max(p[0] for p in all_pad_positions_mm) + margin_mm
    max_y = max(p[1] for p in all_pad_positions_mm) + margin_mm

    grid_w = max(1, int(math.ceil((max_x - min_x) / grid_mm)) + 1)
    grid_h = max(1, int(math.ceil((max_y - min_y) / grid_mm)) + 1)

    clearance_cells = int(math.ceil(clearance_mm / grid_mm))
    grid = PCBGrid(
        grid_w, grid_h,
        num_layers=num_layers,
        clearance=clearance_cells,
    )

    # Stamp pads onto the grid as static obstacles. SMD on one layer;
    # through-hole as a via on all copper layers.
    pad_cells: dict[str, list[RPad]] = {}
    for net_name, pads in pads_by_net.items():
        for (px, py, layers) in pads:
            cx, cy = _mm_to_cell(px, py, min_x, min_y, grid_mm)
            cx = _clamp(cx, 0, grid_w - 1)
            cy = _clamp(cy, 0, grid_h - 1)
            if "*.Cu" in layers:
                layer_idxs: tuple[int, ...] = tuple(range(num_layers))
            else:
                resolved = [
                    _layer_name_to_index(L, copper_layers) for L in layers
                ]
                layer_idxs = tuple(i for i in resolved if i is not None)
            if not layer_idxs:
                continue  # pad on non-copper layer (e.g. paste); skip

            if len(layer_idxs) > 1:
                grid.add_via(cx, cy, layer_idxs)
            else:
                grid.add_obstacle(cx, cy, layer_idxs[0])

            pad_cells.setdefault(net_name, []).append(RPad(cx, cy, layer_idxs))

    # Stamp existing traces / vias as static obstacles so we route around them.
    for item in getattr(board, "traceItems", []):
        if hasattr(item, "start") and hasattr(item, "end"):
            _stamp_existing_segment(grid, item, min_x, min_y, grid_mm, copper_layers)
        elif hasattr(item, "drill") and hasattr(item, "layers"):
            _stamp_existing_via(grid, item, min_x, min_y, grid_mm, copper_layers)

    # Build the netlist: one MST per multi-pad net.
    netlist: list[tuple] = []
    pair_to_net_name: dict[tuple, str] = {}
    for net_name, net_pads in pad_cells.items():
        if not net_name:
            continue
        for pair in _mst_pairs(net_pads):
            netlist.append(pair)
            pair_to_net_name[pair] = net_name

    net_name_to_number = {
        net.name: net.number for net in board.nets if net.name
    }

    ctx = KiCadContext(
        board=board,
        grid_mm=grid_mm,
        origin_x_mm=min_x,
        origin_y_mm=min_y,
        copper_layers=copper_layers,
        trace_width_mm=trace_width_mm,
        via_size_mm=via_size_mm,
        via_drill_mm=via_drill_mm,
        net_name_to_number=net_name_to_number,
        pair_to_net_name=pair_to_net_name,
    )
    return grid, netlist, ctx


def _copper_layer_names(board) -> list[str]:
    """Return copper layer names in stack-up order ``["F.Cu", "In1.Cu", ..., "B.Cu"]``.

    Filters the board's full layer list (which includes silk/mask/etc.) down
    to copper layers. KiCad already lists them in stack-up order.
    """
    out: list[str] = []
    for L in board.layers:
        name = L.name
        if name == "F.Cu" or name == "B.Cu" or (
            name.startswith("In") and name.endswith(".Cu")
        ):
            out.append(name)
    # Reorder: F.Cu first, then In1, In2, ..., then B.Cu last.
    front = [n for n in out if n == "F.Cu"]
    inner = sorted(
        [n for n in out if n.startswith("In")],
        key=lambda n: int(n[2:-3]),  # strip "In" and ".Cu"
    )
    back = [n for n in out if n == "B.Cu"]
    return front + inner + back


def _layer_name_to_index(name: str, copper_layers: list[str]) -> Optional[int]:
    try:
        return copper_layers.index(name)
    except ValueError:
        return None


def _layer_index_to_name(idx: int, copper_layers: list[str]) -> str:
    return copper_layers[idx]


def _collect_pads_by_net(
    board, copper_layers: list[str]
) -> dict[str, list[tuple[float, float, list[str]]]]:
    """Walk all footprints; group pads by net name.

    Returns ``{net_name: [(x_mm, y_mm, layer_names), ...]}``.
    Pads with no assigned net are bucketed under the empty string.
    """
    result: dict[str, list[tuple[float, float, list[str]]]] = {}
    for fp in getattr(board, "footprints", []):
        fp_x = fp.position.X
        fp_y = fp.position.Y
        fp_angle = math.radians(fp.position.angle or 0.0)
        cos_a, sin_a = math.cos(fp_angle), math.sin(fp_angle)
        for pad in fp.pads:
            # Pad position is relative to the footprint; rotate then translate.
            rel_x = pad.position.X
            rel_y = pad.position.Y
            abs_x = fp_x + (rel_x * cos_a - rel_y * sin_a)
            abs_y = fp_y + (rel_x * sin_a + rel_y * cos_a)
            net_name = pad.net.name if pad.net else ""
            result.setdefault(net_name, []).append(
                (abs_x, abs_y, list(pad.layers))
            )
    return result


def _stamp_existing_segment(
    grid: PCBGrid, seg, ox: float, oy: float, grid_mm: float,
    copper_layers: list[str],
) -> None:
    """Rasterize a KiCad `Segment` onto the grid as static obstacles."""
    layer_idx = _layer_name_to_index(seg.layer, copper_layers)
    if layer_idx is None:
        return  # non-copper segment (silk, etc.)
    x0, y0 = _mm_to_cell(seg.start.X, seg.start.Y, ox, oy, grid_mm)
    x1, y1 = _mm_to_cell(seg.end.X, seg.end.Y, ox, oy, grid_mm)
    for x, y in _line_cells(x0, y0, x1, y1):
        if 0 <= x < grid.width and 0 <= y < grid.height:
            grid.add_obstacle(x, y, layer=layer_idx)


def _stamp_existing_via(
    grid: PCBGrid, via, ox: float, oy: float, grid_mm: float,
    copper_layers: list[str],
) -> None:
    x, y = _mm_to_cell(via.position.X, via.position.Y, ox, oy, grid_mm)
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return
    resolved = [_layer_name_to_index(L, copper_layers) for L in via.layers]
    layer_idxs = tuple(i for i in resolved if i is not None)
    if layer_idxs:
        grid.add_via(x, y, layer_idxs)


def _mm_to_cell(
    mm_x: float, mm_y: float, ox: float, oy: float, grid_mm: float,
) -> tuple[int, int]:
    """Convert mm coordinates (relative to board origin) to grid cells."""
    return (
        int(round((mm_x - ox) / grid_mm)),
        int(round((mm_y - oy) / grid_mm)),
    )


def _cell_to_mm(
    cell_x: int, cell_y: int, ox: float, oy: float, grid_mm: float,
) -> tuple[float, float]:
    return (cell_x * grid_mm + ox, cell_y * grid_mm + oy)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _line_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham-ish 4-connected rasterization of a line between cells."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        elif e2 < dx:
            err += dx
            y += sy
    return cells


# ---------------------------------------------------------------------------
# Net-pair extraction (MST)
# ---------------------------------------------------------------------------


def _mst_pairs(pads: list[RPad]) -> list[tuple[RPad, RPad]]:
    """Kruskal's MST over pad positions (Manhattan distance) -> N-1 edges.

    Treats every pad as a node; the spanning tree of N pads needs N-1 edges
    to connect them. Each edge becomes one router pair.
    """
    n = len(pads)
    if n < 2:
        return []

    edges: list[tuple[int, int, int]] = []  # (distance, i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(pads[i].x - pads[j].x) + abs(pads[i].y - pads[j].y)
            edges.append((d, i, j))
    edges.sort()

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs: list[tuple[RPad, RPad]] = []
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[ri] = rj
        pairs.append((pads[i], pads[j]))
        if len(pairs) == n - 1:
            break
    return pairs


# ---------------------------------------------------------------------------
# Writing internals
# ---------------------------------------------------------------------------


def _emit_path_items(board, path, ctx: KiCadContext, net_number: int) -> None:
    """Convert a routed path (list of 3-tuples) into Segment + Via items.

    Adjacent same-layer cells produce one Segment per run; layer transitions
    produce one Via at the transition cell. We collapse straight runs of cells
    into a single segment whose endpoints are the run's start and end (rather
    than one tiny segment per cell, which would explode the output file).
    """
    if not path:
        return

    grid_mm = ctx.grid_mm
    ox, oy = ctx.origin_x_mm, ctx.origin_y_mm
    copper_layers = ctx.copper_layers

    # Walk the path, grouping consecutive same-layer cells into runs.
    runs: list[tuple[int, list[tuple[int, int]]]] = []  # (layer, [(x,y)...])
    current_layer: Optional[int] = None
    current_run: list[tuple[int, int]] = []
    for cell in path:
        if len(cell) == 2:
            x, y, z = cell[0], cell[1], 0
        else:
            x, y, z = cell[0], cell[1], cell[2]
        if z != current_layer:
            if current_run:
                runs.append((current_layer, current_run))  # type: ignore[arg-type]
            current_layer = z
            current_run = [(x, y)]
        else:
            current_run.append((x, y))
    if current_run and current_layer is not None:
        runs.append((current_layer, current_run))

    # Emit segments: for each run, find direction-change points and emit a
    # segment between each pair.
    for layer_idx, run in runs:
        for (x0, y0), (x1, y1) in _segment_endpoints(run):
            sx, sy = _cell_to_mm(x0, y0, ox, oy, grid_mm)
            ex, ey = _cell_to_mm(x1, y1, ox, oy, grid_mm)
            board.traceItems.append(Segment(
                start=Position(X=sx, Y=sy),
                end=Position(X=ex, Y=ey),
                width=ctx.trace_width_mm,
                layer=_layer_index_to_name(layer_idx, copper_layers),
                net=net_number,
                tstamp=_new_tstamp(),
            ))

    # Emit vias at layer transitions.
    for i in range(len(runs) - 1):
        prev_layer, prev_run = runs[i]
        next_layer, next_run = runs[i + 1]
        # The transition cell is the last cell of prev_run == first cell of next_run.
        tx, ty = prev_run[-1]
        mx, my = _cell_to_mm(tx, ty, ox, oy, grid_mm)
        board.traceItems.append(Via(
            position=Position(X=mx, Y=my),
            size=ctx.via_size_mm,
            drill=ctx.via_drill_mm,
            layers=[
                _layer_index_to_name(prev_layer, copper_layers),
                _layer_index_to_name(next_layer, copper_layers),
            ],
            net=net_number,
            tstamp=_new_tstamp(),
        ))


def _segment_endpoints(
    run: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Collapse a run of cells into the minimum number of straight segments.

    ``[(0,0),(1,0),(2,0),(2,1),(2,2)]`` collapses to
    ``[((0,0),(2,0)), ((2,0),(2,2))]`` — two segments instead of four.
    """
    if len(run) < 2:
        return []
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    start = run[0]
    prev = run[0]
    prev_dir: Optional[tuple[int, int]] = None
    for i in range(1, len(run)):
        cur = run[i]
        d = (cur[0] - prev[0], cur[1] - prev[1])
        if prev_dir is None:
            prev_dir = d
        elif d != prev_dir:
            out.append((start, prev))
            start = prev
            prev_dir = d
        prev = cur
    out.append((start, prev))
    return out


def _new_tstamp() -> str:
    return str(uuid.uuid4())
