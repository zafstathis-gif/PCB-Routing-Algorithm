"""Unit tests for KiCad `.kicad_pcb` I/O — loading, routing, saving."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import kiutils.board  # noqa: E402

from kicad_io import (  # noqa: E402
    _mm_to_cell,
    _segment_endpoints,
    load_board,
    save_routed_board,
)
from router import route_board, route_board_best_of  # noqa: E402

# Import the example builders for self-contained tests.
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples"))
from build_examples import build_blinker, build_two_layer_demo  # noqa: E402


class TestUnitConversion(unittest.TestCase):
    def test_mm_to_cell_round_to_nearest(self) -> None:
        # ox=0, oy=0, grid=0.5 -> (0.6, 0.4) -> (1, 1) and (0.5, 0.5)?
        self.assertEqual(_mm_to_cell(0.6, 0.4, 0.0, 0.0, 0.5), (1, 1))
        # 0.5 rounds to nearest even by Python's round(); good enough.

    def test_mm_to_cell_origin_offset(self) -> None:
        # Origin at (10, 20). A point at (12, 21) is (2, 1) mm relative ->
        # (4, 2) cells at 0.5mm grid.
        self.assertEqual(_mm_to_cell(12.0, 21.0, 10.0, 20.0, 0.5), (4, 2))


class TestSegmentEndpoints(unittest.TestCase):
    def test_collapse_straight_horizontal_run(self) -> None:
        run = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(_segment_endpoints(run), [((0, 0), (3, 0))])

    def test_collapse_at_direction_change(self) -> None:
        # Horizontal then vertical -> 2 segments.
        run = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        self.assertEqual(
            _segment_endpoints(run),
            [((0, 0), (2, 0)), ((2, 0), (2, 2))],
        )

    def test_single_cell_returns_empty(self) -> None:
        self.assertEqual(_segment_endpoints([(5, 5)]), [])


class TestLoadBlinker(unittest.TestCase):
    """Load the blinker example board built in `examples/build_examples.py`."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "blinker.kicad_pcb")
        build_blinker().to_file(self.path)

    def test_load_has_three_nets(self) -> None:
        grid, netlist, ctx = load_board(
            self.path, grid_mm=0.5, clearance_mm=0.2,
        )
        # 3 nets each with 2 pads -> MST emits exactly 1 pair per net.
        self.assertEqual(len(netlist), 3)
        net_names = {ctx.pair_to_net_name[p] for p in netlist}
        self.assertEqual(net_names, {"VCC", "SIG", "GND"})

    def test_load_grid_dimensions_match_board(self) -> None:
        grid, _, _ = load_board(self.path, grid_mm=0.5, clearance_mm=0.0)
        # Components live in roughly an 8..24 x 6..14 mm rectangle + 2mm margin.
        # At 0.5mm/cell, grid should be on the order of 40x25.
        self.assertGreater(grid.width, 25)
        self.assertGreater(grid.height, 15)
        self.assertEqual(grid.num_layers, 2)  # default KiCad: F.Cu + B.Cu

    def test_pads_marked_as_static_obstacles(self) -> None:
        grid, netlist, _ = load_board(self.path, grid_mm=0.5)
        # Every pad coordinate from the netlist must be a static obstacle.
        for start, end in netlist:
            for pad in (start, end):
                for layer in pad.layers:
                    self.assertTrue(
                        grid.is_static_obstacle(pad.x, pad.y, layer),
                        f"Pad {pad} not stamped as static on layer {layer}",
                    )


class TestRoundTripRoute(unittest.TestCase):
    """Load -> route -> save, then re-load and check segments are present."""

    def test_blinker_round_trip(self) -> None:
        tmpdir = tempfile.mkdtemp()
        in_path = os.path.join(tmpdir, "blinker_in.kicad_pcb")
        out_path = os.path.join(tmpdir, "blinker_out.kicad_pcb")
        build_blinker().to_file(in_path)

        grid, netlist, ctx = load_board(in_path, grid_mm=0.5, clearance_mm=0.2)
        summary = route_board_best_of(grid, netlist, use_rrr=True)
        self.assertEqual(
            len(summary["routed"]), len(netlist),
            "Blinker should route 3/3 nets",
        )

        save_routed_board(ctx, summary, out_path)

        # Re-load the output and check segments exist with correct nets.
        reloaded = kiutils.board.Board.from_file(out_path)
        segments = [t for t in reloaded.traceItems if hasattr(t, "start")]
        self.assertGreater(len(segments), 0)

        # All segments should have a non-zero net number (i.e. assigned).
        for seg in segments:
            self.assertNotEqual(seg.net, 0, f"Segment {seg} has no net")

        # Segments should reference real net names from the input.
        net_numbers_used = {seg.net for seg in segments}
        valid_net_numbers = {n.number for n in reloaded.nets if n.name}
        self.assertTrue(net_numbers_used.issubset(valid_net_numbers))

    def test_two_layer_demo_routes(self) -> None:
        tmpdir = tempfile.mkdtemp()
        in_path = os.path.join(tmpdir, "demo_in.kicad_pcb")
        out_path = os.path.join(tmpdir, "demo_out.kicad_pcb")
        build_two_layer_demo().to_file(in_path)

        grid, netlist, ctx = load_board(in_path, grid_mm=0.5, clearance_mm=0.0)
        summary = route_board_best_of(grid, netlist)
        self.assertEqual(len(summary["routed"]), 2)

        save_routed_board(ctx, summary, out_path)
        # Output is a valid kicad_pcb file (kiutils round-trip).
        reloaded = kiutils.board.Board.from_file(out_path)
        self.assertGreaterEqual(
            sum(1 for t in reloaded.traceItems if hasattr(t, "start")),
            2,
        )

    def test_pre_routed_traces_treated_as_obstacles(self) -> None:
        # Route a board, save it, then load the saved board: the previously
        # routed segments are now obstacles in the new grid.
        tmpdir = tempfile.mkdtemp()
        in_path = os.path.join(tmpdir, "blinker.kicad_pcb")
        first_out = os.path.join(tmpdir, "first_routed.kicad_pcb")
        build_blinker().to_file(in_path)

        grid, netlist, ctx = load_board(in_path, grid_mm=0.5, clearance_mm=0.0)
        summary = route_board(grid, netlist)
        save_routed_board(ctx, summary, first_out)

        # Re-load: pre-existing traces become static obstacles. We don't try
        # to route again (the nets are already done) — we just verify the load
        # path picks up the existing segments without crashing.
        grid2, _, _ = load_board(first_out, grid_mm=0.5, clearance_mm=0.0)
        # The new grid should have *more* static obstacles than the original
        # (the traces are now static cells).
        original_static_cells = int(grid.static_mask.sum())
        reloaded_static_cells = int(grid2.static_mask.sum())
        self.assertGreater(reloaded_static_cells, original_static_cells)


class TestMSTPairs(unittest.TestCase):
    def test_three_pad_net_emits_two_pairs(self) -> None:
        from kicad_io import _mst_pairs
        from pcb_grid import Pad

        pads = [Pad(0, 0, (0,)), Pad(10, 0, (0,)), Pad(20, 0, (0,))]
        pairs = _mst_pairs(pads)
        self.assertEqual(len(pairs), 2)  # MST of 3 nodes = 2 edges
        # Should connect (0)-(10) and (10)-(20), not (0)-(20).
        connected = set()
        for a, b in pairs:
            connected.add((a.x, b.x))
            connected.add((b.x, a.x))
        self.assertIn((0, 10), connected)
        self.assertIn((10, 20), connected)

    def test_single_pad_net_emits_no_pairs(self) -> None:
        from kicad_io import _mst_pairs
        from pcb_grid import Pad
        self.assertEqual(_mst_pairs([Pad(5, 5, (0,))]), [])


if __name__ == "__main__":
    unittest.main()
