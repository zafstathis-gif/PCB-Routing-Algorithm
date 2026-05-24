"""Streamlit web demo for the PCB autorouter.

Run locally with:

    pip install -e .[web]
    streamlit run app.py

The app has two modes. "Random board" lets you configure board size, layers,
clearance, obstacle count and number of nets, then generates a board, routes
it, and shows the per-layer result. "KiCad upload" takes a `.kicad_pcb`
file, routes it, and offers the routed file as a download.
"""

from __future__ import annotations

import random
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

from pcb_grid import PCBGrid
from router import (
    ALL_STRATEGIES,
    NetPair,
    route_board,
    route_board_best_of,
    route_board_rrr,
)
from visualize import visualize_board

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PCB Autorouter",
    page_icon=":material/electrical_services:",
    layout="wide",
)

st.title("PCB Autorouter — Interactive Demo")
st.markdown(
    "A from-scratch multi-layer PCB autorouter with A\\*, rip-up-and-reroute, "
    "configurable DRC clearance, and learned net ordering. "
    "[Source on GitHub](https://github.com/zafstathis-gif/PCB-Routing-Algorithm)."
)

mode = st.radio(
    "Mode",
    ["Random board", "KiCad upload"],
    horizontal=True,
)


# ---------------------------------------------------------------------------
# Shared sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Routing options")
    strategy_name = st.selectbox(
        "Net-ordering strategy",
        options=["best_of"] + [s for s in ALL_STRATEGIES if s is not None] + ["user"],
        index=0,
        help="`best_of` runs every strategy and keeps the best result; "
             "individual strategies show what each one does in isolation.",
    )
    use_rrr = st.checkbox(
        "Rip-up-and-reroute",
        value=True,
        help="When a net fails, identify nets blocking its ideal path, "
             "rip them up, retry, and re-route the displaced nets.",
    )
    prefer_directions = st.checkbox(
        "Preferred directions",
        value=False,
        help="Bias even layers to horizontal moves, odd layers to vertical "
             "(classical EDA convention).",
    )


def _strategy_arg(name: str):
    """Map a UI strategy name back to the router's sort_strategy value."""
    if name == "user":
        return None
    return name


def _run_router(grid: PCBGrid, netlist):
    if strategy_name == "best_of":
        return route_board_best_of(
            grid, netlist, use_rrr=use_rrr, prefer_directions=prefer_directions,
        )
    runner = route_board_rrr if use_rrr else route_board
    return runner(
        grid, netlist, sort_strategy=_strategy_arg(strategy_name),
        prefer_directions=prefer_directions,
    )


# ---------------------------------------------------------------------------
# Mode 1: random board
# ---------------------------------------------------------------------------

if mode == "Random board":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Board geometry")
        board_size = st.slider("Board size (cells per side)", 10, 40, 20)
        num_layers = st.slider("Number of copper layers", 1, 4, 2)
        clearance = st.slider("Clearance (cells)", 0, 3, 1)
    with col2:
        st.subheader("Netlist")
        num_obstacles = st.slider("Static obstacles", 0, 80, 18)
        num_nets = st.slider("Nets to route", 1, 12, 6)
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Generate + route", type="primary"):
        rng = random.Random(int(seed))

        grid = PCBGrid(
            board_size, board_size, num_layers=num_layers, clearance=clearance,
        )
        # Drop obstacles on layer 0.
        placed = 0
        while placed < num_obstacles:
            x, y = rng.randint(0, board_size - 1), rng.randint(0, board_size - 1)
            if grid.is_valid(x, y, 0):
                grid.add_obstacle(x, y, layer=0)
                placed += 1

        # Build a netlist on layer 0. Pin-pairs avoid overlapping pads.
        pin_used: set[tuple[int, int]] = set()
        netlist: list[NetPair] = []
        while len(netlist) < num_nets:
            sx, sy = rng.randint(0, board_size - 1), rng.randint(0, board_size - 1)
            ex, ey = rng.randint(0, board_size - 1), rng.randint(0, board_size - 1)
            if (sx, sy) == (ex, ey):
                continue
            if not (grid.is_valid(sx, sy, 0) and grid.is_valid(ex, ey, 0)):
                continue
            if (sx, sy) in pin_used or (ex, ey) in pin_used:
                continue
            pin_used.add((sx, sy))
            pin_used.add((ex, ey))
            if num_layers > 1:
                netlist.append(((sx, sy, 0), (ex, ey, 0)))
            else:
                netlist.append(((sx, sy), (ex, ey)))

        with st.spinner("Routing..."):
            summary = _run_router(grid, netlist)

        routed = len(summary["routed"])
        total = len(netlist)
        pct = 100 * routed / total
        if routed == total:
            st.success(f"Routed {routed}/{total} ({pct:.0f}%)")
        else:
            st.warning(f"Routed {routed}/{total} ({pct:.0f}%) — "
                       f"{total - routed} unrouted")

        total_cells = sum(len(n["path"]) for n in summary["routed"])
        total_vias = sum(len(n.get("vias", [])) for n in summary["routed"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nets routed", f"{routed} / {total}")
        c2.metric("Total trace cells", total_cells)
        c3.metric("Vias placed", total_vias)
        if "strategy" in summary:
            c4.metric("Winning strategy", str(summary["strategy"]))

        # Render. visualize_board calls plt.show() internally; we capture
        # the figure via plt.gcf() right before that.
        plt.close("all")
        visualize_board(grid, summary, title="")
        fig = plt.gcf()
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# Mode 2: KiCad upload
# ---------------------------------------------------------------------------

else:  # KiCad upload
    st.subheader("KiCad .kicad_pcb upload")

    st.markdown(
        "Drop in a `.kicad_pcb` file with **pads but no traces**. The app will "
        "route it and offer the routed file back. Behind the scenes it calls "
        "`kicad_io.load_board` → `router.route_board_best_of` → "
        "`kicad_io.save_routed_board`. See "
        "[`examples/`](https://github.com/zafstathis-gif/PCB-Routing-Algorithm/tree/main/examples) "
        "for sample inputs."
    )

    col1, col2 = st.columns(2)
    with col1:
        grid_mm = st.number_input("Routing grid (mm/cell)", value=0.5, step=0.1)
        clearance_mm = st.number_input("Clearance (mm)", value=0.2, step=0.1)
    with col2:
        trace_width_mm = st.number_input("Trace width (mm)", value=0.25, step=0.05)
        via_size_mm = st.number_input("Via outer diameter (mm)", value=0.6, step=0.1)

    uploaded = st.file_uploader("Upload .kicad_pcb", type=["kicad_pcb"])

    if uploaded is not None and st.button("Route this board", type="primary"):
        # kiutils only reads from a file path, so spool the upload to a tmpdir.
        from kicad_io import load_board, save_routed_board

        with tempfile.TemporaryDirectory() as td:
            in_path = f"{td}/in.kicad_pcb"
            out_path = f"{td}/out.kicad_pcb"
            with open(in_path, "wb") as fout:
                fout.write(uploaded.getbuffer())

            try:
                grid, netlist, ctx = load_board(
                    in_path,
                    grid_mm=grid_mm,
                    clearance_mm=clearance_mm,
                    trace_width_mm=trace_width_mm,
                    via_size_mm=via_size_mm,
                )
            except Exception as e:
                st.error(f"Failed to load board: {e}")
                st.stop()

            st.info(
                f"Loaded: {grid.num_layers} copper layers, "
                f"{grid.width}×{grid.height} cells, "
                f"{len(netlist)} pin-pairs "
                f"from {len(set(ctx.pair_to_net_name.values()))} nets."
            )

            with st.spinner("Routing..."):
                summary = _run_router(grid, netlist)

            routed = len(summary["routed"])
            total = len(netlist)
            if routed == total:
                st.success(f"Routed {routed}/{total} (100%)")
            else:
                st.warning(
                    f"Routed {routed}/{total} ({100*routed/total:.0f}%); "
                    f"the unrouted nets are listed in the file as un-traced "
                    "(open in KiCad to inspect)."
                )

            save_routed_board(ctx, summary, out_path)

            with open(out_path, "rb") as fin:
                routed_bytes = fin.read()

            st.download_button(
                "Download routed .kicad_pcb",
                data=routed_bytes,
                file_name=uploaded.name.replace(".kicad_pcb", "_routed.kicad_pcb"),
                mime="application/octet-stream",
            )

            # Show the same per-layer visualization as the random mode.
            plt.close("all")
            visualize_board(grid, summary, title="")
            fig = plt.gcf()
            st.pyplot(fig)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Author: Eustathios Zafeiropoulos · ECE student, Aristotle University of "
    "Thessaloniki · "
    "[GitHub](https://github.com/zafstathis-gif/PCB-Routing-Algorithm)"
)
