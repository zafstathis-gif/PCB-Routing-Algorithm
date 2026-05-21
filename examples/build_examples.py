"""Build small synthetic `.kicad_pcb` files for testing and demos.

Real KiCad projects would normally include a `.kicad_pro` and footprint
libraries, but kiutils can read/write the `.kicad_pcb` board file on its own,
which is enough for routing.

Run from repo root:  python examples/build_examples.py
Outputs:             examples/blinker_unrouted.kicad_pcb
                     examples/two_layer_demo.kicad_pcb
"""

from __future__ import annotations

import os

import kiutils.board
from kiutils.footprint import Footprint, Pad
from kiutils.items.common import Net, Position

_HERE = os.path.dirname(os.path.abspath(__file__))


def _smd_pad(num: str, x: float, y: float, net: Net, layer: str = "F.Cu") -> Pad:
    return Pad(
        number=num, type="smd", shape="rect",
        position=Position(X=x, Y=y),
        size=Position(X=0.8, Y=0.8),
        layers=[layer],
        net=net,
    )


def _tht_pad(num: str, x: float, y: float, net: Net) -> Pad:
    return Pad(
        number=num, type="thru_hole", shape="circle",
        position=Position(X=x, Y=y),
        size=Position(X=1.0, Y=1.0),
        layers=["*.Cu"],
        net=net,
    )


def build_blinker() -> kiutils.board.Board:
    """A trivial "blinker" board: a microcontroller pin -> LED -> ground, etc.

    Three components in a 30 mm × 20 mm layout:
        U1 (4-pin MCU)
        R1 (2-pin resistor)
        D1 (2-pin LED)

    Nets:
        VCC: U1.1, R1.1
        SIG: R1.2, D1.1
        GND: U1.4, D1.2
    """
    b = kiutils.board.Board.create_new()

    vcc = Net(number=1, name="VCC")
    sig = Net(number=2, name="SIG")
    gnd = Net(number=3, name="GND")
    b.nets.extend([vcc, sig, gnd])

    u1 = Footprint(
        entryName="U1", layer="F.Cu", tedit="00000000",
        position=Position(X=8.0, Y=10.0),
        pads=[
            _smd_pad("1", -1.0, -1.5, vcc),
            _smd_pad("2", 1.0, -1.5, Net(number=0, name="")),
            _smd_pad("3", 1.0, 1.5, Net(number=0, name="")),
            _smd_pad("4", -1.0, 1.5, gnd),
        ],
    )
    r1 = Footprint(
        entryName="R1", layer="F.Cu", tedit="00000000",
        position=Position(X=18.0, Y=6.0),
        pads=[
            _smd_pad("1", -1.5, 0.0, vcc),
            _smd_pad("2", 1.5, 0.0, sig),
        ],
    )
    d1 = Footprint(
        entryName="D1", layer="F.Cu", tedit="00000000",
        position=Position(X=24.0, Y=14.0),
        pads=[
            _smd_pad("1", -1.0, 0.0, sig),
            _smd_pad("2", 1.0, 0.0, gnd),
        ],
    )
    b.footprints = [u1, r1, d1]
    return b


def build_two_layer_demo() -> kiutils.board.Board:
    """Force a multi-layer route: two SMD nets crossing each other.

    Two nets in an X shape; on a single layer they would have to detour
    significantly. With two copper layers, one of them can use a via.
    """
    b = kiutils.board.Board.create_new()
    a = Net(number=1, name="A")
    bnet = Net(number=2, name="B")
    b.nets.extend([a, bnet])

    fp1 = Footprint(
        entryName="L1", layer="F.Cu", tedit="00000000",
        position=Position(X=5.0, Y=5.0),
        pads=[_smd_pad("1", 0.0, 0.0, a)],
    )
    fp2 = Footprint(
        entryName="L2", layer="F.Cu", tedit="00000000",
        position=Position(X=25.0, Y=25.0),
        pads=[_smd_pad("1", 0.0, 0.0, a)],
    )
    fp3 = Footprint(
        entryName="L3", layer="F.Cu", tedit="00000000",
        position=Position(X=5.0, Y=25.0),
        pads=[_smd_pad("1", 0.0, 0.0, bnet)],
    )
    fp4 = Footprint(
        entryName="L4", layer="F.Cu", tedit="00000000",
        position=Position(X=25.0, Y=5.0),
        pads=[_smd_pad("1", 0.0, 0.0, bnet)],
    )
    b.footprints = [fp1, fp2, fp3, fp4]
    return b


def main() -> None:
    blinker_path = os.path.join(_HERE, "blinker_unrouted.kicad_pcb")
    build_blinker().to_file(blinker_path)
    print(f"wrote {blinker_path}")

    demo_path = os.path.join(_HERE, "two_layer_demo.kicad_pcb")
    build_two_layer_demo().to_file(demo_path)
    print(f"wrote {demo_path}")


if __name__ == "__main__":
    main()
