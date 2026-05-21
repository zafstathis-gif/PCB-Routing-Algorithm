"""PCB grid environment for the AI PCB routing algorithm.

Supports 1 or more routing layers. The canonical storage is a 3D `int8` array
of shape `(num_layers, height, width)`. For back-compat, single-layer boards
keep their original 2D API: `g.grid` exposes a view of layer 0, and
`is_valid(x, y)` / `add_obstacle(x, y)` default to layer 0.

Vias connect cells across layers at the same `(x, y)`. Through-hole pads are
modelled as vias spanning every layer they touch.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional

import numpy as np


EMPTY: int = 0
OBSTACLE: int = 1


class Pad(NamedTuple):
    """A pad's (x, y) position and the set of layers it occupies.

    SMD pad on top layer:   Pad(3, 5, (0,))
    Through-hole, 4-layer:  Pad(3, 5, (0, 1, 2, 3))
    """
    x: int
    y: int
    layers: tuple[int, ...]


class PCBGrid:
    """A PCB routing environment of `num_layers` stacked grids.

    Each cell on each layer is either EMPTY (0) and routable, or OBSTACLE (1).
    Vias join layers at the same `(x, y)` and are tracked in `self.vias`.
    `clearance` is the minimum spacing (in cells) maintained between locked
    traces — read by the router when stamping paths (see `stamp_path`).
    """

    def __init__(
        self,
        width: int,
        height: int,
        num_layers: int = 1,
        clearance: int = 0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if clearance < 0:
            raise ValueError("clearance must be non-negative")

        self.width: int = width
        self.height: int = height
        self.num_layers: int = num_layers
        self.clearance: int = clearance
        self.layers: np.ndarray = np.zeros(
            (num_layers, height, width), dtype=np.int8
        )
        # Cells that are *immutable* obstacles (pads, board outline, drill
        # holes, user-placed obstacles). Trace halos do not touch this mask,
        # so pin-clearing can safely temporarily-free trace cells around a pin
        # without erasing real static obstacles next to it.
        self.static_mask: np.ndarray = np.zeros(
            (num_layers, height, width), dtype=bool
        )
        self.vias: set[tuple[int, int]] = set()

    # ---- 2D back-compat view ------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """A 2D view of layer 0. Mutations write through to `self.layers[0]`.

        Provided so single-layer callers (the bulk of the original codebase
        and its tests) can continue to use `grid.grid[y, x]` unchanged.
        """
        return self.layers[0]

    def layer(self, z: int) -> np.ndarray:
        """Return a 2D view of layer `z` (writes propagate to `self.layers`)."""
        if not 0 <= z < self.num_layers:
            raise IndexError(f"layer {z} out of range [0, {self.num_layers})")
        return self.layers[z]

    # ---- obstacle / via mutation -------------------------------------------

    def add_obstacle(self, x: int, y: int, layer: int = 0) -> None:
        """Mark `(x, y)` on `layer` as a static obstacle."""
        if not self._in_bounds(x, y):
            raise IndexError(
                f"({x}, {y}) is outside the {self.width}x{self.height} grid"
            )
        if not 0 <= layer < self.num_layers:
            raise IndexError(
                f"layer {layer} out of range [0, {self.num_layers})"
            )
        self.layers[layer, y, x] = OBSTACLE
        self.static_mask[layer, y, x] = True

    def add_via(self, x: int, y: int, layers: Iterable[int]) -> None:
        """Reserve column `(x, y)` on every listed layer and record the via.

        Used for through-hole pads (all layers) and for vias introduced by the
        router during multi-layer routing.
        """
        if not self._in_bounds(x, y):
            raise IndexError(
                f"({x}, {y}) is outside the {self.width}x{self.height} grid"
            )
        for z in layers:
            if not 0 <= z < self.num_layers:
                raise IndexError(
                    f"layer {z} out of range [0, {self.num_layers})"
                )
            self.layers[z, y, x] = OBSTACLE
            self.static_mask[z, y, x] = True
        self.vias.add((x, y))

    # ---- trace stamping (halo-aware) ---------------------------------------

    def stamp_path(
        self,
        path: Iterable[tuple[int, ...]],
        *,
        halo: Optional[int] = None,
    ) -> None:
        """Mark every cell of `path` (plus a Chebyshev `halo` buffer) as OBSTACLE.

        `path` cells may be 2-tuples (single-layer back-compat, default layer
        0) or 3-tuples `(x, y, layer)`. `halo` defaults to `self.clearance`;
        pass `halo=0` to stamp without any clearance buffer (used by R&R's
        ideal-path computation).

        Trace stamping never sets `static_mask` — only `add_obstacle` /
        `add_via` do.
        """
        if halo is None:
            halo = self.clearance
        for cell in path:
            if len(cell) == 2:
                x, y = cell
                z = 0
            else:
                x, y, z = cell[0], cell[1], cell[2]
            x0 = max(0, x - halo)
            x1 = min(self.width, x + halo + 1)
            y0 = max(0, y - halo)
            y1 = min(self.height, y + halo + 1)
            self.layers[z, y0:y1, x0:x1] = OBSTACLE

    # ---- queries -----------------------------------------------------------

    def is_valid(self, x: int, y: int, layer: int = 0) -> bool:
        """Return True if `(x, y, layer)` is in-bounds and not an obstacle."""
        if not self._in_bounds(x, y):
            return False
        if not 0 <= layer < self.num_layers:
            return False
        return bool(self.layers[layer, y, x] == EMPTY)

    def is_static_obstacle(self, x: int, y: int, layer: int = 0) -> bool:
        """Return True if `(x, y, layer)` is a static obstacle (pad/wall/via)."""
        if not self._in_bounds(x, y):
            return False
        if not 0 <= layer < self.num_layers:
            return False
        return bool(self.static_mask[layer, y, x])

    # ---- cloning -----------------------------------------------------------

    def clone(self) -> "PCBGrid":
        """Return an independent copy of this grid (layers, vias, clearance)."""
        copy = PCBGrid(
            self.width, self.height,
            num_layers=self.num_layers,
            clearance=self.clearance,
        )
        copy.layers = self.layers.copy()
        copy.static_mask = self.static_mask.copy()
        copy.vias = set(self.vias)
        return copy

    # ---- internals ---------------------------------------------------------

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def __repr__(self) -> str:
        if self.num_layers == 1:
            return f"PCBGrid(width={self.width}, height={self.height})"
        return (
            f"PCBGrid(width={self.width}, height={self.height}, "
            f"num_layers={self.num_layers})"
        )

    def __str__(self) -> str:
        if self.num_layers == 1:
            return "\n".join(
                " ".join(str(cell) for cell in row) for row in self.layers[0]
            )
        parts = []
        for z in range(self.num_layers):
            parts.append(f"--- layer {z} ---")
            parts.append("\n".join(
                " ".join(str(cell) for cell in row) for row in self.layers[z]
            ))
        return "\n".join(parts)


if __name__ == "__main__":
    grid = PCBGrid(20, 20)
    grid.add_obstacle(5, 5)
    grid.add_obstacle(5, 6)
    grid.add_obstacle(6, 5)

    print(grid)
    print()
    print(f"(0, 0)  valid? {grid.is_valid(0, 0)}")
    print(f"(5, 5)  valid? {grid.is_valid(5, 5)}")
    print(f"(20, 0) valid? {grid.is_valid(20, 0)}")
