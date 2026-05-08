"""PCB grid environment for the AI PCB routing algorithm."""

from __future__ import annotations

import numpy as np


EMPTY: int = 0
OBSTACLE: int = 1


class PCBGrid:
    """A 2D grid representing a PCB routing environment.

    Each cell is either empty (0) and routable, or an obstacle (1).
    """

    def __init__(self, width: int, height: int) -> None:
        """Initialize an empty grid of the given dimensions.

        Args:
            width: Number of columns (x-axis size). Must be positive.
            height: Number of rows (y-axis size). Must be positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        self.width: int = width
        self.height: int = height
        self.grid: np.ndarray = np.zeros((height, width), dtype=np.int8)

    def add_obstacle(self, x: int, y: int) -> None:
        """Mark the cell at (x, y) as an obstacle.

        Args:
            x: Column index.
            y: Row index.
        """
        if not self._in_bounds(x, y):
            raise IndexError(
                f"({x}, {y}) is outside the {self.width}x{self.height} grid"
            )
        self.grid[y, x] = OBSTACLE

    def is_valid(self, x: int, y: int) -> bool:
        """Return True if (x, y) is in-bounds and not an obstacle."""
        if not self._in_bounds(x, y):
            return False
        return bool(self.grid[y, x] == EMPTY)

    def clone(self) -> "PCBGrid":
        """Return an independent copy of this grid."""
        copy = PCBGrid(self.width, self.height)
        copy.grid = self.grid.copy()
        return copy

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def __repr__(self) -> str:
        return f"PCBGrid(width={self.width}, height={self.height})"

    def __str__(self) -> str:
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)


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
