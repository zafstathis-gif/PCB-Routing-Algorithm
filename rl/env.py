"""Gym-style environment for sequential PCB net routing.

State at each step: (current obstacle grid, list of remaining net pairs).
Action:             index into the list of remaining nets — which one to route
                    next.
Reward:             +1 if A* finds a path for the chosen net, 0 otherwise.
Episode ends when every net in the netlist has been attempted.

The episode return is therefore equal to the number of successfully routed
nets — the same metric the heuristic baselines optimize, so policy and
heuristics are directly comparable.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

from pcb_grid import PCBGrid
from router import (
    NetPair,
    RoutedNet,
    _stamp_path_on_grid,
    _try_route_with_pin_clear,
)


class Observation(TypedDict):
    board: np.ndarray
    remaining: list[NetPair]


class RoutingEnv:
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        num_layers: int = 1,
    ) -> None:
        self.width = width
        self.height = height
        self.num_layers = num_layers
        self.grid: Optional[PCBGrid] = None
        self.remaining: list[NetPair] = []
        self.routed: list[RoutedNet] = []
        self.unrouted: list[NetPair] = []

    def reset(self, grid: PCBGrid, netlist: list[NetPair]) -> Observation:
        """Start a new episode. The provided grid is cloned (not mutated)."""
        self.grid = grid.clone()
        self.remaining = list(netlist)
        self.routed = []
        self.unrouted = []
        return self._observation()

    def step(self, action_idx: int) -> tuple[Observation, float, bool]:
        """Route the net at index `action_idx` from the remaining list."""
        if self.grid is None:
            raise RuntimeError("Call reset() before step().")
        if not self.remaining:
            raise RuntimeError("Episode is already done; call reset().")
        if not 0 <= action_idx < len(self.remaining):
            raise IndexError(f"action {action_idx} out of range for "
                             f"{len(self.remaining)} remaining nets")

        pair = self.remaining.pop(action_idx)
        path = _try_route_with_pin_clear(self.grid, pair)

        if path is None:
            self.unrouted.append(pair)
            reward = 0.0
        else:
            self.routed.append({"pair": pair, "path": path})
            _stamp_path_on_grid(self.grid, path)
            reward = 1.0

        done = len(self.remaining) == 0
        return self._observation(), reward, done

    def _observation(self) -> Observation:
        assert self.grid is not None
        # Single-layer back-compat: expose the 2D layer-0 view.
        # Multi-layer: expose the full 3D layer stack.
        if self.grid.num_layers == 1:
            board = self.grid.grid.copy()
        else:
            board = self.grid.layers.copy()
        return {"board": board, "remaining": list(self.remaining)}


class RipupObservation(TypedDict):
    board: np.ndarray
    remaining: list[NetPair]
    routed: list[RoutedNet]


class RoutingEnvRipup:
    """Routing env with both route and rip-up actions (Phase 3).

    The action space at each step is the concatenation of two lists:

        actions[0 .. R-1]         = "route remaining[i]"
        actions[R .. R+K-1]       = "rip up routed[j]"

    where R = len(remaining) and K = len(routed).

    Step semantics
    --------------
    * Route action: try A* on the chosen remaining net. On success, add to
      the routed list and lock its cells; reward = +1. On failure, the net is
      placed on a `failed_attempts` queue (tried but couldn't route on the
      current grid); reward = 0.
    * Rip-up action: free the chosen routed net's cells (rebuild grid from
      static obstacles plus the *other* routed nets), put its pair back on
      the remaining queue. Reward = −1. Any nets in `failed_attempts` are
      also returned to remaining, since the grid has changed and they may
      now be routable.

    Termination
    -----------
    Episode ends when either (a) remaining is empty AND no progress can be
    made, or (b) `max_steps` actions have been taken. Final episode return
    equals the number of nets currently routed.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        max_steps: Optional[int] = None,
        num_layers: int = 1,
    ) -> None:
        self.width = width
        self.height = height
        self.num_layers = num_layers
        self.max_steps = max_steps  # set at reset() if None

        self.static: Optional[PCBGrid] = None
        self.grid: Optional[PCBGrid] = None
        self.remaining: list[NetPair] = []
        self.routed: list[RoutedNet] = []
        self.failed_attempts: list[NetPair] = []
        self.step_count = 0
        self.ripup_count = 0

    def reset(self, grid: PCBGrid, netlist: list[NetPair]) -> RipupObservation:
        self.static = grid.clone()
        self.grid = grid.clone()
        self.remaining = list(netlist)
        self.routed = []
        self.failed_attempts = []
        self.step_count = 0
        self.ripup_count = 0

        if self.max_steps is None or self.max_steps <= 0:
            # Default budget: enough room for one full pass plus ~1.5x ripup-and-retry.
            self._effective_budget = max(8, len(netlist) * 3)
        else:
            self._effective_budget = self.max_steps

        return self._observation()

    def num_actions(self) -> int:
        return len(self.remaining) + len(self.routed)

    def step(self, action_idx: int) -> tuple[RipupObservation, float, bool]:
        if self.grid is None or self.static is None:
            raise RuntimeError("Call reset() before step().")
        if action_idx < 0 or action_idx >= self.num_actions():
            raise IndexError(f"action {action_idx} out of range "
                             f"(R={len(self.remaining)}, K={len(self.routed)})")

        self.step_count += 1
        R = len(self.remaining)

        if action_idx < R:
            reward = self._do_route(action_idx)
        else:
            reward = self._do_ripup(action_idx - R)

        # Termination conditions:
        #   * all nets settled (routed, no failures pending): all_settled
        #   * agent has no legal action this step: nothing_to_act_on
        #     (covers "remaining and routed both empty")
        #   * budget exhausted: budget_done
        # We do NOT terminate just because remaining is empty — the agent may
        # still want to rip up a routed net to retry a failed attempt.
        all_settled = not self.remaining and not self.failed_attempts
        nothing_to_act_on = self.num_actions() == 0
        budget_done = self.step_count >= self._effective_budget
        done = all_settled or nothing_to_act_on or budget_done

        return self._observation(), reward, done

    # --- action handlers -------------------------------------------------

    def _do_route(self, idx: int) -> float:
        assert self.grid is not None
        pair = self.remaining.pop(idx)
        path = _try_route_with_pin_clear(self.grid, pair)
        if path is None:
            self.failed_attempts.append(pair)
            return 0.0

        self.routed.append({"pair": pair, "path": path})
        _stamp_path_on_grid(self.grid, path)
        return 1.0

    def _do_ripup(self, idx: int) -> float:
        assert self.grid is not None and self.static is not None
        net = self.routed.pop(idx)

        # Rebuild grid: static obstacles + remaining routed paths.
        self.grid.layers[:] = self.static.layers
        self.grid.vias = set(self.static.vias)
        for n in self.routed:
            _stamp_path_on_grid(self.grid, n["path"])

        # Return the ripped pair to the remaining queue.
        self.remaining.append(net["pair"])
        # Previously-failed attempts may now succeed since the grid changed.
        self.remaining.extend(self.failed_attempts)
        self.failed_attempts = []

        self.ripup_count += 1
        return -1.0

    def _observation(self) -> RipupObservation:
        assert self.grid is not None
        if self.grid.num_layers == 1:
            board = self.grid.grid.copy()
        else:
            board = self.grid.layers.copy()
        return {
            "board": board,
            "remaining": list(self.remaining),
            "routed": [
                {"pair": n["pair"], "path": list(n["path"])} for n in self.routed
            ],
        }
