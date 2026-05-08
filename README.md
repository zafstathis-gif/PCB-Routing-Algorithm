# PCB Autorouter (A\* + Heuristics + Rip-up-and-Reroute + Learned Net Ordering)

A grid-based PCB autorouter implemented from scratch in Python. Connects pin pairs from a netlist using A\* pathfinding on a 2D obstacle grid, supports shared pins, applies **iterative rip-up-and-reroute** to escape greedy dead-ends, selects the best of several net-ordering heuristics, and includes a **PyTorch reinforcement-learning policy** that learns net ordering directly from random boards via REINFORCE.

Built as an ECE portfolio project to explore the algorithmic core of electronic design automation (EDA) — the same family of problems solved by tools like KiCad's `freerouting` or commercial autorouters, in their simplest 2D form — and to quantify the gap between heuristic search and a learned policy on a small, well-defined task.

![demo](demo_output.png)

## Features

- **A\* pathfinding** with the Manhattan-distance heuristic — provably optimal on 4-connected unit-cost grids.
- **Sequential netlist routing** with shared-pin support: a single pin can serve as the endpoint for multiple nets, as in power and ground rails.
- **Net-ordering heuristics** — `manhattan_asc`/`desc`, `bbox_area_asc`/`desc`, plus user-order. Net order is the dominant lever for a sequential router; the included benchmark quantifies the effect.
- **`route_board_best_of`** runs every heuristic and keeps the result that routes the most nets (tie-broken by total trace length). Establishes a strong deterministic baseline.
- **Rip-up-and-reroute (`route_board_rrr`)** — when a net fails, identify routed nets that block its ideal path, rip them up, retry, and re-route the displaced nets. A real EDA technique, here implemented with per-net rip-up caps and an iteration limit to prevent oscillation. Improves every ordering heuristic on the held-out test set.
- **Reinforcement-learning policy** for net ordering. A small PyTorch network is trained with REINFORCE (moving-average baseline + entropy bonus) to pick which net to route next given the current board. Matches the strongest single heuristic and beats the user-order baseline by 6.5% on a 200-board test set.
- **Visualization** — heatmap grid, distinct trace colors per net, circle/star markers for start/end pins, faded crosses + dashed line for nets the router could not connect.
- **Unit tests** — 20 tests (`unittest`, no third-party dependencies).

## Project Structure

```
.
├── pcb_grid.py         # PCBGrid: 2D obstacle grid environment
├── router.py           # A* + sequential routing + ordering heuristics
├── visualize.py        # Matplotlib renderer
├── bench.py            # Compare ordering strategies on a fixed netlist
├── rl/
│   ├── env.py          # Gym-style routing environment
│   ├── policy.py       # PyTorch policy network + per-net features
│   ├── train.py        # REINFORCE training loop
│   ├── evaluate.py     # Trained policy vs heuristic baselines
│   └── policy.pt       # Trained weights (produced by train.py)
├── tests/
│   └── test_router.py  # unittest suite (20 tests)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

python visualize.py                 # routing demo + matplotlib plot
python bench.py                     # benchmark ordering heuristics
python -m rl.train                  # Phase 1: REINFORCE + flat MLP  (~2 min on CPU)
python -m rl.train_ppo              # Phase 2: PPO + CNN              (~10 min on CPU)
python -m rl.train_ppo_ripup        # Phase 3: PPO + CNN + rip-up     (~30-60 min on CPU)
python -m rl.evaluate               # compare all trained policies vs heuristics
python -m unittest discover tests   # run the test suite (35 tests)
```

Programmatic use:

```python
from pcb_grid import PCBGrid
from router import route_board_best_of
from visualize import visualize_board

grid = PCBGrid(20, 20)
for x, y in [(5, 5), (5, 6), (10, 8), (10, 9)]:
    grid.add_obstacle(x, y)

netlist = [
    ((1, 1), (18, 18)),
    ((1, 18), (18, 1)),
    ((6, 6), (16, 6)),
]

summary = route_board_best_of(grid, netlist)
print(f"Routed {len(summary['routed'])}/{len(netlist)} "
      f"using strategy: {summary['strategy']}")
visualize_board(grid, summary)
```

## Algorithmic Notes

### A\* (single net)

`route_single_net(grid, start, end)`:

- Open set is a binary heap (`heapq`) keyed by `(f, counter, node)`. The counter is a monotonic tiebreaker so the heap never has to compare grid coordinates.
- `g[n]` = best-known cost from start to n. `h(n)` = Manhattan distance from n to end. `f(n) = g + h`.
- Manhattan distance is *admissible* and *consistent* on a 4-connected unit-cost grid, so A\* is guaranteed to return the shortest path.
- No decrease-key (not supported by `heapq`): when a cheaper `g` is found we re-push the node, and stale pops are filtered by the `tentative_g < g_score[...]` guard.

### Sequential routing (multiple nets)

`route_board(grid, netlist, sort_strategy=None)` calls A\* for each pair, then locks the resulting path by marking all of its cells as obstacles so subsequent nets cannot cross. A failed net is reported as unrouted; later nets still attempt to route around the locked traces.

**Pin sharing.** A pin may be the endpoint of multiple nets (a power rail with several sinks, or a fanout). To support this, each net's start and end cells are temporarily un-marked before its A\* call and re-locked as part of the routed path afterward — so a previously-routed trace ending at a pin does not block a new net starting from the same pin.

### Net ordering

The order in which nets are presented to a sequential router is the dominant driver of success rate. Supported strategies:

| Strategy            | Meaning                                |
|---------------------|----------------------------------------|
| `None`              | Preserve the user's order.             |
| `manhattan_asc`     | Shortest pin-to-pin distance first.    |
| `manhattan_desc`    | Longest first.                         |
| `bbox_area_asc`     | Smallest bounding box first.           |
| `bbox_area_desc`    | Largest bounding box first.            |

`route_board_best_of` runs all strategies and keeps the result with the most routed nets, tie-broken by total trace length.

### Rip-up-and-reroute

`route_board_rrr` extends sequential routing with iterative repair:

1. Run an initial sequential pass with the chosen `sort_strategy`.
2. For each net that failed, compute its **ideal path** on a *static-only* grid (i.e. as if no traces existed). Any already-routed net whose cells intersect that ideal path is a *blocker* — a candidate for rip-up.
3. Tentatively rip up all rippable blockers, route the failed net on the cleared grid, then attempt to re-route the ripped nets in shortest-Manhattan order. **Accept** the swap iff the total number of routed nets strictly increased; otherwise **revert**.
4. Repeat until no progress is made or `max_iterations` is reached.

Oscillation is bounded by a per-net rip-up cap (`max_ripups_per_net`) and the outer iteration limit. The "ideal path" heuristic is a tighter blocker filter than bounding-box overlap: it identifies the routed nets that an unblocked router would actually run into, not just nets that are merely nearby.

R&R is exposed as a flag everywhere it makes sense: `route_board_best_of(..., use_rrr=True)`, an extra column in `bench.py`, and a separate section in the RL evaluation table.

## Benchmark — Heuristics & R&R

Sample output from `python bench.py` on a 20×20 board with 18 obstacles and 8 nets:

```
strategy           greedy            + R&R
--------------------------------------------------
user-order         4/8  (134 cells)  6/8  (162 cells)
manhattan_asc      5/8  (143 cells)  6/8  (146 cells)
manhattan_desc     5/8  (153 cells)  5/8  (153 cells)
bbox_area_asc      5/8  (120 cells)  6/8  (141 cells)
bbox_area_desc     5/8  (153 cells)  7/8  (185 cells)
```

Two effects are visible:

- **Net ordering matters** — every greedy heuristic beats user-order; the spread between best and worst is 2 nets out of 8 (25% of the netlist).
- **R&R recovers from greedy mistakes** — every strategy improves or ties under R&R, and the `bbox_area_desc + R&R` combination reaches 7/8 (87.5%), strictly better than any greedy result.

## Learned Net Ordering (Reinforcement Learning)

**Motivation.** The benchmark above shows that net order is the dominant lever for a sequential router. `route_board_best_of` exploits this by trying all five heuristic orderings and keeping the winner — but it requires running the router five times per board. A learned policy can in principle pick a good ordering in one shot, conditioned on the actual board geometry.

**Three-stage progression.** The project layers learned components in three phases; each can be trained, evaluated, and compared independently.

| Phase | Algorithm | Encoder | Action space | Script |
|---|---|---|---|---|
| 1 | REINFORCE + moving-average baseline | flat MLP | route remaining net `i` | `rl/train.py` |
| 2 | PPO (clipped surrogate, GAE, value head) | CNN (3 conv + 2 maxpool) | route remaining net `i` | `rl/train_ppo.py` |
| 3 | PPO + CNN | CNN | route net `i` **or** rip up routed net `j` | `rl/train_ppo_ripup.py` |

**Common setup.**
- **Environment** (`rl/env.py`) — Gym-style episodic MDP. Two variants: `RoutingEnv` (Phase 1/2) supports only "route" actions; `RoutingEnvRipup` (Phase 3) accepts both "route remaining net `i`" and "rip up routed net `j`", with reward `+1` for a successful route, `0` for a failed route, and `−1` for a rip-up. Episode return therefore equals the final routed-net count.
- **Policies** (`rl/policy.py`) — `PolicyNet` (Phase 1, flat-MLP), `CNNActorCritic` (Phase 2), `CNNActorCriticRipup` (Phase 3, augmented action features with a route/rip-up indicator).
- **Per-action features** (6-dim, normalized to `[0, 1]`) — start (x, y), end (x, y), Manhattan distance, bounding-box area. Phase 3 adds a 7th feature: the route/rip-up indicator.
- **Evaluation** (`rl/evaluate.py`) — 200 held-out random boards with an RNG seed disjoint from training. All policies act greedily (argmax) at evaluation.

**Results (mean nets routed out of 8, 200 held-out test boards):**

| Method                                | Mean routed | Std    |
|---------------------------------------|------------:|-------:|
| **Greedy (no R&R)**                   |             |        |
| user-order                            | 6.350       | 1.272  |
| manhattan_desc                        | 5.720       | 1.364  |
| bbox_area_desc                        | 5.850       | 1.307  |
| bbox_area_asc                         | 6.695       | 1.006  |
| manhattan_asc                         | 6.770       | 0.983  |
| best_of                               | 7.055       | 0.832  |
| **With rip-up-and-reroute**           |             |        |
| user-order + R&R                      | 6.890       | 0.932  |
| manhattan_desc + R&R                  | 6.565       | 1.125  |
| bbox_area_desc + R&R                  | 6.595       | 1.096  |
| bbox_area_asc + R&R                   | 6.970       | 0.854  |
| manhattan_asc + R&R                   | 7.000       | 0.831  |
| **best_of + R&R (oracle)**            | **7.220**   | 0.769  |
| **Learned policies**                  |             |        |
| Phase 1 — REINFORCE + flat MLP        | 6.760       | 1.001  |
| Phase 2 — PPO + CNN                   | 6.725       | 0.985  |
| Phase 3 — PPO + CNN + rip-up actions  | *training pipeline complete; weights not yet trained — run `python -m rl.train_ppo_ripup`* | |

**Reading the table.**

- **Net ordering effect.** Worst greedy ordering (`manhattan_desc`, 5.72) → best (`manhattan_asc`, 6.77) is a swing of more than one routed net per board — 12.5% of the netlist — purely from ordering.
- **R&R effect.** R&R lifts every strategy. Largest gains are on the *weak* orderings (`manhattan_desc` +0.85, `bbox_area_desc` +0.75) — exactly the cases where the initial greedy pass made the most damaging early decisions. R&R also tightens the variance everywhere.
- **The current bar.** `best_of + R&R` is the strongest deterministic baseline at **7.22/8 = 90.3%** routed.
- **REINFORCE vs PPO+CNN (Phase 1 vs Phase 2).** Both learned policies essentially tie the strongest *no-R&R* heuristic (~6.77). The Phase 2 upgrade (CNN + actor-critic + clipped surrogate) did **not** unlock new gains over REINFORCE, because at this scale (8 nets, 20×20 board) the optimal one-shot ordering is well-approximated by "shortest first" — a rule the simpler model already finds. This is an honest negative result: better algorithms only buy you more on harder problems.
- **Phase 3 motivation.** Beating R&R-enabled baselines requires per-board iterative repair the policy can carry out itself. The Phase 3 action space adds "rip up routed net `j`" alongside "route remaining net `i`", with reward `+1` for a successful route, `0` for a failed route, `−1` for a rip-up — making the cumulative episode reward equal to the final routed count and giving the policy something genuinely non-trivial to learn. The training script (`rl/train_ppo_ripup.py`), env (`RoutingEnvRipup`), and policy (`CNNActorCriticRipup`) are implemented and unit-tested; weights for the held-out comparison have not yet been trained in this session.

## Limitations and Future Work

This is a 2D grid-based baseline. Concrete next steps, in order of expected impact:

- **Beat `best_of + R&R` with the RL policy.** The current REINFORCE policy ties the best single greedy heuristic but is behind R&R-enabled baselines. PPO with a value-function critic (lower-variance gradients), a CNN board encoder, and an extended action space (route-next-net **or** rip-up-net-and-retry) is the planned Phase 2/3 path to per-board iterative-repair learned from data.
- **Learn the A\* heuristic itself.** Replace Manhattan with a small CNN that predicts cost-to-go from the current cell + obstacle pattern; compare path lengths and search-node counts to vanilla A\*.
- **Multi-layer support** with vias.
- **Trace width, clearance, and design-rule checks (DRC).**
- **Length matching** for high-speed differential pairs.

## Tech Stack

Python 3.9+, NumPy, matplotlib, PyTorch.

## Author

**Eustathios Zafeiropoulos**
ECE student, Aristotle University of Thessaloniki, Greece

## License

MIT — see [LICENSE](LICENSE).
