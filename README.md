# PCB Autorouter (Multi-Layer A\* + DRC + KiCad I/O + Learned Net Ordering)

[![CI](https://github.com/zafstathis-gif/PCB-Routing-Algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/zafstathis-gif/PCB-Routing-Algorithm/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch PCB autorouter in Python. Routes multi-layer netlists with vias, enforces trace-to-trace clearance, reads and writes real KiCad `.kicad_pcb` files, and includes a **PyTorch reinforcement-learning** track that learns net ordering directly from random boards plus a **learned A\* heuristic** that cuts search-node expansions.

Built as an ECE portfolio project to explore the algorithmic core of electronic design automation (EDA) — the same family of problems solved by tools like KiCad's `freerouting` or commercial autorouters — and to quantify the gap between heuristic search and a learned policy on small, well-defined tasks.

![demo](demo_output.png)

A 20×20 2-layer board routed with `clearance=1` and rip-up-and-reroute: black cells are static obstacles, light gray is the per-trace clearance halo, white-outlined circles are vias bridging layers. Animated version: [`routing_demo.gif`](routing_demo.gif).

## Features

- **Multi-layer A\* pathfinding** with the Manhattan heuristic — admissible across layers because layer-switch (via) cost is treated as 0 in the lower bound. Provably optimal at `clearance=0`; `prefer_directions=True` enables the classical even-horizontal / odd-vertical EDA layer bias.
- **Through-hole and SMD pads** via a dedicated `Pad(x, y, layers)` type. Multi-source A\* seeds every layer a pad occupies and accepts any goal layer.
- **Configurable trace-to-trace clearance (DRC)** via halo-on-lock dilation. A `static_mask` distinguishes pads/walls from trace halos, so shared pins still route under clearance without lifting real obstacles.
- **Sequential netlist routing** with shared-pin support, **five net-ordering heuristics** (`manhattan_asc`/`desc`, `bbox_area_asc`/`desc`, user-order), and `route_board_best_of` to pick the winning ordering automatically.
- **Rip-up-and-reroute (`route_board_rrr`)** — when a net fails, identify routed nets that block its ideal path, rip them up, retry, and re-route the displaced nets. Real EDA technique with per-net rip-up caps and an iteration limit; halo-aware so it works under non-zero clearance.
- **KiCad I/O** — `kicad_io.py` reads a `.kicad_pcb` (via `kiutils`), routes every net, and writes `(segment ...)` / `(via ...)` items back to disk. Multi-pad nets are connected via an MST of pad positions.
- **CLI** — `python cli.py board.kicad_pcb -o routed.kicad_pcb --clearance 0.2 --rrr` for one-shot routing.
- **Reinforcement-learning track** — three-phase PyTorch pipeline (REINFORCE → PPO+CNN → PPO+CNN+rip-up) for learned net ordering. Honest benchmarks against the deterministic baselines.
- **Visualization** — per-layer subplots, halo shading distinct from static obstacles, via markers bridging layers, faded crosses for unrouted nets.
- **Unit tests** — 86 tests (`unittest`, only numpy/torch as test-time deps).

## Project Structure

```
.
├── pcb_grid.py            # PCBGrid: 3D layer stack, static_mask, halo stamping
├── router.py              # Multi-layer A* + sequential routing + R&R + ordering heuristics
├── kicad_io.py            # Read/write .kicad_pcb (load_board, save_routed_board)
├── cli.py                 # `pcb-route input.kicad_pcb -o output.kicad_pcb`
├── app.py                 # Streamlit web demo (random boards + KiCad upload)
├── visualize.py           # Per-layer matplotlib renderer + animate_board GIF helper
├── bench.py               # Benchmark with --layers / --clearance flags
├── pyproject.toml         # Packaging: pip install -e .[rl,web,dev]
├── .github/workflows/
│   └── ci.yml             # Ruff + mypy + pytest on Python 3.9-3.12
├── examples/
│   ├── build_examples.py        # Generates the .kicad_pcb fixtures below
│   ├── blinker_unrouted.kicad_pcb
│   └── two_layer_demo.kicad_pcb
├── rl/
│   ├── env.py                 # Gym-style routing environment (multi-layer aware)
│   ├── policy.py              # PyTorch policy nets + per-net features
│   ├── train.py               # REINFORCE training loop (Phase 1)
│   ├── train_ppo.py           # PPO + CNN (Phase 2)
│   ├── train_ppo_ripup.py     # PPO + CNN + rip-up actions (Phase 3)
│   ├── evaluate.py            # Trained net-ordering policies vs heuristic baselines
│   ├── heuristic_net.py       # CNN cost-to-go function + admissibility clamps (Phase 4)
│   ├── heuristic_data.py      # Dijkstra-from-goal supervised data generator
│   ├── train_heuristic.py     # One-sided Huber loss training for HeuristicNet
│   ├── evaluate_heuristic.py  # Learned A* heuristic vs vanilla Manhattan A*
│   ├── policy.pt              # Trained weights for Phases 1-3
│   ├── policy_ppo.pt
│   ├── policy_ppo_ripup.pt
│   └── heuristic_net.pt       # Trained learned-heuristic weights
├── tests/
│   ├── test_router.py     # Multi-layer / clearance / R&R / custom-heuristic tests
│   ├── test_rl.py         # Env + policy-network shape tests
│   ├── test_heuristic.py  # Dijkstra + HeuristicNet + clamp tests
│   └── test_kicad_io.py   # Round-trip load -> route -> save tests
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Core install (numpy, matplotlib, kiutils):
pip install -e .

# Optional extras:
pip install -e ".[rl]"     # adds torch for the RL track
pip install -e ".[web]"    # adds streamlit for the interactive web demo
pip install -e ".[dev]"    # adds ruff + mypy + pyflakes
pip install -e ".[rl,web,dev]"   # everything (used by CI)

# Routing demos
python visualize.py                                       # matplotlib demo plot
python bench.py                                           # legacy single-layer benchmark
python bench.py --layers 2 --clearance 1                  # multi-layer + DRC benchmark

# CLI on a KiCad board (works after `pip install -e .`)
pcb-route examples/blinker_unrouted.kicad_pcb \
    -o /tmp/blinker_routed.kicad_pcb --rrr

# RL track (requires `[rl]` extra)
python -m rl.train                                        # Phase 1: REINFORCE          (~2 min CPU)
python -m rl.train_ppo                                    # Phase 2: PPO + CNN           (~10 min CPU)
python -m rl.train_ppo_ripup                              # Phase 3: PPO + rip-up        (~30-60 min CPU)
python -m rl.train_heuristic --boards 2000 --epochs 10    # Phase 4: learned A* heuristic (~5-10 min CPU)
python -m rl.evaluate                                     # net-ordering policies vs heuristics
python -m rl.evaluate_heuristic --boards 200              # learned heuristic vs Manhattan A*

# Web demo (requires `[web]` extra)
streamlit run app.py                                      # interactive UI for board / KiCad routing

# Test suite
python -m unittest discover tests                         # 86 tests
```

Programmatic use:

```python
from pcb_grid import PCBGrid
from router import route_board_best_of
from visualize import visualize_board

# Two-layer board with 1-cell clearance between traces.
grid = PCBGrid(20, 20, num_layers=2, clearance=1)
for x, y in [(5, 5), (5, 6), (10, 8), (10, 9)]:
    grid.add_obstacle(x, y, layer=0)

netlist = [
    ((1, 1, 0), (18, 18, 0)),
    ((1, 18, 0), (18, 1, 0)),
    ((6, 6, 0), (16, 6, 0)),
]

summary = route_board_best_of(grid, netlist, use_rrr=True)
print(f"Routed {len(summary['routed'])}/{len(netlist)} "
      f"using strategy: {summary['strategy']}")
for net in summary["routed"]:
    print(f"  {net['pair']}: {len(net['path'])} cells, {len(net['vias'])} vias")
visualize_board(grid, summary)
```

KiCad workflow (programmatic):

```python
from kicad_io import load_board, save_routed_board
from router import route_board_best_of

grid, netlist, ctx = load_board(
    "examples/blinker_unrouted.kicad_pcb",
    grid_mm=0.5,        # routing grid resolution
    clearance_mm=0.2,   # trace-to-trace clearance
)
summary = route_board_best_of(grid, netlist, use_rrr=True)
save_routed_board(ctx, summary, "blinker_routed.kicad_pcb")
```

## Algorithmic Notes

### Multi-layer A\* (single net)

`route_single_net(grid, start, end)`:

- Open set is a binary heap (`heapq`) keyed by `(f, counter, node)`. The counter is a monotonic tiebreaker so the heap never has to compare coordinates.
- Coordinates are `(x, y, layer)` 3-tuples internally. 2-tuple endpoints (`(x, y)`) on single-layer boards are auto-promoted to layer 0 for back-compat.
- Edges: planar neighbors cost `1.0` (or `1.0/1.2` with `prefer_directions=True`); layer-switches (vias) cost `VIA_COST = 10.0` (configurable per call).
- Manhattan distance `|dx| + |dy|` is admissible across layers because it underestimates layer-switch cost as `0`. With `prefer_directions=True` the lower bound is still `≥ 1·hops`, so the heuristic remains admissible.
- Multi-source / multi-goal: through-hole pads (`Pad(x, y, layers=(0, 1, ...))`) seed every layer they occupy with `g=0`, and any layer in the goal pad's `layers` tuple terminates the search.
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

R&R is exposed as a flag everywhere it makes sense: `route_board_best_of(..., use_rrr=True)`, an extra column in `bench.py`, and a separate section in the RL evaluation table. The R&R rebuild step is halo-aware, so it remains correct under non-zero clearance.

### Design rules (clearance)

Trace-to-trace clearance is enforced via **halo-on-lock dilation**: each routed path stamps not just its own cells but a Chebyshev `clearance`-radius buffer around them. The buffer is reserved by `PCBGrid.stamp_path`, so the next net's A\* sees those cells as blocked.

`PCBGrid` also keeps a `static_mask` boolean array that records which cells were placed by `add_obstacle` / `add_via` (pads, walls, board outline) as opposed to halos stamped by traces. `_try_route_with_pin_clear` consults this mask: when temporarily clearing the halo around a shared pin so a new net can route out of it, **static obstacles are never lifted** — only trace halos. This keeps walls / adjacent pads safely blocking A\* even mid-search.

### KiCad I/O

`kicad_io.load_board` parses a `.kicad_pcb` file (via `kiutils`), discovers copper layers in stack-up order (`F.Cu`, `In1.Cu`, …, `B.Cu`), maps pads onto a `PCBGrid` (SMD on one layer, through-hole as a via on all layers), stamps any pre-existing traces as static obstacles, and emits a netlist via a per-net minimum spanning tree (Kruskal over pad positions). Existing nets and net numbers are preserved.

`kicad_io.save_routed_board` collapses each routed path into the minimum number of straight `(segment ...)` items per layer and emits a `(via ...)` at every layer transition. The output is a valid KiCad PCB file you can open directly in KiCad and run its own DRC against.

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

### Multi-layer + DRC

Same netlist, with two copper layers and 1-cell clearance (`python bench.py --layers 2 --clearance 1`):

```
strategy           greedy            + R&R
--------------------------------------------------
user-order         8/8  (197 cells)  8/8  (197 cells)
manhattan_asc      6/8  (127 cells)  7/8  (166 cells)
manhattan_desc     6/8  (152 cells)  7/8  (167 cells)
bbox_area_asc      7/8  (154 cells)  8/8  (195 cells)
bbox_area_desc     6/8  (152 cells)  7/8  (167 cells)
```

Three takeaways:

- **Multi-layer relieves congestion** — the second copper layer is enough to push most strategies from 5/8 to 7-8/8 even under clearance.
- **Clearance hurts congestion-sensitive orderings most** — `manhattan_asc` and `bbox_area_desc` drop more than `user-order` because their tightly-packed early routes generate halos that block later nets.
- **R&R + best-of still wins** — combining `bbox_area_asc + R&R` reaches 8/8 (100%) on this board.

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
| Phase 3 — PPO + CNN + rip-up actions  | 6.245       | 1.102  |

**Reading the table.**

- **Net ordering effect.** Worst greedy ordering (`manhattan_desc`, 5.72) → best (`manhattan_asc`, 6.77) is a swing of more than one routed net per board — 12.5% of the netlist — purely from ordering.
- **R&R effect.** R&R lifts every strategy. Largest gains are on the *weak* orderings (`manhattan_desc` +0.85, `bbox_area_desc` +0.75) — exactly the cases where the initial greedy pass made the most damaging early decisions. R&R also tightens the variance everywhere.
- **The current bar.** `best_of + R&R` is the strongest deterministic baseline at **7.22/8 = 90.3%** routed.
- **REINFORCE vs PPO+CNN (Phase 1 vs Phase 2).** Both learned policies essentially tie the strongest *no-R&R* heuristic (~6.77). The Phase 2 upgrade (CNN + actor-critic + clipped surrogate) did **not** unlock new gains over REINFORCE, because at this scale (8 nets, 20×20 board) the optimal one-shot ordering is well-approximated by "shortest first" — a rule the simpler model already finds. This is an honest negative result: better algorithms only buy you more on harder problems.
- **Phase 3 motivation and outcome.** Beating R&R-enabled baselines requires per-board iterative repair the policy can carry out itself. The Phase 3 action space adds "rip up routed net `j`" alongside "route remaining net `i`", with reward `+1` for a successful route, `0` for a failed route, `−1` for a rip-up — making the cumulative episode reward equal to the final routed count. Training converges (rolling-mean episode return climbs from 4.4 to ~6.4 over 300 iterations and plateaus) and the agent uses its rip-up action ~3.9 times per episode at evaluation.
- **Phase 3 result: an honest negative.** On the held-out set the Phase 3 policy reaches **6.245 ± 1.102** — *worse* than Phase 1 / Phase 2 (~6.75) and behind even the weakest R&R-enabled heuristic (`manhattan_desc + R&R`, 6.57). The expanded action space (`route` ∪ `ripup`, about 3× the per-step branching of Phase 1/2) makes credit assignment harder, and the `−1` rip-up reward shapes for *fewer* rip-ups rather than *better-targeted* ones, so the agent learns to mostly route, occasionally rip up, and ends up worse than the simpler `route-only` policies. Beating the `best_of + R&R` oracle (7.22) at this scale almost certainly needs (a) hierarchical actions ("pick which net to fix" → "pick what to do with it"), (b) reward shaping that distinguishes useful rip-ups from wasted ones, or (c) much more training data on bigger boards where R&R has more room to matter — the next direction this project will explore.

## Learned A* Heuristic (Phase 4)

The other RL track: replace A\*'s Manhattan heuristic with a small CNN that predicts cost-to-go from a local crop of the obstacle map, and quantify the search-efficiency win.

**Network** (`rl/heuristic_net.py`). 11×11 obstacle crop centered on the current cell, stacked across copper layers (zero-padded to `max_layers=4` so one model fits boards with 1-4 copper layers), plus three scalar features `[dx/W, dy/H, crop_density]`. Three Conv2d layers → one MaxPool → two FC → softplus output. ~70k parameters.

**Training data** (`rl/heuristic_data.py`). Per board, pick a random goal and run **one** Dijkstra pass *from the goal* — that single O(H·W·log(H·W)) pass gives exact cost-to-go for every reachable cell. 2,000 random boards × ~200 samples each → 400k supervised `(crop, scalars, true_cost)` triples.

**Training loss** (`rl/train_heuristic.py`). **One-sided Huber**: positive residuals (`pred > target`) penalised 5× more than negative residuals. This biases the network to *under*-predict — admissible by construction iff overshoots are rare.

**Three integration modes** in `make_learned_heuristic(model, grid, clamp=…)`:

| Clamp | `h_used(n) = ` | Property |
|---|---|---|
| `"min"` | `min(learned, manhattan)` | **Strictly admissible** — falls back to manhattan if the net overshoots. Optimal paths guaranteed. |
| `"max"` | `max(learned, manhattan)` | **Tighter** when the net predicts above manhattan; admissible iff the net doesn't overshoot. The setup that *can* reduce A\* expansions. |
| `"raw"` | `learned` | Weighted-A\* style. No admissibility guarantee. |

**Results on 200 held-out boards** (`python -m rl.evaluate_heuristic --boards 200`), measuring nodes A\* pops from the open set:

```
method                      nodes     vs M   wall (s)   path len  len ratio
------------------------------------------------------------------------------
Manhattan                    53.3    1.000    0.00011      14.29     1.0000
Learned (max)                53.0    0.995    0.01922      14.29     1.0000
Learned (min)                48.5    0.911    0.01741      14.29     1.0000
Learned (raw)                53.1    0.997    0.01946      14.29     1.0000
```

**The reduction comes from better A\* heap tie-breaking, not from a tighter heuristic.** Manhattan ties many cells with the same `f = g + h` along the optimal-path frontier, and A\* must explore them all in counter-order. The learned heuristic breaks those ties — even though its value is *below* Manhattan on average (so `min` selects it), it provides a small per-cell ordering signal that points A\* toward cells closer to the goal first. The path-length ratio stays exactly `1.0000` everywhere because `min(learned, manhattan)` is admissible by construction.

**Out-of-distribution behaviour.** The network was trained on 18-obstacle boards. Re-evaluating on denser boards:

| Board difficulty           | Manhattan nodes | Learned (min) nodes | Learned (max) overshoot rate |
|----------------------------|----------------:|--------------------:|-----------------------------:|
| 18 obstacles (in-distribution) | 53.3        | **48.5 (-9%)**       | 0.0 %                        |
| 60 obstacles               | 47.6            | **46.0 (-3%)**       | 2.8 %                        |
| 100 obstacles              | 69.5            | 70.7 (+2%)           | 4.2 %                        |

The improvement degrades gracefully as obstacle density moves outside the training distribution, but never causes incorrect paths thanks to the admissibility clamp — even at 100 obstacles, `Learned (min)` still produces optimal paths.

**Wall-clock caveat.** The learned heuristic is ~170× slower per route because the CNN runs once per A\* node expansion. **Batching neighbour evaluations** (one forward pass per popped node instead of per neighbour, and a single tensor build instead of per-cell `extract_crop`) would shrink that to roughly 4–6× and is the obvious next optimization.

## Limitations and Future Work

**Recently landed** (was on this list, now in the codebase):

- ✅ Multi-layer routing with vias and configurable `VIA_COST`.
- ✅ Trace-to-trace clearance via halo-on-lock dilation, with a `static_mask` that protects walls/pads during pin-clear.
- ✅ KiCad `.kicad_pcb` read/write — `kicad_io.py` + the `pcb-route` CLI entry point make the router usable on real boards.
- ✅ Phase 3 PPO + rip-up training — converges to 6.25/8 on the held-out test set; honestly *worse* than Phase 1/2, see the discussion above for why and what would actually help.
- ✅ Learned A\* heuristic — CNN cost-to-go function trained on Dijkstra ground truth; 9% reduction in A\* node expansions on in-distribution boards with 0% path-length cost; admissibility preserved via the `min(learned, manhattan)` clamp.
- ✅ Packaging — `pyproject.toml` with optional `[rl]`, `[web]`, `[dev]` extras and the `pcb-route` CLI entry point.
- ✅ GitHub Actions CI — `ruff` + `pyflakes` + `mypy` + `unittest` on Python 3.9-3.12 for every push and PR.
- ✅ Streamlit web demo (`app.py`) — random-board mode and KiCad-upload mode with the routed file as a download.
- ✅ Animated routing visualization (`animate_board` → GIF) — used in the README header.

**Still on the list**, in order of expected impact:

- **Batched neighbour evaluation** in the learned-heuristic A\* loop to bring wall-clock per route from ~170× to ~5× vs vanilla Manhattan.
- **Better Phase 3 RL** — hierarchical action space (choose-net → choose-action) and reward shaping that distinguishes productive rip-ups, on bigger boards where R&R has real room to help.
- **Trace width > 1 cell.** The halo machinery is already kernel-based, so trace width is one structuring-element parameter away.
- **Length matching** for high-speed differential pairs.
- **Hosted Streamlit Cloud deployment** of `app.py` so the demo runs without a clone.

## Tech Stack

Python 3.9+, NumPy, matplotlib, PyTorch, [kiutils](https://github.com/mvnmgrx/kiutils) (KiCad I/O).

## Author

**Eustathios Zafeiropoulos**
ECE student, Aristotle University of Thessaloniki, Greece

## License

MIT — see [LICENSE](LICENSE).
