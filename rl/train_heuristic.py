"""Train the learned A* heuristic on Dijkstra-derived ground truth.

Run from the project root:

    python -m rl.train_heuristic                    # generates data + trains
    python -m rl.train_heuristic --epochs 10         # longer training
    python -m rl.train_heuristic --data rl/heuristic_data.npz   # use a cached dataset

Outputs trained weights to `rl/heuristic_net.pt` and an 80/20 train/val
split with progress prints per epoch.

**One-sided Huber loss**: overestimates are penalised `overshoot_weight`x
more than underestimates, biasing the network's predictions below the
true cost-to-go so the `min(learned, manhattan)` clamp in
`make_learned_heuristic` actually uses the learned value (rather than
falling back to manhattan because the learned value overshot).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rl.heuristic_data import build_dataset                      # noqa: E402
from rl.heuristic_net import HeuristicNet                         # noqa: E402


DEFAULT_WEIGHTS = os.path.join(_HERE, "heuristic_net.pt")


def one_sided_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
    overshoot_weight: float = 5.0,
) -> torch.Tensor:
    """Huber loss where positive residuals (pred > target) are penalised
    `overshoot_weight`x more than negative residuals.

    Pushes the network to predict at-or-below the true cost-to-go so the
    learned heuristic actually contributes (rather than getting clamped
    back to manhattan everywhere).
    """
    err = pred - target
    over = torch.where(err > 0, err, torch.zeros_like(err))
    under = torch.where(err < 0, -err, torch.zeros_like(err))

    huber_over = torch.where(
        over < beta, 0.5 * over.pow(2) / beta, over - 0.5 * beta,
    )
    huber_under = torch.where(
        under < beta, 0.5 * under.pow(2) / beta, under - 0.5 * beta,
    )
    return (overshoot_weight * huber_over + huber_under).mean()


def evaluate(model: HeuristicNet, loader: DataLoader) -> tuple[float, float, float]:
    """Return (mean loss, mean |error|, fraction overshooting) on `loader`."""
    model.eval()
    total_loss = 0.0
    total_abs_err = 0.0
    total_overshoots = 0
    n = 0
    with torch.no_grad():
        for crop, scalars, target in loader:
            pred = model(crop, scalars)
            total_loss += float(one_sided_huber_loss(pred, target).item()) * len(target)
            err = pred - target
            total_abs_err += float(err.abs().sum().item())
            total_overshoots += int((err > 0).sum().item())
            n += len(target)
    return total_loss / n, total_abs_err / n, total_overshoots / n


def train(
    crops: np.ndarray,
    scalars: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 3e-4,
    overshoot_weight: float = 5.0,
    weights_out: str = DEFAULT_WEIGHTS,
    seed: int = 0,
) -> HeuristicNet:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(targets)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    def make_loader(rows: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(crops[rows]),
            torch.from_numpy(scalars[rows]),
            torch.from_numpy(targets[rows]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)

    model = HeuristicNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {len(train_idx):,} examples, validating on {len(val_idx):,}")
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print(f"{'epoch':>5}  {'train_loss':>10}  {'val_loss':>10}  "
          f"{'val |err|':>10}  {'val overshoot %':>16}")
    print("-" * 64)

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        running_n = 0
        for crop, scal, tgt in train_loader:
            pred = model(crop, scal)
            loss = one_sided_huber_loss(
                pred, tgt, overshoot_weight=overshoot_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(tgt)
            running_n += len(tgt)

        train_loss = running_loss / running_n
        val_loss, val_abs_err, val_overshoot = evaluate(model, val_loader)
        print(f"{ep + 1:>5}  {train_loss:>10.3f}  {val_loss:>10.3f}  "
              f"{val_abs_err:>10.3f}  {100*val_overshoot:>14.2f}%")

    torch.save(model.state_dict(), weights_out)
    print(f"\nSaved trained heuristic to {weights_out}")
    return model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=None,
        help="Optional .npz cache from rl.heuristic_data; otherwise "
             "regenerate on the fly.",
    )
    parser.add_argument("--boards", type=int, default=2000)
    parser.add_argument("--max-per-board", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--overshoot-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_WEIGHTS)
    args = parser.parse_args(argv)

    if args.data and os.path.exists(args.data):
        print(f"Loading cached dataset from {args.data}")
        z = np.load(args.data)
        crops, scalars, targets = z["crops"], z["scalars"], z["targets"]
    else:
        print(f"Generating dataset: boards={args.boards}, seed={args.seed}")
        crops, scalars, targets = build_dataset(
            n_boards=args.boards, seed=args.seed,
            max_examples_per_board=args.max_per_board,
        )
    print(f"Dataset shapes: crops={crops.shape}, targets={targets.shape}")
    print(f"  target stats: min={targets.min():.2f}, "
          f"mean={targets.mean():.2f}, max={targets.max():.2f}")
    print()

    train(
        crops, scalars, targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        overshoot_weight=args.overshoot_weight, weights_out=args.out,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
