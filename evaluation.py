#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Tuple
import logging
from pathlib import Path
import json
import os
import torch
from torch import nn
from models.NN_models import MLP
from models.odeint_classes import NODEFunc, StochasticNODE
from src.helpers import get_model, get_flow_matching_inputs
from data_preprocessing.dataloader import ProcessedDataset, get_dataloaders

from src.validation import validate_metrics

# Ensure the synthetic-data modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# from dataloader import get_dataloaders  # noqa: E402
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple flow-matching training script")
    parser.add_argument("--data_config", default="lotka_volterra", help="Name of data config JSON")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training loader")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--model_config", default="MLP", help="Name of model config JSON")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--lr_scheduler", choices=["none", "cosine"], default="none", help="Optional LR scheduler")
    parser.add_argument("--lr_t_max", type=int, default=0, help="T_max for CosineAnnealingLR (defaults to epochs if 0)")
    parser.add_argument("--lr_eta_min", type=float, default=0.0, help="Eta_min for CosineAnnealingLR")
    parser.add_argument("--interpolant_kind", default="linear", help="Kind of interpolant to use")
    parser.add_argument("--degree", type=int, default=3, help="Degree of the B-spline interpolant")
    parser.add_argument("--subsample_per_interval", type=int, default=1, help="Number of points to subsample per interval")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every n epochs")
    parser.add_argument("--sigma", type=float, default=0.08, help="The noise added to the conditional path")
    parser.add_argument("--data_sample_complexity", type=float, default=1, help="Data complexity")
    parser.add_argument("--exp_name", type=str, default="base", help="Experiment name")
    parser.add_argument("--dynamics_kind", type=str, default="ode", choices=["ode", "sde_constant_sigma", "sde_quadratic_sigma", "sde_time_varying_sigma"], help="whether we have ode/sde dynamics")
    return parser.parse_args()



def find_run_dirs(base: Path):
    """
    base like: .../results/SDE_synthetic/{exp_name}_{data_config}_{interpolant_kind}
    returns: [base, base_0, base_1, ...] that actually exist as directories
    """
    parent = base.parent
    prefix = base.name
    run_dirs = []
    for p in parent.iterdir():
        if p.is_dir() and (p.name == prefix or p.name.startswith(prefix + "_")):
            run_dirs.append(p)
    return sorted(run_dirs)

def main() -> None:
    args = parse_args()

    root = Path("/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/results/SDE_synthetic")
    base = root / f"{args.exp_name}_{args.data_config}_{args.interpolant_kind}"

    run_dirs = find_run_dirs(base)
    if len(run_dirs) == 0:
        raise FileNotFoundError(f"No run dirs found for prefix: {base}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build loaders ONCE (deterministic split via seed in data_config)
    train_loader_unprocessed, test_loader_unprocessed = get_dataloaders(
        args.data_config, batch_size=args.batch_size, data_sample_complexity=args.data_sample_complexity
    )
    sample_batch = next(iter(train_loader_unprocessed))
    data_dim = sample_batch["values"].shape[-1]
    input_dim = data_dim + 1

    # recompute stats + sigma like training did
    train_loader = ProcessedDataset(
        train_loader_unprocessed,
        train=True,
        interpolant_kind=args.interpolant_kind,
        degree=args.degree,
        subsample_per_interval=args.subsample_per_interval,
        device=device,
        dynamics_kind=args.dynamics_kind,
        sigma=args.sigma,
    )
    test_loader = ProcessedDataset(
        test_loader_unprocessed,
        interpolant_kind=args.interpolant_kind,
        degree=args.degree,
        subsample_per_interval=args.subsample_per_interval,
        device=device,
        stats=train_loader.stats,
        dynamics_kind=args.dynamics_kind,
        sigma=train_loader.sigma,
    )

    # init model(s) ONCE
    model = get_model(args.model_config, data_dim, input_dim, device)
    score_model = None if args.dynamics_kind == "ode" else get_model(
        args.model_config, data_dim, input_dim, device, dynamics_kind=args.dynamics_kind
    )

    for save_dir in run_dirs:
        print(f"Evaluating {save_dir}")
        model_path = save_dir / "model.pt"
        score_path = save_dir / "score_model.pt"

        if not model_path.exists():
            print(f"[skip] missing {model_path}")
            continue
        if args.dynamics_kind != "ode" and not score_path.exists():
            print(f"[skip] missing {score_path}")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        if score_model is not None:
            score_model.load_state_dict(torch.load(score_path, map_location=device))

        metrics = validate_metrics(
            model,
            score_model,
            test_loader,
            dynamics_kind=args.dynamics_kind,
            stats=train_loader.stats,
            sigma=train_loader.sigma,
            save_dir=save_dir,
            subsample_per_interval=args.subsample_per_interval,
            metrics_to_run=["mse"],
            trajectories_png=False,
            num_traj_to_draw=10,
        )

        with open(save_dir / "test_validation_metrics_from_evaluation.json", "w") as f:
            json.dump(metrics, f)

        print(f"done writing {save_dir / 'test_validation_metrics_from_evaluation.json'}")

if __name__ == "__main__":
    main()