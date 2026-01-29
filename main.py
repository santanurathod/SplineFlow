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
import numpy as np
from src.validation import validate_metrics
import time
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


def prepare_flow_matching_batch(batch, interpolant_kind="linear", degree=3, subsample_per_interval=8, device=None):
    """Build (input, target, mask) tuples for flow matching from a dataloader batch."""

    values = batch["values"].to(device).float()          # [B, W, D]
    times = batch["times"].to(device).float()   # [B, W]
    mask = batch["mask"].to(device)               # [B, W, D]

    inputs, targets, mask = get_flow_matching_inputs(values, times, mask, interpolant_kind, degree, subsample_per_interval=subsample_per_interval)
    return inputs, targets, mask



def mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (pred - target) ** 2
    # weighted = squared * mask
    weighted = squared
    denom = mask.sum()
    if denom <= 0:
        return torch.tensor(0.0, device=pred.device)
    return weighted.sum() / denom


def train_epoch(model: nn.Module, score_model: nn.Module, loader, optimizer: torch.optim.Optimizer, interpolant_kind="linear", degree=3, subsample_per_interval=8, sigma=0.01, dynamics_kind='ode', device= 'cpu'):
    model.train()

    if score_model!=None:
        score_model.train()

    total_loss, batches, score_loss = 0.0, 0, 0.0
    for batch in loader:
        
        # inputs, times, targets, mask, score, lambda_t, eps_xt, eps_score= batch['inputs'].to(device), batch['times'].to(device), batch['targets'].to(device), batch['mask'].to(device), batch['score'].to(device), batch['lambda_t'].to(device), batch['eps_xt'].to(device), batch['eps_score'].to(device)
        inputs, times, targets, mask, sigma_t, lambda_t, der_sigma_t= batch['inputs'].to(device), batch['times'].to(device), batch['targets'].to(device), batch['mask'].to(device), batch['sigma_t'].to(device), batch['lambda_t'].to(device), batch['der_sigma_t'].to(device)
        if inputs.numel() == 0:
            continue
        
        if dynamics_kind=='ode':
            # in the new pipeline, the noise comes pre-added
            # inputs= inputs + sigma*torch.randn_like(inputs)
            inputs= inputs + sigma_t*torch.randn_like(inputs)

            data_dim= inputs.shape[-1]
            inputs= torch.cat([inputs, times], dim=-1)
            inputs=inputs.reshape(-1, data_dim+1)

            targets= targets.reshape(-1, data_dim)
            
            optimizer.zero_grad(set_to_none=True)
            preds = model(inputs)

            l1_term = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            l2_term = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)


            # import pdb; pdb.set_trace()
            loss = mse_loss(preds, targets, mask)+0.0005*l2_term
        
        elif dynamics_kind=='sde_constant_sigma':
            # the noise needs to be sampled fresh, batchwise for each epochs
            # the eps_t from the score term
            ######
            # this is wrong
            # eps_score= sigma_t*torch.randn_like(inputs)
            ######
            eps_score= torch.randn_like(inputs)
            inputs= inputs + sigma_t*torch.randn_like(inputs)
            data_dim= inputs.shape[-1]
            inputs= torch.cat([inputs, times], dim=-1)
            inputs=inputs.reshape(-1, data_dim+1)
            
            # probability flow velocity
            targets= targets.reshape(-1, data_dim)
            
            optimizer.zero_grad(set_to_none=True)


            preds = model(inputs)
            score_preds= score_model(inputs)
            lambda_t= lambda_t.reshape(-1, data_dim)
            eps_score= eps_score.reshape(-1, data_dim)

            loss_velocity= mse_loss(preds, targets, mask)
            loss_score= torch.mean((lambda_t*score_preds + eps_score) ** 2)
            loss= loss_velocity+loss_score
        
        elif dynamics_kind=='sde_quadratic_sigma':
            eps_score= torch.randn_like(inputs)
            eps_velocity= torch.randn_like(inputs)
            # inputs= inputs + sigma_t*torch.randn_like(inputs)
            inputs= inputs + sigma_t*eps_velocity
            data_dim= inputs.shape[-1]
            inputs= torch.cat([inputs, times], dim=-1)
            inputs=inputs.reshape(-1, data_dim+1)
            
            
            # probability flow velocity
            sigma_t = torch.clamp(sigma_t, min=0.01)
            # targets= eps_velocity*der_sigma_t/(sigma_t + 1e-8) + targets
            targets= eps_velocity*der_sigma_t + targets

            targets= targets.reshape(-1, data_dim)

            optimizer.zero_grad(set_to_none=True)


            preds = model(inputs)
            score_preds= score_model(inputs)
            lambda_t= lambda_t.reshape(-1, data_dim)
            eps_score= eps_score.reshape(-1, data_dim)
            loss_velocity= mse_loss(preds, targets, mask)
            loss_score= torch.mean((lambda_t*score_preds + eps_score) ** 2)
            loss= loss_velocity+loss_score

            # if loss_velocity==torch.inf or loss_score==torch.inf or loss==torch.nan:
            #     import pdb; pdb.set_trace()
            # print('loss_velocity: ', loss_velocity.item(), 'loss_score: ', loss_score.item())

            
    

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)

'''
sigma = 0.25
sf2m_model = MLP(dim=dim, time_varying=True, w=64).to(device)
sf2m_score_model = MLP(dim=dim, time_varying=True, w=64).to(device)
sf2m_optimizer = torch.optim.AdamW(
    list(sf2m_model.parameters()) + list(sf2m_score_model.parameters()), 1e-4
)
SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

max_norm_ut = torch.tensor(0.0)
for i in tqdm(range(10000)):
    sf2m_optimizer.zero_grad()
    t, xt, ut, eps = get_batch(SF2M, X, batch_size, n_times, return_noise=True)
    lambda_t = SF2M.compute_lambda(t % 1)
    vt = sf2m_model(torch.cat([xt, t[:, None]], dim=-1))
    st = sf2m_score_model(torch.cat([xt, t[:, None]], dim=-1))
    flow_loss = torch.mean((vt - ut) ** 2)
    # max_norm_ut = torch.maximum(torch.max(torch.sum(ut**2, dim=1)), max_norm_ut)
    score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
    if i % 1000 == 0:
        # print(max_norm_ut)
        print(f"{i}: {flow_loss.item():0.2f}, {score_loss.item():0.2f}")
    loss = flow_loss + score_loss

    loss.backward()
    sf2m_optimizer.step()
'''

@torch.no_grad()
def evaluate(model: nn.Module, score_model: nn.Module, loader, interpolant_kind="linear", degree=3, subsample_per_interval=8, dynamics_kind='ode', device= 'cpu') -> float:
    model.eval()
    total_loss, batches = 0.0, 0
    for batch in loader:
        # inputs, targets, mask = prepare_flow_matching_batch(batch, interpolant_kind, subsample_per_interval, device)
        inputs, times, targets, mask= batch['inputs'].to(device), batch['times'].to(device), batch['targets'].to(device), batch['mask'].to(device)
        if inputs.numel() == 0:
            continue


        data_dim= inputs.shape[-1]
        inputs= torch.cat([inputs, times], dim=-1)
        inputs=inputs.reshape(-1, data_dim+1)

        targets= targets.reshape(-1, data_dim)

        preds = model(inputs)
        loss = mse_loss(preds, targets, mask)
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)




def main() -> None:
    args = parse_args()
    # base = f'/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/results/SDE_synthetic_postspline_correction/{args.exp_name}_{args.data_config}_{args.interpolant_kind}'
    # base = f'/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/results/SDE_synthetic_quadratic_01/{args.exp_name}_{args.data_config}_{args.interpolant_kind}'
    # base = f'/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/results/debug/{args.exp_name}_{args.data_config}_{args.interpolant_kind}'
    base = f'/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/results/timecomplexity/{args.exp_name}_{args.data_config}_{args.interpolant_kind}'
    # find a free directory
    save_dir = Path(base)
    if save_dir.exists():
        i = 0
        while Path(f"{base}_{i}").exists():
            i += 1
        save_dir = Path(f"{base}_{i}")

    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=str(Path(save_dir) / "log.txt"), level=logging.INFO)
    log = logging.getLogger()
    epoch_dict= {'train':[], 'test':[], 'time_preprocessing':[], 'time':[]}
    final_dict= {'train':[], 'test':[], 'time_preprocessing':[], 'time':[]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    print(f"Using data config: {args.data_config}")
    train_loader_unprocessed, test_loader_unprocessed = get_dataloaders(args.data_config, batch_size=args.batch_size, data_sample_complexity= args.data_sample_complexity)
    sample_batch = next(iter(train_loader_unprocessed))
    data_dim = sample_batch["values"].shape[-1] # exclude the time
    input_dim = data_dim+1  # state + time already included

    # transform the loaders into precomputed interpolants

    # if 'sde' in args.dynamics_kind:
    #     sigma= None
    # else:
    #     sigma= args.sigma

    sigma= args.sigma
    
    time_before_processing = time.time()
    train_loader= ProcessedDataset(train_loader_unprocessed, train=True, interpolant_kind= args.interpolant_kind, degree=args.degree, subsample_per_interval= args.subsample_per_interval, device=device, dynamics_kind=args.dynamics_kind, sigma=sigma)
    test_loader= ProcessedDataset(test_loader_unprocessed, interpolant_kind= args.interpolant_kind, degree=args.degree, subsample_per_interval= args.subsample_per_interval, device=device, stats= train_loader.stats, dynamics_kind=args.dynamics_kind, sigma=train_loader.sigma)
    print("Data loaders created")
    time_after_processing = time.time()
    time_taken = time_after_processing - time_before_processing
    print(f"Time taken to process data: {time_taken:.5f}s")
    log.info(f"Time taken to process data: {time_taken:.5f}s")
    epoch_dict['time_preprocessing'].append([time_taken])

    args.sigma= train_loader.sigma

    if args.dynamics_kind=='ode':
        model= get_model(args.model_config, data_dim, input_dim, device)
        optimizer= torch.optim.AdamW(model.parameters(), lr=args.lr)
        score_model= None
    else:
        model = get_model(args.model_config, data_dim, input_dim, device)
        score_model= get_model(args.model_config, data_dim, input_dim, device, dynamics_kind= args.dynamics_kind)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(score_model.parameters()), lr=args.lr)

    scheduler = None
    if args.lr_scheduler == "cosine":
        t_max = args.lr_t_max if args.lr_t_max > 0 else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=args.lr_eta_min
        )

    for epoch in range(1, args.epochs + 1):
        time_start = time.time()
        train_loss = train_epoch(model, score_model, train_loader, optimizer, args.interpolant_kind, args.degree, args.subsample_per_interval, args.sigma, args.dynamics_kind, device)
        time_end = time.time()
        time_taken = time_end - time_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} lr={current_lr:.2e} time={time_taken:.2f}s")
        log.info(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} lr={current_lr:.2e} time={time_taken:.2f}s")
        epoch_dict['train'].append([epoch, train_loss])
        epoch_dict['time'].append([epoch, time_taken])
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss = evaluate(model, score_model, test_loader, args.interpolant_kind, args.degree, args.subsample_per_interval, args.dynamics_kind, device)
            print(f"          eval_loss={val_loss:.6f}")
            log.info(f"          eval_loss={val_loss:.6f}")
            epoch_dict['test'].append([epoch, val_loss])

        if scheduler is not None:
            scheduler.step()

    final_dict['train'].append([epoch, train_loss])
    final_dict['test'].append([epoch, val_loss])
    print("Training complete.")

    with open(os.path.join(save_dir, 'train_dynamics.json'), 'w') as f:
        json.dump(epoch_dict, f)
    
    with open(os.path.join(save_dir, 'final_mse_metrics.json'), 'w') as f:
        json.dump(final_dict, f)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    if score_model is not None:
        torch.save(score_model.state_dict(), os.path.join(save_dir, "score_model.pt"))
    # draw_trajectories(model, test_loader, NODEFunc, stats=train_loader.stats, save_dir=save_dir, traj_index=0)

    test_validation_metrics_dict= validate_metrics(model, score_model, test_loader, dynamics_kind=args.dynamics_kind, stats=train_loader.stats, sigma= args.sigma, save_dir=save_dir, subsample_per_interval=args.subsample_per_interval, metrics_to_run= ["mse"], trajectories_png=True, num_traj_to_draw=10)

    with open(os.path.join(save_dir, 'test_validation_metrics.json'), 'w') as f:
        json.dump(test_validation_metrics_dict, f)
    
    with open(save_dir / "test_validation_metrics_from_evaluation.json", "w") as f:
        json.dump(test_validation_metrics_dict, f)

if __name__ == "__main__":
    main()
