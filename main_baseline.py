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
from torchdiffeq import odeint
import time
# Ensure the synthetic-data modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Add latent ODE modules (now in local directory)
import latent_ode.lib.utils as latent_ode_utils
from latent_ode.lib.create_latent_ode_model import create_LatentODE_model
from latent_ode.lib.ode_func import ODEFunc as LatentODEFunc
from latent_ode.lib.encoder_decoder import Encoder_z0_ODE_RNN, Encoder_z0_RNN, Decoder
from latent_ode.lib.diffeq_solver import DiffeqSolver
from torch.distributions.normal import Normal

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
    parser.add_argument("--baseline", type=str, default="flow_matching",
                       choices=["flow_matching", "neural_ode", "latent_ode"],
                       help="Training method: flow_matching or neural_ode or latent_ode baseline")
    
    # Latent ODE specific arguments
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension for Latent ODE")
    parser.add_argument("--rec_dims", type=int, default=30, help="Recognition model dimensions for Latent ODE")
    parser.add_argument("--gru_units", type=int, default=100, help="GRU units for Latent ODE encoder")
    parser.add_argument("--units", type=int, default=100, help="Hidden units for Latent ODE networks")
    parser.add_argument("--gen_layers", type=int, default=2, help="Number of layers in generative ODE network")
    parser.add_argument("--rec_layers", type=int, default=2, help="Number of layers in recognition ODE network")
    parser.add_argument("--z0_encoder", type=str, default="odernn", choices=["odernn", "rnn"], help="Type of encoder for Latent ODE")
    parser.add_argument("--obsrv_std", type=float, default=0.01, help="Observation noise std for Latent ODE")
    parser.add_argument("--kl_coef", type=float, default=1.0, help="KL divergence coefficient for Latent ODE")
    
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



########## for neuralode

# ============== NeuralODE Baseline Functions ==============

def remove_close_times(times, min_dt=1e-4):
    """Remove time points that are too close together."""
    if len(times) < 2:
        return times, torch.arange(len(times), device=times.device)
    
    # Start with first time point
    keep_indices = [0]
    last_kept_time = times[0]
    
    for i in range(1, len(times)):
        if times[i] - last_kept_time > min_dt:
            keep_indices.append(i)
            last_kept_time = times[i]
    
    if len(keep_indices) < 2:
        # Need at least 2 points, so keep first and last
        keep_indices = [0, len(times) - 1]
    
    filtered_times = times[keep_indices]
    
    # Create mapping from original to filtered indices
    mapping = torch.zeros(len(times), dtype=torch.long, device=times.device)
    for orig_idx in range(len(times)):
        # Find closest kept time
        diffs = torch.abs(times[orig_idx] - filtered_times)
        mapping[orig_idx] = torch.argmin(diffs)
    
    return filtered_times, mapping


def train_epoch_neural_ode(model, loader, optimizer, device):
    """NeuralODE: supervised trajectory reconstruction (irregular sampling)."""
    from models.odeint_classes import NODEFunc
    
    model.train()
    total_loss, total_points = 0.0, 0
    
    for batch in loader:
        times = batch['times'].to(device)           # (B, T)
        values = batch['values'].to(device)         # (B, T, D)
        mask = batch['mask'].to(device)             # (B, T, D)
        
        if times.dim() == 3:
            times = times.squeeze(-1)
        
        B, T, D = values.shape
        
        # For sparse data, use the actual time range instead of std for normalization
        # This prevents time points from collapsing when std is small
        time_min = times.min()
        time_max = times.max()
        time_range = time_max - time_min
        
        # If time range is too small, use std but with a larger minimum
        if time_range < 1e-3:
            time_scale = torch.clamp(times.std(), min=1e-2)
            time_center = times.mean()
        else:
            time_scale = time_range
            time_center = time_min
        
        # Compute simple normalization stats on-the-fly
        stats = {
            'values_mean': values.mean(dim=(0,1), keepdim=True).squeeze(0),
            'values_std': values.std(dim=(0,1), keepdim=True).squeeze(0) + 1e-8,
            'times_mean': time_center,
            'times_std': time_scale,
            'targets_mean': torch.zeros(D, device=device),
            'targets_std': torch.ones(D, device=device),
        }
        
        # Normalize
        values_norm = (values - stats['values_mean']) / stats['values_std']
        times_norm = (times - stats['times_mean']) / stats['times_std']
        
        node_func = NODEFunc(model, stats).to(device)
        
        optimizer.zero_grad()
        
        # Check if same grid
        same_grid = torch.allclose(times_norm, times_norm[0].unsqueeze(0), atol=1e-6)
        
        if same_grid:
            # Batch ODE
            t_grid = times_norm[0]
            
            # Remove times that are too close together (more robust than just unique)
            filtered_times, time_mapping = remove_close_times(t_grid, min_dt=1e-4)
            
            # Check if we have at least 2 time points
            if len(filtered_times) < 2:
                continue  # Skip this batch if not enough time points
            
            x0 = values_norm[:, 0, :]
            x_pred = odeint(node_func, x0, filtered_times, rtol=1e-5, atol=1e-5).permute(1, 0, 2)
            
            # Map predictions back to original time grid
            x_pred = x_pred[:, time_mapping, :]
            
            sq_err = ((x_pred - values_norm) ** 2) * mask
            loss = sq_err.sum() / mask.sum()
        else:
            # Per-trajectory
            batch_loss = 0.0
            batch_points = 0
            for b in range(B):
                t_b = times_norm[b]
                x_b = values_norm[b]
                mask_b = mask[b]
                valid = (mask_b.sum(dim=-1) > 0)
                if valid.sum() == 0:
                    continue
                t_valid = t_b[valid]
                x_valid = x_b[valid]
                mask_valid = mask_b[valid]
                
                # Remove times that are too close together
                filtered_times, time_mapping = remove_close_times(t_valid, min_dt=1e-4)
                
                if len(filtered_times) < 2:
                    continue  # Need at least 2 time points
                
                # Get initial condition
                x0_b = x_valid[0:1]
                
                x_pred_b = odeint(node_func, x0_b, filtered_times, rtol=1e-5, atol=1e-5)[:, 0, :]
                
                # Map back to original indices
                x_pred_b = x_pred_b[time_mapping]
                
                sq_err_b = ((x_pred_b - x_valid) ** 2) * mask_valid
                batch_loss += sq_err_b.sum()
                batch_points += mask_valid.sum()
            
            if batch_points == 0:
                continue
            loss = batch_loss / batch_points
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * mask.sum().item()
        total_points += mask.sum().item()
    
    return total_loss / max(total_points, 1)


@torch.no_grad()
def evaluate_neural_ode(model, loader, device):
    """Evaluate NeuralODE on irregular data."""
    from models.odeint_classes import NODEFunc
    
    model.eval()
    total_loss, total_points = 0.0, 0
    
    for batch in loader:
        times = batch['times'].to(device)
        values = batch['values'].to(device)
        mask = batch['mask'].to(device)
        
        if times.dim() == 3:
            times = times.squeeze(-1)
        
        B, T, D = values.shape
        
        # For sparse data, use the actual time range instead of std for normalization
        time_min = times.min()
        time_max = times.max()
        time_range = time_max - time_min
        
        if time_range < 1e-3:
            time_scale = torch.clamp(times.std(), min=1e-2)
            time_center = times.mean()
        else:
            time_scale = time_range
            time_center = time_min
        
        # Same normalization
        stats = {
            'values_mean': values.mean(dim=(0,1), keepdim=True).squeeze(0),
            'values_std': values.std(dim=(0,1), keepdim=True).squeeze(0) + 1e-8,
            'times_mean': time_center,
            'times_std': time_scale,
            'targets_mean': torch.zeros(D, device=device),
            'targets_std': torch.ones(D, device=device),
        }
        
        values_norm = (values - stats['values_mean']) / stats['values_std']
        times_norm = (times - stats['times_mean']) / stats['times_std']
        
        node_func = NODEFunc(model, stats).to(device)
        
        same_grid = torch.allclose(times_norm, times_norm[0].unsqueeze(0), atol=1e-6)
        
        if same_grid:
            t_grid = times_norm[0]
            
            # Remove times that are too close together
            filtered_times, time_mapping = remove_close_times(t_grid, min_dt=1e-4)
            
            if len(filtered_times) < 2:
                continue
            
            x0 = values_norm[:, 0, :]
            x_pred = odeint(node_func, x0, filtered_times, rtol=1e-5, atol=1e-5).permute(1, 0, 2)
            
            # Map predictions back to original time grid
            x_pred = x_pred[:, time_mapping, :]
            
            sq_err = ((x_pred - values_norm) ** 2) * mask
            total_loss += sq_err.sum().item()
            total_points += mask.sum().item()
        else:
            for b in range(B):
                t_b = times_norm[b]
                x_b = values_norm[b]
                mask_b = mask[b]
                valid = (mask_b.sum(dim=-1) > 0)
                if valid.sum() == 0:
                    continue
                t_valid = t_b[valid]
                x_valid = x_b[valid]
                mask_valid = mask_b[valid]
                
                # Remove times that are too close together
                filtered_times, time_mapping = remove_close_times(t_valid, min_dt=1e-4)
                
                if len(filtered_times) < 2:
                    continue
                
                x0_b = x_valid[0:1]
                x_pred_b = odeint(node_func, x0_b, filtered_times, rtol=1e-5, atol=1e-5)[:, 0, :]
                
                # Map back to original indices
                x_pred_b = x_pred_b[time_mapping]
                
                sq_err_b = ((x_pred_b - x_valid) ** 2) * mask_valid
                total_loss += sq_err_b.sum().item()
                total_points += mask_valid.sum().item()
    
    return total_loss / max(total_points, 1)

###########

# ============== Latent ODE Baseline Functions ==============

def prepare_latent_ode_batch(batch, device, normalize=True):
    """Prepare batch in the format expected by Latent ODE."""
    times = batch['times'].to(device)           # (B, T)
    values = batch['values'].to(device)         # (B, T, D)
    mask = batch['mask'].to(device)             # (B, T, D)
    
    if times.dim() == 3:
        times = times.squeeze(-1)
    
    # Replace NaN/inf values with 0 where mask is 0 (missing data)
    # This is important because Latent ODE expects clean data with mask indicating missingness
    values_clean = values.clone()
    
    # Handle NaN values
    nan_mask = torch.isnan(values_clean)
    values_clean[nan_mask] = 0.0
    
    # Handle inf values
    inf_mask = torch.isinf(values_clean)
    values_clean[inf_mask] = 0.0
    
    # Also ensure mask is binary (0 or 1) and has no NaN/inf
    mask_clean = mask.clone().float()
    mask_clean[torch.isnan(mask_clean)] = 0.0
    mask_clean[torch.isinf(mask_clean)] = 0.0
    mask_clean = (mask_clean > 0.5).float()  # Binarize
    
    # Ensure times has no NaN
    if torch.isnan(times).any():
        raise ValueError("Times contain NaN values!")
    
    # Normalize data (important for stable training!)
    if normalize:
        # Only compute stats over observed (masked) values
        B, T, D = values_clean.shape
        
        # For values: compute mean/std only where mask is 1
        masked_values = values_clean * mask_clean
        n_observed = mask_clean.sum(dim=(0, 1)) + 1e-8  # Per dimension
        
        values_mean = masked_values.sum(dim=(0, 1)) / n_observed  # (D,)
        values_centered = (values_clean - values_mean.unsqueeze(0).unsqueeze(0)) * mask_clean
        values_std = torch.sqrt((values_centered ** 2).sum(dim=(0, 1)) / n_observed) + 1e-8  # (D,)
        
        # Normalize values
        values_norm = (values_clean - values_mean.unsqueeze(0).unsqueeze(0)) / values_std.unsqueeze(0).unsqueeze(0)
        
        # For times: use range-based normalization (more stable for sparse data)
        time_min = times.min()
        time_max = times.max()
        time_range = time_max - time_min
        
        if time_range < 1e-3:
            time_scale = torch.clamp(times.std(), min=1e-2)
            time_center = times.mean()
        else:
            time_scale = time_range
            time_center = time_min
        
        times_norm = (times - time_center) / time_scale
    else:
        values_norm = values_clean
        times_norm = times
    
    # For latent ODE, we use all data for both observation and prediction (interpolation task)
    batch_dict = {
        'observed_data': values_norm,
        'data_to_predict': values_norm,
        'observed_tp': times_norm[0],  # Assuming same time grid for all trajectories
        'tp_to_predict': times_norm[0],
        'observed_mask': mask_clean,
        'mask_predicted_data': mask_clean,
        'labels': None,
        'mode': 'interp'
    }
    
    return batch_dict


def create_latent_ode_model_simple(args, input_dim, device):
    """Create a Latent ODE model with simplified configuration."""
    latent_dim = args.latent_dim
    
    # Create z0 prior (standard normal)
    z0_prior = Normal(
        torch.zeros(latent_dim).to(device),
        torch.ones(latent_dim).to(device)
    )
    
    # Create generative ODE function network
    gen_ode_func_net = latent_ode_utils.create_net(
        latent_dim, latent_dim,
        n_layers=args.gen_layers,
        n_units=args.units,
        nonlinear=nn.Tanh
    )
    
    gen_ode_func = LatentODEFunc(
        input_dim=input_dim,
        latent_dim=latent_dim,
        ode_func_net=gen_ode_func_net,
        device=device
    ).to(device)
    
    # Create diffeq solver for generative model
    diffeq_solver = DiffeqSolver(
        input_dim, gen_ode_func, 'dopri5', latent_dim,
        odeint_rtol=1e-3, odeint_atol=1e-4, device=device
    )
    
    # Create encoder
    if args.z0_encoder == "odernn":
        rec_ode_func_net = latent_ode_utils.create_net(
            args.rec_dims, args.rec_dims,
            n_layers=args.rec_layers,
            n_units=args.units,
            nonlinear=nn.Tanh
        )
        
        rec_ode_func = LatentODEFunc(
            input_dim=input_dim * 2,  # data + mask
            latent_dim=args.rec_dims,
            ode_func_net=rec_ode_func_net,
            device=device
        ).to(device)
        
        z0_diffeq_solver = DiffeqSolver(
            input_dim * 2, rec_ode_func, "euler", latent_dim,
            odeint_rtol=1e-3, odeint_atol=1e-4, device=device
        )
        
        encoder_z0 = Encoder_z0_ODE_RNN(
            args.rec_dims, input_dim * 2, z0_diffeq_solver,
            z0_dim=latent_dim, n_gru_units=args.gru_units, device=device
        ).to(device)
    else:
        encoder_z0 = Encoder_z0_RNN(
            latent_dim, input_dim * 2,
            lstm_output_size=args.rec_dims, device=device
        ).to(device)
    
    # Create decoder
    decoder = Decoder(latent_dim, input_dim).to(device)
    
    # Import LatentODE class
    from latent_ode.lib.latent_ode import LatentODE
    
    model = LatentODE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_z0=encoder_z0,
        decoder=decoder,
        diffeq_solver=diffeq_solver,
        z0_prior=z0_prior,
        device=device,
        obsrv_std=args.obsrv_std,
        use_poisson_proc=False,
        use_binary_classif=False,
        classif_per_tp=False,
        n_labels=1,
        train_classif_w_reconstr=False
    ).to(device)
    
    return model


def train_epoch_latent_ode(model, loader, optimizer, kl_coef, device):
    """Train Latent ODE for one epoch."""
    model.train()
    total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
    batches = 0
    
    for batch in loader:
        batch_dict = prepare_latent_ode_batch(batch, device)
        
        optimizer.zero_grad()
        results, _ = model.compute_all_losses(batch_dict, n_traj_samples=1, kl_coef=kl_coef)
        
        loss = results['loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += results['mse'].item()
        if 'kl_first_p' in results:
            total_kl += results['kl_first_p'].item()
        batches += 1
    
    return total_loss / max(batches, 1), total_mse / max(batches, 1), total_kl / max(batches, 1)


@torch.no_grad()
def evaluate_latent_ode(model, loader, kl_coef, device):
    """Evaluate Latent ODE."""
    model.eval()
    total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
    batches = 0
    
    for batch in loader:
        batch_dict = prepare_latent_ode_batch(batch, device)
        
        results, _ = model.compute_all_losses(batch_dict, n_traj_samples=1, kl_coef=kl_coef)
        
        total_loss += results['loss'].item()
        total_mse += results['mse'].item()
        if 'kl_first_p' in results:
            total_kl += results['kl_first_p'].item()
        batches += 1
    
    return total_loss / max(batches, 1), total_mse / max(batches, 1), total_kl / max(batches, 1)


@torch.no_grad()
def draw_latent_ode_trajectories(model, loader, save_dir, num_traj_to_draw=5):
    """Draw trajectories for Latent ODE predictions."""
    model.eval()
    
    # Get one batch
    batch = next(iter(loader))
    batch_dict = prepare_latent_ode_batch(batch, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Get predictions
    pred_y, info = model.get_reconstruction(
        batch_dict["tp_to_predict"],
        batch_dict["observed_data"],
        batch_dict["observed_tp"],
        mask=batch_dict["observed_mask"],
        n_traj_samples=1
    )
    
    # pred_y shape: [n_traj_samples, n_traj, n_tp, n_dims]
    pred_y = pred_y[0]  # Take first sample: [n_traj, n_tp, n_dims]
    true_y = batch_dict["data_to_predict"]  # [n_traj, n_tp, n_dims]
    times = batch_dict["tp_to_predict"]  # [n_tp]
    
    # Convert to numpy
    pred_y_np = pred_y.cpu().numpy()
    true_y_np = true_y.cpu().numpy()
    times_np = times.cpu().numpy()
    
    n_traj, n_tp, n_dims = true_y_np.shape
    num_to_draw = min(num_traj_to_draw, n_traj)
    
    # Create plots
    fig, axes = plt.subplots(num_to_draw, n_dims, figsize=(4 * n_dims, 3 * num_to_draw))
    if num_to_draw == 1:
        axes = axes.reshape(1, -1)
    if n_dims == 1:
        axes = axes.reshape(-1, 1)
    
    for traj_idx in range(num_to_draw):
        for dim_idx in range(n_dims):
            ax = axes[traj_idx, dim_idx]
            ax.plot(times_np, true_y_np[traj_idx, :, dim_idx], 'b-', label='True', alpha=0.7)
            ax.plot(times_np, pred_y_np[traj_idx, :, dim_idx], 'r--', label='Predicted', alpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel(f'Dim {dim_idx}')
            ax.set_title(f'Trajectory {traj_idx}, Dim {dim_idx}')
            if traj_idx == 0 and dim_idx == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'latent_ode_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Latent ODE trajectories to {save_dir / 'latent_ode_trajectories.png'}")


def main() -> None:
    args = parse_args()
    
    # Determine save path based on baseline
    if args.baseline == "neural_ode":
        base = f'/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/results/baselines/neural_ode/{args.exp_name}_{args.data_config}'
    elif args.baseline == "latent_ode":
        base = f'/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/results/baselines/latent_ode/{args.exp_name}_{args.data_config}'
    else:
        base = f'/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/results/SDE_synthetic_quadratic/{args.exp_name}_{args.data_config}_{args.interpolant_kind}'
    
    # Find free directory
    save_dir = Path(base)
    if save_dir.exists():
        i = 0
        while Path(f"{base}_{i}").exists():
            i += 1
        save_dir = Path(f"{base}_{i}")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(filename=str(Path(save_dir) / "log.txt"), level=logging.INFO)
    log = logging.getLogger()
    epoch_dict = {'train': [], 'test': [], 'time': []}
    final_dict = {'train': [], 'test': [], 'time': []}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using data config: {args.data_config}, baseline: {args.baseline}")
    train_loader_unprocessed, test_loader_unprocessed = get_dataloaders(
        args.data_config, batch_size=args.batch_size, 
        data_sample_complexity=args.data_sample_complexity
    )
    sample_batch = next(iter(train_loader_unprocessed))
    data_dim = sample_batch["values"].shape[-1]
    input_dim = data_dim + 1
    
    # ========== BRANCH ON BASELINE ==========
    if args.baseline == "neural_ode":
        print("=" * 50)
        print("Training NeuralODE baseline...")
        print("=" * 50)
        
        # NeuralODE doesn't need interpolants
        model = get_model(args.model_config, data_dim, input_dim, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        score_model = None
        
        scheduler = None
        if args.lr_scheduler == "cosine":
            t_max = args.lr_t_max if args.lr_t_max > 0 else args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=args.lr_eta_min
            )
        
        # Training loop for NeuralODE
        for epoch in range(1, args.epochs + 1):
            time_start = time.time()
            train_loss = train_epoch_neural_ode(model, train_loader_unprocessed, optimizer, device)
            time_end = time.time()
            time_taken = time_end - time_start
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} lr={current_lr:.2e} time={time_taken:.2f}s")
            log.info(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} lr={current_lr:.2e} time={time_taken:.2f}s")
            epoch_dict['train'].append([epoch, train_loss])
            epoch_dict['time'].append([epoch, time_taken])
            
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                val_loss = evaluate_neural_ode(model, test_loader_unprocessed, device)
                print(f"          eval_loss={val_loss:.6f}")
                log.info(f"          eval_loss={val_loss:.6f}")
                epoch_dict['test'].append([epoch, val_loss])
            
            if scheduler is not None:
                scheduler.step()
        
        # For validation metrics, create ProcessedDataset just to get stats
        print("Creating processed loaders for validation metrics...")
        train_loader = ProcessedDataset(
            train_loader_unprocessed, train=True,
            interpolant_kind='linear', degree=3, subsample_per_interval=1,
            device=device, dynamics_kind='ode', sigma=0.01
        )
        test_loader = ProcessedDataset(
            test_loader_unprocessed,
            interpolant_kind='linear', degree=3, subsample_per_interval=1,
            device=device, stats=train_loader.stats, dynamics_kind='ode', sigma=0.01
        )
        args.sigma = 0.01
    
    elif args.baseline == "latent_ode":
        print("=" * 50)
        print("Training Latent ODE baseline...")
        print("=" * 50)
        
        # Create Latent ODE model
        model = create_latent_ode_model_simple(args, data_dim, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        score_model = None
        
        scheduler = None
        if args.lr_scheduler == "cosine":
            t_max = args.lr_t_max if args.lr_t_max > 0 else args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=args.lr_eta_min
            )
        
        # Training loop for Latent ODE
        for epoch in range(1, args.epochs + 1):
            time_start = time.time()
            train_loss, train_mse, train_kl = train_epoch_latent_ode(
                model, train_loader_unprocessed, optimizer, args.kl_coef, device
            )
            time_end = time.time()
            time_taken = time_end - time_start
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} train_mse={train_mse:.6f} train_kl={train_kl:.6f} lr={current_lr:.2e}")
            log.info(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} train_mse={train_mse:.6f} train_kl={train_kl:.6f} lr={current_lr:.2e}")
            epoch_dict['train'].append([epoch, train_loss])
            epoch_dict['time'].append([epoch, time_taken])
            
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                val_loss, val_mse, val_kl = evaluate_latent_ode(
                    model, test_loader_unprocessed, args.kl_coef, device
                )
                print(f"          eval_loss={val_loss:.6f} eval_mse={val_mse:.6f} eval_kl={val_kl:.6f}")
                log.info(f"          eval_loss={val_loss:.6f} eval_mse={val_mse:.6f} eval_kl={val_kl:.6f}")
                epoch_dict['test'].append([epoch, val_loss])
            
            if scheduler is not None:
                scheduler.step()
        
        # For validation metrics, create ProcessedDataset just to get stats
        print("Creating processed loaders for validation metrics...")
        train_loader = ProcessedDataset(
            train_loader_unprocessed, train=True,
            interpolant_kind='linear', degree=3, subsample_per_interval=1,
            device=device, dynamics_kind='ode', sigma=0.01
        )
        test_loader = ProcessedDataset(
            test_loader_unprocessed,
            interpolant_kind='linear', degree=3, subsample_per_interval=1,
            device=device, stats=train_loader.stats, dynamics_kind='ode', sigma=0.01
        )
        args.sigma = 0.01
    
    else:  # flow_matching
        print("=" * 50)
        print("Training Flow Matching...")
        print("=" * 50)
        
        sigma = args.sigma
        train_loader = ProcessedDataset(
            train_loader_unprocessed, train=True,
            interpolant_kind=args.interpolant_kind, degree=args.degree,
            subsample_per_interval=args.subsample_per_interval,
            device=device, dynamics_kind=args.dynamics_kind, sigma=sigma
        )
        test_loader = ProcessedDataset(
            test_loader_unprocessed,
            interpolant_kind=args.interpolant_kind, degree=args.degree,
            subsample_per_interval=args.subsample_per_interval,
            device=device, stats=train_loader.stats,
            dynamics_kind=args.dynamics_kind, sigma=train_loader.sigma
        )
        print("Data loaders created")
        args.sigma = train_loader.sigma
        
        # Model setup
        if args.dynamics_kind == 'ode':
            model = get_model(args.model_config, data_dim, input_dim, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            score_model = None
        else:
            model = get_model(args.model_config, data_dim, input_dim, device)
            score_model = get_model(args.model_config, data_dim, input_dim, device, 
                                   dynamics_kind=args.dynamics_kind)
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(score_model.parameters()), lr=args.lr
            )
        
        scheduler = None
        if args.lr_scheduler == "cosine":
            t_max = args.lr_t_max if args.lr_t_max > 0 else args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=args.lr_eta_min
            )
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            time_start = time.time()
            train_loss = train_epoch(
                model, score_model, train_loader, optimizer,
                args.interpolant_kind, args.degree, args.subsample_per_interval,
                args.sigma, args.dynamics_kind, device
            )
            time_end = time.time()
            time_taken = time_end - time_start
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} lr={current_lr:.2e} time={time_taken:.2f}s")
            log.info(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} lr={current_lr:.2e} time={time_taken:.2f}s")
            epoch_dict['train'].append([epoch, train_loss])
            epoch_dict['time'].append([epoch, time_taken])
            
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                val_loss = evaluate(
                    model, score_model, test_loader,
                    args.interpolant_kind, args.degree, args.subsample_per_interval,
                    args.dynamics_kind, device
                )
                print(f"          eval_loss={val_loss:.6f}")
                log.info(f"          eval_loss={val_loss:.6f}")
                epoch_dict['test'].append([epoch, val_loss])
            
            if scheduler is not None:
                scheduler.step()
    
    # ========== COMMON SAVING/VALIDATION ==========
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
    
    print("Running validation metrics...")
    
    # Determine dynamics_kind for validation
    if args.baseline == "flow_matching":
        val_dynamics_kind = args.dynamics_kind
    else:
        val_dynamics_kind = "ode"
    
    # For latent ODE, we need to handle validation differently
    if args.baseline == "latent_ode":
        # Draw trajectories for latent ODE
        draw_latent_ode_trajectories(model, test_loader_unprocessed, save_dir, num_traj_to_draw=10)
        
        # For latent ODE, we'll just use the MSE from evaluation
        test_validation_metrics_dict = {
            'mse': val_mse,
            'loss': val_loss,
            'kl': val_kl
        }
    else:
        test_validation_metrics_dict = validate_metrics(
            model, score_model, test_loader,
            dynamics_kind=val_dynamics_kind,
            stats=train_loader.stats, sigma=args.sigma, save_dir=save_dir,
            subsample_per_interval=args.subsample_per_interval,
            metrics_to_run=["mse"], trajectories_png=True, num_traj_to_draw=10
        )
    
    with open(os.path.join(save_dir, 'test_validation_metrics.json'), 'w') as f:
        json.dump(test_validation_metrics_dict, f)
    
    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    main()