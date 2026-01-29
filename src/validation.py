import torch
import numpy as np

import os
import random
import json
import torch
import os
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torchsde import sdeint
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from models.odeint_classes import NODEFunc, StochasticNODE
from scipy.stats import wasserstein_distance

from sklearn.metrics import r2_score
import dcor
import ot  # POT
import torch


def compute_mmd_multi_rbf(X, Y, gammas=(2, 1, 0.5, 0.1, 0.01, 0.005)):
    XX = torch.cdist(X, X, p=2) ** 2
    YY = torch.cdist(Y, Y, p=2) ** 2
    XY = torch.cdist(X, Y, p=2) ** 2
    m, n = X.size(0), Y.size(0)
    if m < 2 or n < 2:
        return 0.0
    mmd_total = 0.0
    for gamma in gammas:
        K_XX = torch.exp(-gamma * XX)
        K_YY = torch.exp(-gamma * YY)
        K_XY = torch.exp(-gamma * XY)
        mmd2 = (K_XX.sum() - torch.diagonal(K_XX).sum()) / (m * (m - 1)) \
             + (K_YY.sum() - torch.diagonal(K_YY).sum()) / (n * (n - 1)) \
             - 2 * K_XY.mean()
        mmd_total += mmd2
    return float((mmd_total / len(gammas)).item())

def compute_energy_distance(X, Y):
    # true multivariate energy distance (as in constrained_FM)
    return float(dcor.energy_distance(X.detach().cpu().numpy(), Y.detach().cpu().numpy()))

def compute_r2(X, Y):
    # flattened R2 (as in constrained_FM)
    Xf = X.detach().cpu().numpy().reshape(-1)
    Yf = Y.detach().cpu().numpy().reshape(-1)
    return float(r2_score(Yf, Xf))


def mse_of_predicted_traj(vf_net, test_loader, stats, subsample_per_interval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vf_net = vf_net.to(device)
    vf_net.eval()
    node_func = NODEFunc(vf_net, stats).to(device)

    total_sq_error = 0.0
    total_num_points = 0

    with torch.no_grad():
        for batch in test_loader:
            targets = batch['targets'].to(device)  # (B, T_refined, D) – only for D
            orig_t  = batch['original_times_normalized'].to(device)    # (B, T) or (B, T, 1)
            orig_x  = batch['original_values_normalized'].to(device)   # (B, T, F_or_D)

            state_dim = targets.shape[-1]  # D

            if orig_x.shape[-1] > state_dim:
                x_true_all = orig_x[..., :state_dim]   # (B, T, D)
            else:
                x_true_all = orig_x                   # (B, T, D)

            if orig_t.dim() == 3 and orig_t.size(-1) == 1:
                orig_t = orig_t.squeeze(-1)           # (B, T)

            # assert same time grid across batch
            t0 = orig_t[0]                            # (T,)
            assert torch.allclose(orig_t, t0.unsqueeze(0)), \
                "orig_t differs across batch; need per-trajectory loop"

            B, T, D = x_true_all.shape

            # initial state for all B trajectories
            x0 = x_true_all[:, 0, :]                  # (B, D)

            # batched ODE solve: outputs (T, B, D)
            x_pred = odeint(node_func, x0, t0, rtol=1e-6, atol=1e-6)
            x_pred = x_pred.permute(1, 0, 2)          # (B, T, D)

            sq_err = (x_pred - x_true_all) ** 2       # (B, T, D)
            total_sq_error += sq_err.sum().item()
            total_num_points += sq_err.numel()

    mse = total_sq_error / total_num_points
    return mse


# def metrics_of_predicted_stochastic_traj(
#     vf_net,
#     score_model,
#     dynamics_kind,
#     test_loader,
#     stats,
#     sigma,
#     subsample_per_interval,
#     num_sliced_projections=128,
#     seed=0,
# ):
#     """
#     Stochastic traj metrics (SDE):
#       - mse: pointwise MSE between ONE sampled SDE rollout and x_true (paired by traj)
#       - sliced_wasserstein: joint sliced Wasserstein between {x_pred(t)} and {x_true(t)} over trajectories, averaged over time
#       - mmd: RBF-MMD^2 between {x_pred(t)} and {x_true(t)} over trajectories, averaged over time
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if score_model is None:
#         raise ValueError("SDE metrics require score_model (got None).")

#     vf_net = vf_net.to(device).eval()
#     score_model = score_model.to(device).eval()
#     sde_func = StochasticNODE(vf_net, score_model, stats, sigma).to(device)

#     def _sliced_wasserstein_joint(x: torch.Tensor, y: torch.Tensor) -> float:
#         # x,y: (N,D) and (M,D)
#         try:
#             import ot  # POT
#             x_np = x.detach().float().cpu().numpy()
#             y_np = y.detach().float().cpu().numpy()
#             return float(
#                 ot.sliced_wasserstein_distance(
#                     x_np, y_np, n_projections=num_sliced_projections, seed=seed
#                 )
#             )
#         except Exception:
#             # fallback: SciPy 1D wasserstein on random projections
#             g = torch.Generator(device=x.device)
#             g.manual_seed(seed)
#             D = x.size(-1)
#             theta = torch.randn(num_sliced_projections, D, generator=g, device=x.device)
#             theta = theta / (theta.norm(dim=-1, keepdim=True) + 1e-12)

#             x_proj = (x @ theta.T).detach().cpu().numpy()  # (N,P)
#             y_proj = (y @ theta.T).detach().cpu().numpy()  # (M,P)

#             sw = 0.0
#             for p in range(num_sliced_projections):
#                 sw += wasserstein_distance(x_proj[:, p], y_proj[:, p])
#             return float(sw / num_sliced_projections)

#     def _mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         # x,y: (N,D) and (M,D)
#         z = torch.cat([x, y], dim=0)
#         d2 = torch.cdist(z, z, p=2) ** 2
#         d2 = d2[~torch.eye(d2.size(0), dtype=torch.bool, device=d2.device)]
#         sigma2 = torch.clamp(torch.median(d2), min=1e-6)

#         def k(a, b):
#             return torch.exp(-(torch.cdist(a, b, p=2) ** 2) / (2.0 * sigma2))

#         Kxx = k(x, x)
#         Kyy = k(y, y)
#         Kxy = k(x, y)

#         n = x.size(0)
#         m = y.size(0)
#         if n < 2 or m < 2:
#             return torch.tensor(0.0, device=x.device)

#         mmd2 = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (n * (n - 1))
#         mmd2 = mmd2 + (Kyy.sum() - torch.diagonal(Kyy).sum()) / (m * (m - 1))
#         mmd2 = mmd2 - 2.0 * Kxy.mean()
#         return mmd2

#     total_sq_error = 0.0
#     total_num_points = 0

#     sw_sum = 0.0
#     mmd_sum = 0.0
#     num_terms = 0

#     with torch.no_grad():
#         for batch in test_loader:
#             targets = batch["targets"].to(device)  # for state_dim
#             orig_t = batch["original_times_normalized"].to(device)
#             orig_x = batch["original_values_normalized"].to(device)

#             state_dim = targets.shape[-1]
#             x_true_all = orig_x[..., :state_dim] if orig_x.shape[-1] > state_dim else orig_x

#             if orig_t.dim() == 3 and orig_t.size(-1) == 1:
#                 orig_t = orig_t.squeeze(-1)

#             # assume same time grid across batch (matches existing mse_of_predicted_traj)
#             t0 = orig_t[0]
#             assert torch.allclose(orig_t, t0.unsqueeze(0)), \
#                 "orig_t differs across batch; need per-trajectory loop"

#             x0 = x_true_all[:, 0, :]  # (B,D)

#             # sample one SDE rollout per trajectory: (T,B,D) -> (B,T,D)
#             x_pred = sdeint(sde_func, x0, t0).permute(1, 0, 2)

#             # stochastic MSE (paired)
#             sq_err = (x_pred - x_true_all) ** 2
#             total_sq_error += sq_err.sum().item()
#             total_num_points += sq_err.numel()

#             # distribution metrics over trajectories at subsampled timepoints
#             B, T, D = x_true_all.shape
#             t_idx = torch.arange(0, T, max(1, subsample_per_interval), device=device)
#             for ti in t_idx:
#                 xt = x_true_all[:, ti, :]  # (B,D)
#                 xp = x_pred[:, ti, :]      # (B,D)
#                 sw_sum += _sliced_wasserstein_joint(xp, xt)
#                 mmd_sum += float(_mmd_rbf(xp, xt).item())
#                 num_terms += 1

#     mse = total_sq_error / max(total_num_points, 1)
#     sliced_wasserstein = sw_sum / max(num_terms, 1)
#     mmd = mmd_sum / max(num_terms, 1)
#     return mse, sliced_wasserstein, mmd

def metrics_of_predicted_stochastic_traj(
    vf_net,
    score_model,
    dynamics_kind,
    test_loader,
    stats,
    sigma,
    subsample_per_interval,
    n_projections=200,
):
    """
    Returns: stochastic_mse, sliced_wasserstein, mmd, energy, r2
    SWD is joint (not dim-wise): ot.sliced.sliced_wasserstein_distance on (B,D) samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if score_model is None:
        raise ValueError("SDE metrics require score_model (got None).")

    vf_net = vf_net.to(device).eval()
    score_model = score_model.to(device).eval()
    sde_func = StochasticNODE(vf_net, score_model, stats, sigma).to(device)

    total_sq_error, total_num_points = 0.0, 0
    sw_sum = mmd_sum = energy_sum = r2_sum = 0.0
    num_terms = 0

    with torch.no_grad():
        for batch in test_loader:
            targets = batch["targets"].to(device)
            orig_t = batch["original_times_normalized"].to(device)
            orig_x = batch["original_values_normalized"].to(device)

            state_dim = targets.shape[-1]
            x_true_all = orig_x[..., :state_dim] if orig_x.shape[-1] > state_dim else orig_x

            if orig_t.dim() == 3 and orig_t.size(-1) == 1:
                orig_t = orig_t.squeeze(-1)

            t0 = orig_t[0]
            assert torch.allclose(orig_t, t0.unsqueeze(0)), "orig_t differs across batch"

            x0 = x_true_all[:, 0, :]  # (B,D)
            x_pred = sdeint(sde_func, x0, t0).permute(1, 0, 2)  # (B,T,D)

            sq_err = (x_pred - x_true_all) ** 2
            total_sq_error += sq_err.sum().item()
            total_num_points += sq_err.numel()

            B, T, D = x_true_all.shape
            t_idx = torch.arange(0, T, max(1, subsample_per_interval), device=device)

            for ti in t_idx:
                xt = x_true_all[:, ti, :]  # (B,D)
                xp = x_pred[:, ti, :]      # (B,D)

                sw_sum += float(
                    ot.sliced.sliced_wasserstein_distance(
                        xp.detach().cpu().numpy(),
                        xt.detach().cpu().numpy(),
                        n_projections=n_projections,
                        p=2,
                    )
                )
                mmd_sum += compute_mmd_multi_rbf(xp, xt)
                energy_sum += compute_energy_distance(xp, xt)
                # r2_sum += compute_r2(xp, xt)
                num_terms += 1

    stochastic_mse = total_sq_error / max(total_num_points, 1)
    sliced_wasserstein = sw_sum / max(num_terms, 1)
    mmd = mmd_sum / max(num_terms, 1)
    energy = energy_sum / max(num_terms, 1)
    # r2 = r2_sum / max(num_terms, 1)
    return stochastic_mse, sliced_wasserstein, mmd, energy

    



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_trajectories(vf_net, score_model, dynamics_kind, test_loader, stats, sigma, save_dir=None, num_traj_to_draw=5):
    """
    Plot true vs predicted trajectories for a single example from test_loader.

    Assumptions about test_loader __getitem__ / batch:
        batch is a dict with keys:
            - 'inputs':  (B, T, F) or (B, T, D)  -- model input features per time
            - 'times':   (B, T) or (B, T, 1)     -- time points
            - 'targets': (B, T, D)               -- true velocities or states (used to infer D)
            - 'mask':    (B, T, D) or similar    -- unused here

        We assume the first D features of 'inputs' are the state x_t
        (if your inputs are just x_t, that's automatically true).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vf_net = vf_net.to(device)
    vf_net.eval()

    if score_model!=None:
        score_model = score_model.to(device)
        score_model.eval()

    # ----- get one batch & pick a trajectory -----
    batch = next(iter(test_loader))
    inputs  = batch['inputs'].to(device)   # (B, T, F or D)
    times   = batch['times'].to(device)    # (B, T) or (B, T, 1)
    targets = batch['targets'].to(device)  # (B, T, D)
    # mask    = batch['mask'].to(device)   # not used for plotting

    times_normalized  = batch['original_times_normalized'].to(device)    # (B, T) or (B, T, 1)
    values_normalized  = batch['original_values_normalized'].to(device) 

    num_test_trajectories= inputs.shape[0]

    random.seed(2025)
    # can't draw more than what's present in the dataset
    if num_traj_to_draw>= num_test_trajectories:
        traj_indices= [i for i in range(num_test_trajectories)]
    else:
        traj_indices = [random.randint(0, num_test_trajectories) for _ in range(num_traj_to_draw)]


    for traj_index in traj_indices:

        # choose which trajectory in the batch to visualize

        # inputs_traj  = inputs[traj_index]      # (T, F or D)
        # times_traj   = times[traj_index]       # (T,) or (T,1)
        
        inputs_traj= values_normalized[traj_index]
        times_traj= times_normalized[traj_index]
        targets_traj = targets[traj_index]     # (T, D)

        # ensure times is 1D: (T,)
        if times_traj.dim() == 2 and times_traj.size(-1) == 1:
            times_traj = times_traj.squeeze(-1)

        # infer state dimension from targets (derivative/state dim)
        state_dim = targets_traj.shape[-1]

        # assume first state_dim features of inputs are the actual state x_t
        # if inputs is already just state, this is a no-op
        if inputs_traj.shape[-1] > state_dim:
            x_traj_true = inputs_traj[..., :state_dim]   # (T, D)
        else:
            x_traj_true = inputs_traj                    # (T, D)

        # initial condition for NODE: x(t0)
        x0 = x_traj_true[0].unsqueeze(0)  # (1, D)

        # time grid for integration
        t_eval = times_traj               # (T,)

        # ----- build NODE func wrapper -----
        node_func = NODEFunc(vf_net, stats).to(device)
        if score_model!=None:
            sde_func= StochasticNODE(vf_net, score_model, stats, sigma)

        # ----- integrate -----
        with torch.no_grad():
            if score_model==None:
                x_pred = odeint(node_func, x0, t_eval, rtol=1e-6, atol=1e-6)
            else:
                x_pred= sdeint(sde_func, x0, t_eval)
            

        x_pred = x_pred[:, 0, :].cpu().numpy()          # (T, D)
        x_true = x_traj_true.cpu().numpy()              # (T, D)
        t_np   = t_eval.cpu().numpy()                   # (T,)

        # ----- plotting -----
        D = state_dim

        if D == 1:
            # 1D: just plot x(t)
            plt.figure(figsize=(6, 4))
            plt.plot(t_np, x_true[:, 0], label="true x")
            plt.plot(t_np, x_pred[:, 0], "--", label="pred x")
            plt.xlabel("t")
            plt.ylabel("x")
            plt.legend()
            plt.tight_layout()

        elif D == 2:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            # x1(t)
            axes[0].plot(t_np, x_true[:, 0], label="true x1")
            axes[0].plot(t_np, x_pred[:, 0], "--", label="pred x1")
            axes[0].set_xlabel("t")
            axes[0].set_ylabel("x1")
            axes[0].legend()

            # x2(t)
            axes[1].plot(t_np, x_true[:, 1], label="true x2")
            axes[1].plot(t_np, x_pred[:, 1], "--", label="pred x2")
            axes[1].set_xlabel("t")
            axes[1].set_ylabel("x2")
            axes[1].legend()

            # phase portrait
            axes[2].plot(x_true[:, 0], x_true[:, 1], label="true traj")
            axes[2].plot(x_pred[:, 0], x_pred[:, 1], "--", label="pred traj")
            axes[2].set_xlabel("x1")
            axes[2].set_ylabel("x2")
            axes[2].legend()

            fig.tight_layout()

        elif D == 3:
            fig = plt.figure(figsize=(14, 6))

            # time series for each dim
            ax1 = fig.add_subplot(2, 3, 1)
            ax2 = fig.add_subplot(2, 3, 2)
            ax3 = fig.add_subplot(2, 3, 3)
            ax3d = fig.add_subplot(2, 3, (4, 6), projection='3d')

            ax1.plot(t_np, x_true[:, 0], label="true x1")
            ax1.plot(t_np, x_pred[:, 0], "--", label="pred x1")
            ax1.set_xlabel("t")
            ax1.set_ylabel("x1")
            ax1.legend()

            ax2.plot(t_np, x_true[:, 1], label="true x2")
            ax2.plot(t_np, x_pred[:, 1], "--", label="pred x2")
            ax2.set_xlabel("t")
            ax2.set_ylabel("x2")
            ax2.legend()

            ax3.plot(t_np, x_true[:, 2], label="true x3")
            ax3.plot(t_np, x_pred[:, 2], "--", label="pred x3")
            ax3.set_xlabel("t")
            ax3.set_ylabel("x3")
            ax3.legend()

            # 3D phase trajectory
            ax3d.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], label="true traj")
            ax3d.plot(x_pred[:, 0], x_pred[:, 1], x_pred[:, 2], "--", label="pred traj")
            ax3d.set_xlabel("x1")
            ax3d.set_ylabel("x2")
            ax3d.set_zlabel("x3")
            ax3d.legend()

            fig.tight_layout()

        else:
            # D > 3: just do D time-series subplots
            fig, axes = plt.subplots(D, 1, figsize=(8, 2.5 * D), sharex=True)
            if D == 1:
                axes = [axes]
            for d in range(D):
                ax = axes[d]
                ax.plot(t_np, x_true[:, d], label=f"true x{d+1}")
                ax.plot(t_np, x_pred[:, d], "--", label=f"pred x{d+1}")
                ax.set_ylabel(f"x{d+1}")
                ax.legend()
            axes[-1].set_xlabel("t")
            fig.tight_layout()

        traj_dir= os.path.join(save_dir, 'figures')
        if traj_dir is not None:
            os.makedirs(traj_dir, exist_ok=True)
            fname = os.path.join(traj_dir, f"trajectory_dim:{D}_traj:{traj_index}.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()




def validate_metrics(model, score_model, test_loader,  dynamics_kind, stats, sigma, save_dir, subsample_per_interval, metrics_to_run= ["mse"], trajectories_png=True, num_traj_to_draw=10):

    if trajectories_png==True:
        # extend to include multiple trajectories
        draw_trajectories(model, score_model, dynamics_kind, test_loader, stats=stats, sigma=sigma, save_dir=save_dir, num_traj_to_draw=num_traj_to_draw)

    metric_dicts= {}
    for metric in metrics_to_run:
        if metric=="mse":
            metric_val= mse_of_predicted_traj(model, test_loader, stats, subsample_per_interval)
            metric_dicts["mse"]= metric_val

            if score_model is not None or "sde" in dynamics_kind:
                stochastic_mse_val, wasserstein_val, mmd_val, energy_val= metrics_of_predicted_stochastic_traj(model, score_model, dynamics_kind, test_loader, stats, sigma, subsample_per_interval)
                metric_dicts["stochastic_mse_with_score_model"]= stochastic_mse_val
                metric_dicts["wasserstein_with_score_model"]= wasserstein_val
                metric_dicts["mmd_with_score_model"]= mmd_val
                metric_dicts["energy_with_score_model"]= energy_val
        elif metric=="wasserstein":
            pass
    return metric_dicts
