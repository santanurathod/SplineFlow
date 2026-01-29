import json
import torch
import os
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from models.NN_models import MLP
# from src.interpolants import linear_interpolant, linear_interpolant_velocity_field, bspline_interpolant, bspline_interpolant_velocity_field, cubic_interpolant, cubic_interpolant_velocity_field, lagrange_interpolant, lagrange_interpolant_velocity_field
from src.interpolants import linear_interpolant_values, bspline_interpolant_values


def get_model(model_config, data_dim, input_dim, device, dynamics_kind='ode'):
    model_config = json.load(open(f"/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/configs/model_configs/{model_config}.json"))
    model_type= model_config["model_type"]

    if model_type == "MLP" and dynamics_kind=='ode':
        hidden_dim = model_config["hidden_dim"]
        num_layers = model_config["num_layers"]
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=data_dim, num_layers=num_layers).to(device)

    elif model_type == "MLP" and dynamics_kind!='ode': # for the score model, for now, I'm giving little complex model since we need to fit random noise
        hidden_dim = model_config["hidden_dim"]
        num_layers = 2*model_config["num_layers"]
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=data_dim, num_layers=num_layers).to(device)
    else:
        raise ValueError(f"Model type {model_type} is not implemented yet")
    return model


# def get_flow_matching_inputs(values, times, mask, interpolant_kind="linear", degree=3, subsample_per_interval=32):

#     # import pdb; pdb.set_trace()
#     if interpolant_kind == "linear":
#         print('values_interpolated')
#         values_interpolated = linear_interpolant(values, times, mask)
#         print('velocity_calc')
#         xt, vt, t = linear_interpolant_velocity_field(values_interpolated, times, mask, subsample_per_interval)
#     elif interpolant_kind == "lagrange":
#         print("Lagrange interpolant not implemented yet")
#         raise NotImplementedError
#     elif interpolant_kind == "bspline":
#         print('values_interpolated')
#         values_interpolated = bspline_interpolant(values, times, mask, degree= degree)
#         print('velocity_calc')
#         xt, vt, t = bspline_interpolant_velocity_field(values_interpolated, times, mask, subsample_per_interval= subsample_per_interval, degree= degree)
#     elif interpolant_kind == "cubic":
#         print("Cubic interpolant not implemented yet")
#         raise NotImplementedError
#     else:
#         raise ValueError(f"Interpolant kind {interpolant_kind} not supported.")

    
#     inputs = xt
#     targets = vt
#     mask = mask
    

#     return inputs, t, targets, mask


def get_flow_matching_inputs(values, times, mask, interpolant_kind="linear", degree=3, subsample_per_interval=32, dynamics_kind='ode', sigma=0.001):

    if interpolant_kind == "linear":
        # xt, vt, t, score, lambda_t, eps_xt, eps_score = linear_interpolant_values(values, times, mask, subsample_per_interval, dynamics_kind=dynamics_kind, sigma=sigma)
        xt, vt, t, sigma_t, lambda_t, der_sigma_t = linear_interpolant_values(values, times, mask, subsample_per_interval, dynamics_kind=dynamics_kind, sigma=sigma)
    elif interpolant_kind == "lagrange":
        print("Lagrange interpolant not implemented yet")
        raise NotImplementedError
    elif interpolant_kind == "bspline":
        # xt, vt, t, score, lambda_t, eps_xt, eps_score = bspline_interpolant_values(values, times, mask, subsample_per_interval= subsample_per_interval, degree= degree, dynamics_kind= dynamics_kind, sigma=sigma)
        xt, vt, t, sigma_t, lambda_t, der_sigma_t = bspline_interpolant_values(values, times, mask, subsample_per_interval= subsample_per_interval, degree= degree, dynamics_kind= dynamics_kind, sigma=sigma)
    elif interpolant_kind == "cubic":
        print("Cubic interpolant not implemented yet")
        raise NotImplementedError
    else:
        raise ValueError(f"Interpolant kind {interpolant_kind} not supported.")

    
    inputs = xt
    targets = vt
    mask = mask


    # return inputs, t, targets, mask, score, lambda_t, eps_xt, eps_score
    return inputs, t, targets, mask, sigma_t, lambda_t, der_sigma_t



def get_sde_terms():
    '''
    returns score, noise, other misc terms required for SDE implementation
    '''
    pass



def get_data_stats(items):

    values= torch.concatenate([x[0] for x in items])
    times= torch.concatenate([x[1] for x in items])
    targets= torch.concatenate([x[2] for x in items])

    data_dim= values.shape[-1]

    values_mean= values.reshape(-1, data_dim).mean(axis=0, keepdims=True)
    values_std= values.reshape(-1, data_dim).std(axis=0, keepdims=True)+1e-8

    times_mean= times.reshape(-1, 1).mean(axis=0, keepdims=True)
    times_std= times.reshape(-1, 1).std(axis=0, keepdims=True)+1e-8

    targets_mean= targets.reshape(-1, data_dim).mean(axis=0, keepdims=True)
    targets_std= targets.reshape(-1, data_dim).std(axis=0, keepdims=True)+1e-8

    
    return { 'values_mean': values_mean,
    'values_std': values_std, 
    'times_mean': times_mean,
    'times_std': times_std,
    'targets_mean':targets_mean,
    'targets_std':targets_std

    }



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_trajectories(vf_net, test_loader, NODEFunc, stats, save_dir=None, traj_index=0):
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

    # ----- get one batch & pick a trajectory -----
    batch = next(iter(test_loader))
    inputs  = batch['inputs'].to(device)   # (B, T, F or D)
    times   = batch['times'].to(device)    # (B, T) or (B, T, 1)
    targets = batch['targets'].to(device)  # (B, T, D)
    # mask    = batch['mask'].to(device)   # not used for plotting

    # choose which trajectory in the batch to visualize
    inputs_traj  = inputs[traj_index]      # (T, F or D)
    times_traj   = times[traj_index]       # (T,) or (T,1)
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

    # ----- integrate -----
    with torch.no_grad():
        x_pred = odeint(node_func, x0, t_eval, rtol=1e-6, atol=1e-6)
        # x_pred: (T, 1, D)

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

    # ----- save or show -----
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"trajectory_D{D}_traj{traj_index}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
