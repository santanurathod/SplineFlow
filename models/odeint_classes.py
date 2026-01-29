import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class NODEFunc(nn.Module):
    def __init__(self, vf_net, stats):
        """
        Wrap the learned velocity field with the normalization used in the dataset.
        stats must contain values_std, targets_mean, targets_std, times_std.
        """
        super().__init__()
        self.vf_net = vf_net
        device = next(vf_net.parameters()).device
        # stats tensors were computed in ProcessedDataset; move and register them
        self.register_buffer("values_std", stats["values_std"].to(device))
        self.register_buffer("targets_mean", stats["targets_mean"].to(device))
        self.register_buffer("targets_std", stats["targets_std"].to(device))
        self.register_buffer("times_std", stats["times_std"].to(device))

        # precompute the chain-rule scale factor times_std/values_std
        self.register_buffer("norm_velocity_scale", (self.times_std / self.values_std))

    def forward(self, t, x):
        """
        t: scalar tensor (normalized)
        x: (batch, dim) normalized state
        returns: (batch, dim) normalized dx/dt for the solver
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, dim)

        batch_size = x.shape[0]
        t = t.expand(batch_size, 1)

        inp = torch.cat([x, t], dim=-1)
        vf_pred_norm = self.vf_net(inp)  # z-scored raw velocity
        vf_raw = vf_pred_norm * self.targets_std + self.targets_mean  # undo target z-score
        # chain rule: dx_norm/dt_norm = (dx/dt_raw) * (times_std / values_std)
        vf_norm = vf_raw * self.norm_velocity_scale
        return vf_norm


class StochasticNODE(torch.nn.Module):
    noise_type, sde_type = "diagonal", "ito"
    def __init__(self, vf_net, score_net, stats, sigma):
        super().__init__()
        self.vf, self.score, self.sigma = vf_net, score_net, sigma
        self.register_buffer("values_std", stats["values_std"])
        self.register_buffer("values_mean", stats["values_mean"])
        self.register_buffer("targets_std", stats["targets_std"])
        self.register_buffer("targets_mean", stats["targets_mean"])
        self.register_buffer("times_std", stats["times_std"])
        self.register_buffer("times_mean", stats["times_mean"])
        self.register_buffer("norm_scale", self.times_std / self.values_std)

    def f(self, t, x_norm):
        if x_norm.dim() == 1: x_norm = x_norm.unsqueeze(0)
        t_in = t.expand(x_norm.size(0), 1)
        inp = torch.cat([x_norm, t_in], dim=-1)
        vf_pred_norm = self.vf(inp)                   # trained on normalized targets
        vf_raw = vf_pred_norm * self.targets_std + self.targets_mean
        vf_norm = vf_raw * self.norm_scale            # chain rule
        score_norm = self.score(inp)                  # already normalized scale
        return vf_norm + score_norm                   # drift

    def g(self, t, x_norm):
        return torch.ones_like(x_norm) * self.sigma * self.norm_scale